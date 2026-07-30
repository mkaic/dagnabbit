"""Precompute a (behaviour image, encoded latent) dataset to disk, once.

The autoencoder this hangs off is **frozen**, so a graph's latent never
changes. Generating random graphs and evaluating them is the documented
bottleneck of the training loop -- so for a stage-two run there is no reason to
do it in the loop at all. Build the dataset once, stream it forever, and the
training step becomes pure tensor work with no Python DAG traversal anywhere in
it.

The images dominate the file. One 256x256x8 behaviour image is 512 KiB as
uint8, against 4 KiB for its ``[8, 128]`` float32 latent, so a million examples
would be half a terabyte. Two things keep that sane:

* Images are stored **bit-packed**, exactly as the evaluator produces them:
  ``[C, num_words]`` uint8, 8 rows per byte. That is the 512 KiB down to 64 KiB
  with no information lost, and :func:`unpack_cached_images` turns a batch back
  into the float image the encoder wants on the fly. Unpacking a batch is
  cheap; it is the *graph evaluation* that was expensive, and that is what has
  been paid for already.
* You almost certainly do not need a million. The latent is ~1000 dimensions
  and the model has ~45M parameters; 200k-500k distinct examples is a sensible
  starting corpus, and one epoch over it is not the unit that matters anyway
  since the sampler sees each example at many blend fractions.

Layout on disk is a directory holding three memmaps plus a JSON manifest, so a
reader can validate shapes and the source checkpoint before trusting a cache.

Usage::

    python -m dagnabbit.scripts.build_latent_cache runs/<run> \\
        --out caches/adder-200k --num-graphs 200000
"""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from dagnabbit.dag.checkpoint import load_model, pick_device, resolve_checkpoint
from dagnabbit.dag.description import make_random_graph_description
from dagnabbit.scripts import config as cfg
from dagnabbit.tasks.logic_gates.evaluate import (
    BitpackedTask,
    adder_task,
    evaluate_graphs,
)
from dagnabbit.tasks.logic_gates.truth_table_image import (
    image_dimensions,
    outputs_to_image,
)

MANIFEST_NAME = "manifest.json"
PACKED_IMAGES_NAME = "packed_behaviours.u8"
LATENTS_NAME = "latents.f32"


@dataclass(frozen=True)
class LatentCache:
    """A memory-mapped (packed behaviour, latent) dataset opened for reading."""

    packed_behaviours: np.memmap  # [N, C, W] uint8
    latents: np.memmap  # [N, K, D] float32
    image_height: int
    image_width: int
    manifest: dict

    def __len__(self) -> int:
        return self.latents.shape[0]

    def batch(
        self,
        indices: np.ndarray,
        device: torch.device,
        gray: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Fetch ``indices`` as ``(images [B, C, H, W] float, latents [B, K, D])``.

        ``indices`` is sorted internally: these are memmap reads, and a sorted
        gather touches far fewer pages than a shuffled one for the same set of
        rows. The returned order therefore is *not* the order asked for, which
        is fine for training batches drawn uniformly at random and would not be
        for anything that needs to line up with an external ordering.
        """
        ordered = np.sort(np.asarray(indices))
        packed = torch.from_numpy(np.ascontiguousarray(self.packed_behaviours[ordered]))
        latents = torch.from_numpy(np.ascontiguousarray(self.latents[ordered]))
        images = unpack_cached_images(
            packed.to(device, non_blocking=True),
            self.image_height,
            self.image_width,
            gray=gray,
        )
        return images, latents.to(device, non_blocking=True)


def unpack_cached_images(
    packed: Tensor,
    image_height: int,
    image_width: int,
    gray: bool = False,
) -> Tensor:
    """``[B, C, W]`` packed bits -> ``[B, C, H, W]`` float image on the same device.

    The inverse of what the cache stores, and deliberately the *same* transform
    :func:`~dagnabbit.tasks.logic_gates.proposer.behaviour_images` applies, so a
    cached example and a freshly generated one are byte-identical inputs.
    """
    return outputs_to_image(packed, image_height, image_width, gray=gray).float()


def open_latent_cache(directory: str | Path) -> LatentCache:
    """Open a cache directory built by this script, validating its manifest."""
    directory = Path(directory)
    manifest_path = directory / MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{directory} is not a latent cache (no {MANIFEST_NAME}); build one "
            "with python -m dagnabbit.scripts.build_latent_cache"
        )
    manifest = json.loads(manifest_path.read_text())
    num_examples = manifest["num_examples"]
    if num_examples <= 0:
        raise ValueError(f"cache at {directory} is empty")

    packed_behaviours = np.memmap(
        directory / PACKED_IMAGES_NAME,
        dtype=np.uint8,
        mode="r",
        shape=(num_examples, manifest["num_output_bits"], manifest["num_words"]),
    )
    latents = np.memmap(
        directory / LATENTS_NAME,
        dtype=np.float32,
        mode="r",
        shape=(num_examples, manifest["num_latent_tokens"], manifest["latent_dim"]),
    )
    return LatentCache(
        packed_behaviours=packed_behaviours,
        latents=latents,
        image_height=manifest["image_height"],
        image_width=manifest["image_width"],
        manifest=manifest,
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="frozen autoencoder .ckpt or run directory")
    parser.add_argument("--out", required=True, help="cache directory to create")
    parser.add_argument("--num-graphs", type=int, default=200_000)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="graphs generated, evaluated and encoded per chunk",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    torch.manual_seed(args.seed)
    device = pick_device(args.device)

    model, checkpoint = load_model(args.checkpoint, device)
    model.requires_grad_(False)
    print(f"frozen autoencoder from step {checkpoint.get('step')} on {device}")

    task: BitpackedTask = adder_task(device)
    image_height, image_width = image_dimensions(task.root_values.shape[0])
    num_output_bits = int(task.target_values.shape[0])
    num_words = int(task.target_values.shape[1])

    out_dir = Path(args.out)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(
            f"{out_dir} already exists and is not empty; pick another --out or "
            "remove it yourself (this script will not overwrite a cache)"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    packed_behaviours = np.memmap(
        out_dir / PACKED_IMAGES_NAME,
        dtype=np.uint8,
        mode="w+",
        shape=(args.num_graphs, num_output_bits, num_words),
    )
    latents = np.memmap(
        out_dir / LATENTS_NAME,
        dtype=np.float32,
        mode="w+",
        shape=(args.num_graphs, model.num_output_nodes, model.node_embedding_dim),
    )

    started = time.perf_counter()
    written = 0
    with tqdm(total=args.num_graphs, unit="graph") as progress:
        while written < args.num_graphs:
            chunk = min(args.batch_size, args.num_graphs - written)
            graphs = [
                make_random_graph_description(
                    num_root_nodes=model.num_root_nodes,
                    num_trunk_nodes=model.num_trunk_nodes,
                    num_output_nodes=model.num_output_nodes,
                    trunk_node_in_degrees=model.trunk_node_in_degrees,
                    num_trunk_node_types=model.num_trunk_node_types,
                )
                for _ in range(chunk)
            ]
            packed = evaluate_graphs(graphs, task)
            encoded = model.encode_to_latent(graphs).float()

            packed_behaviours[written : written + chunk] = packed.cpu().numpy()
            latents[written : written + chunk] = encoded.cpu().numpy()
            written += chunk
            progress.update(chunk)

    packed_behaviours.flush()
    latents.flush()
    elapsed = time.perf_counter() - started

    manifest = {
        "num_examples": args.num_graphs,
        "num_output_bits": num_output_bits,
        "num_words": num_words,
        "image_height": image_height,
        "image_width": image_width,
        "num_latent_tokens": int(model.num_output_nodes),
        "latent_dim": int(model.node_embedding_dim),
        "num_root_nodes": int(model.num_root_nodes),
        "num_trunk_nodes": int(model.num_trunk_nodes),
        "num_trunk_node_types": int(model.num_trunk_node_types),
        "trunk_node_in_degrees": list(model.trunk_node_in_degrees),
        "source_checkpoint": str(resolve_checkpoint(args.checkpoint)),
        "source_step": checkpoint.get("step"),
        "seed": args.seed,
        # Recorded so a reader can tell whether a cache predates a config change
        # that would make its latents meaningless.
        "config_num_output_nodes": cfg.NUM_OUTPUT_NODES,
        "config_node_embedding_dim": cfg.NODE_EMBEDDING_DIM,
    }
    (out_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")

    megabytes = (packed_behaviours.nbytes + latents.nbytes) / 1e6
    print(
        f"wrote {args.num_graphs} examples to {out_dir} "
        f"({megabytes:.0f} MB) in {elapsed:.1f}s "
        f"({args.num_graphs / max(elapsed, 1e-9):.0f} graphs/s)"
    )


if __name__ == "__main__":
    main()
