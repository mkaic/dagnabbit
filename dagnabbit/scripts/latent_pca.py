"""How low-dimensional is the graph latent space?

Encodes a batch of freshly sampled random graphs (drawn exactly the way the
training loop draws them) and asks three questions about the resulting
``[N, K, D]`` latents:

1. **Linear spectrum** -- PCA over the flattened ``K*D`` latent. Reports the
   eigenvalue spectrum, its participation ratio (a threshold-free effective
   dimensionality), and the component counts needed for 50/90/95/99% of the
   variance.
2. **Token structure** -- whether the K latent tokens span one shared subspace
   or K independent ones, via per-token spectra and the mean cross-token
   correlation. A latent whose tokens are near-copies of each other is
   really D-dimensional, not K*D-dimensional.
3. **Functional truncation cost** -- the question PCA variance alone cannot
   answer. For a sweep of ranks k, held-out latents are projected onto the
   top-k principal subspace, decoded, and scored against the graph they came
   from (trunk-type accuracy, parent-pointer accuracy, exact-match rate). A
   rank that keeps 99% of the variance is useless if it cannot round-trip a
   graph; this is what says whether searching in a k-dim subspace is viable.

Model geometry is read off the checkpoint's state dict, not ``config.py``,
since the config tracks the newest run and drifts away from older checkpoints.

Usage::

    python -m dagnabbit.scripts.latent_pca --checkpoint 'runs/<run>/best.ckpt'
    python -m dagnabbit.scripts.latent_pca --checkpoint runs/<run> --num-graphs 8192
"""

import argparse
import re
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from dagnabbit.dag.autoencoder import DagnabbitAutoEncoder
from dagnabbit.dag.description import (
    FixedInDegreeDAGDescription,
    graphs_match,
    make_random_graph_description,
)
from dagnabbit.scripts import config as cfg

VARIANCE_THRESHOLDS = (0.50, 0.90, 0.95, 0.99)
DEFAULT_TRUNCATION_RANKS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        required=True,
        help=(
            "Checkpoint to analyze: a .ckpt file or a run directory (its "
            "best.ckpt is preferred, falling back to latest.ckpt)."
        ),
    )
    parser.add_argument(
        "--num-graphs",
        type=int,
        default=4096,
        help=(
            "Random graphs to encode. The flattened covariance has K*D "
            "dimensions, so the spectrum is rank-limited until this exceeds "
            "it comfortably."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Graphs per encoder forward pass (default: cfg.GRAPH_BATCH_SIZE).",
    )
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.2,
        help=(
            "Fraction of encoded graphs held out of the PCA fit and used for "
            "the truncation sweep, so truncation numbers are out-of-sample."
        ),
    )
    parser.add_argument(
        "--truncation-graphs",
        type=int,
        default=256,
        help="Held-out graphs to decode per truncation rank.",
    )
    parser.add_argument(
        "--skip-truncation",
        action="store_true",
        help="Only report the spectrum; skip the decode-and-score sweep.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help=(
            "torch device ('auto' picks cuda, else cpu). 'mps' is not chosen "
            "automatically: the encoder's per-rank gather returns garbage "
            "indices there."
        ),
    )
    parser.add_argument(
        "--save-latents",
        action="store_true",
        help=(
            "Include the raw [N, K, D] latents in the .npz (a few hundred MB "
            "at the default sample count)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=cfg.SEED,
        help="Seed for graph sampling.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory for spectrum.npz and spectrum.png "
            "(default: the checkpoint's directory)."
        ),
    )
    return parser.parse_args()


def resolve_checkpoint(argument: str) -> Path:
    path = Path(argument)
    if path.is_dir():
        best = path / "best.ckpt"
        resolved = best if best.exists() else path / "latest.ckpt"
    else:
        resolved = path
    if not resolved.exists():
        raise FileNotFoundError(f"checkpoint not found: {resolved}")
    return resolved


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    # MPS is deliberately not auto-selected: the recursive encoder's rank
    # gather produces out-of-bounds indices there, so it is opt-in only.
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_blocks(state_dict: dict, prefix: str) -> int:
    pattern = re.compile(re.escape(prefix) + r"\.blocks\.(\d+)\.")
    indices = {
        int(match.group(1))
        for key in state_dict
        if (match := pattern.match(key)) is not None
    }
    return len(indices)


def build_model_from_checkpoint(
    state_dict: dict,
    device: torch.device,
) -> DagnabbitAutoEncoder:
    """Reconstruct the model geometry the checkpoint was trained with.

    Everything the weights determine is read from them; only the trunk/output
    node counts and the per-type in-degrees come from ``config.py``, because
    the tensors that encode those (position encodings, the pointer candidate
    mask) are non-persistent buffers and never hit the state dict. The
    in-degree list is validated against the number of slot query projections.
    """
    node_embedding_dim = state_dict["mask_token"].shape[0]
    num_root_nodes = state_dict["root_node_embeddings.weight"].shape[0]
    num_trunk_node_types = state_dict["node_type_predictor.weight"].shape[0]
    num_slot_projs = len(
        {
            key.split(".")[1]
            for key in state_dict
            if key.startswith("pointer_slot_query_projs.")
        }
    )

    in_degrees = cfg.TRUNK_NODE_TYPE_IN_DEGREES
    if isinstance(in_degrees, int):
        in_degrees = [in_degrees] * num_trunk_node_types
    if max([1, *in_degrees]) != num_slot_projs:
        raise ValueError(
            f"checkpoint has {num_slot_projs} pointer slot projections, but "
            f"cfg.TRUNK_NODE_TYPE_IN_DEGREES={cfg.TRUNK_NODE_TYPE_IN_DEGREES} "
            f"implies a maximum in-degree of {max([1, *in_degrees])}"
        )

    model = DagnabbitAutoEncoder(
        node_embedding_dim=node_embedding_dim,
        trunk_node_type_in_degrees=in_degrees,
        num_trunk_node_types=num_trunk_node_types,
        num_root_nodes=num_root_nodes,
        num_trunk_nodes=cfg.NUM_TRUNK_NODES,
        num_output_nodes=cfg.NUM_OUTPUT_NODES,
        mlp_expansion_factor=cfg.MLP_EXPANSION_FACTOR,
        encoder_num_layers=count_blocks(
            state_dict, "node_encoder.sequence_transformer"
        ),
        compressor_num_layers=count_blocks(state_dict, "compressor"),
        decoder_num_layers=count_blocks(state_dict, "decoder"),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def sample_graphs(
    model: DagnabbitAutoEncoder,
    count: int,
) -> list[FixedInDegreeDAGDescription]:
    """Draw graphs the way the training loop does, at the model's geometry."""
    return [
        make_random_graph_description(
            num_root_nodes=model.num_root_nodes,
            num_trunk_nodes=model.num_trunk_nodes,
            num_output_nodes=model.num_output_nodes,
            trunk_node_in_degrees=model.trunk_node_in_degrees,
            num_trunk_node_types=model.num_trunk_node_types,
        )
        for _ in range(count)
    ]


def encode_dataset(
    model: DagnabbitAutoEncoder,
    num_graphs: int,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[FixedInDegreeDAGDescription]]:
    """Sample and encode ``num_graphs`` graphs; returns [N, K, D] latents."""
    latents: list[torch.Tensor] = []
    graphs: list[FixedInDegreeDAGDescription] = []
    with torch.no_grad():
        with tqdm(total=num_graphs, unit="graph", desc="encoding") as progress:
            while len(graphs) < num_graphs:
                count = min(batch_size, num_graphs - len(graphs))
                batch = sample_graphs(model, count)
                latents.append(model.encode_to_latent(batch).float().cpu())
                graphs.extend(batch)
                progress.update(count)
    return torch.cat(latents), graphs


def spectrum_of(matrix: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """PCA eigenvalues and component basis of a [N, D] matrix.

    Returns eigenvalues (descending, as variances) and the [D, D_rank]
    component matrix whose columns are the principal directions.
    """
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    # SVD of the centered data is more numerically stable than eigendecomposing
    # an explicitly formed covariance, and we want the basis anyway.
    _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    eigenvalues = (singular_values**2) / max(centered.shape[0] - 1, 1)
    return eigenvalues.numpy(), vh.T


def participation_ratio(eigenvalues: np.ndarray) -> float:
    """(sum lambda)^2 / sum(lambda^2): effective dimensionality, no threshold.

    Equals D for a perfectly isotropic spectrum and 1 when a single direction
    carries all the variance.
    """
    total = eigenvalues.sum()
    if total <= 0:
        return 0.0
    return float(total**2 / (eigenvalues**2).sum())


def components_for_thresholds(
    eigenvalues: np.ndarray,
    thresholds: tuple[float, ...],
) -> dict[float, int]:
    cumulative = np.cumsum(eigenvalues) / eigenvalues.sum()
    return {
        threshold: int(np.searchsorted(cumulative, threshold) + 1)
        for threshold in thresholds
    }


def powerlaw_exponent(eigenvalues: np.ndarray) -> float:
    """Slope of log(lambda) vs log(rank) over the spectrum's first decade.

    Natural-image-like representations sit near -1; a much steeper slope means
    the variance collapses into a handful of directions.
    """
    usable = eigenvalues[eigenvalues > 0]
    upper = min(len(usable), max(10, len(usable) // 10))
    if upper < 3:
        return float("nan")
    ranks = np.arange(1, upper + 1, dtype=np.float64)
    slope, _ = np.polyfit(np.log(ranks), np.log(usable[:upper]), 1)
    return float(slope)


def report_spectrum(name: str, eigenvalues: np.ndarray, ambient_dim: int) -> None:
    counts = components_for_thresholds(eigenvalues, VARIANCE_THRESHOLDS)
    ratio = participation_ratio(eigenvalues)
    print(f"\n--- {name} (ambient dim {ambient_dim}) ---")
    print(f"  participation ratio       {ratio:.1f}  ({ratio / ambient_dim:.1%} of ambient)")
    print(f"  log-log spectrum slope    {powerlaw_exponent(eigenvalues):.2f}")
    for threshold, count in counts.items():
        print(
            f"  components for {threshold:>4.0%} var   {count:>5d}"
            f"  ({count / ambient_dim:.1%} of ambient)"
        )
    top = eigenvalues[:5] / eigenvalues.sum()
    print("  top-5 variance shares     " + ", ".join(f"{share:.3f}" for share in top))


def cross_token_correlation(latents: torch.Tensor) -> float:
    """Mean |correlation| between distinct latent tokens across graphs.

    Each token is flattened to a per-graph vector; a value near 1 means the K
    tokens carry redundant copies of the same information.
    """
    num_graphs, num_tokens, _ = latents.shape
    flat = latents.reshape(num_graphs, num_tokens, -1)
    centered = flat - flat.mean(dim=0, keepdim=True)
    normalized = centered / centered.norm(dim=(0, 2), keepdim=True).clamp(min=1e-8)
    gram = torch.einsum("ntd,nsd->ts", normalized, normalized).abs()
    off_diagonal = ~torch.eye(num_tokens, dtype=torch.bool)
    return float(gram[off_diagonal].mean())


@torch.no_grad()
def truncation_scores(
    model: DagnabbitAutoEncoder,
    latents: torch.Tensor,
    graphs: list[FixedInDegreeDAGDescription],
    components: torch.Tensor,
    mean: torch.Tensor,
    rank: int | None,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    """Decode latents (optionally rank-truncated) and score against the truth.

    ``rank=None`` decodes the untruncated latents, giving the ceiling that the
    truncated ranks are competing against.
    """
    num_graphs, num_tokens, embedding_dim = latents.shape
    type_correct = 0
    type_total = 0
    pointer_correct = 0
    pointer_total = 0
    exact_matches = 0

    for start in range(0, num_graphs, batch_size):
        stop = min(start + batch_size, num_graphs)
        batch_graphs = graphs[start:stop]
        flat = latents[start:stop].reshape(stop - start, -1)
        if rank is not None:
            basis = components[:, :rank]
            flat = (flat - mean) @ basis @ basis.T + mean
        batch_latent = flat.reshape(-1, num_tokens, embedding_dim).to(device)

        reconstructed = model.decode_latent(batch_latent)
        predicted_types = model.node_type_predictor(
            reconstructed[:, model.num_root_nodes : model.output_start]
        ).argmax(dim=-1)
        predicted_parents = model.parent_pointer_logits(reconstructed).argmax(dim=-1)

        true_types = torch.stack(
            [graph.canonical_node_types for graph in batch_graphs]
        ).to(device)[:, model.num_root_nodes : model.output_start]
        true_parents = torch.stack(
            [graph.canonical_parent_positions for graph in batch_graphs]
        ).to(device)
        slot_mask = torch.stack(
            [graph.canonical_parent_slot_mask for graph in batch_graphs]
        ).to(device)

        type_correct += int((predicted_types == true_types).sum())
        type_total += true_types.numel()
        pointer_hits = (predicted_parents == true_parents) & slot_mask
        pointer_correct += int(pointer_hits.sum())
        pointer_total += int(slot_mask.sum())

        rebuilt = model.generate(batch_latent)
        exact_matches += sum(
            graphs_match(graph, candidate)
            for graph, candidate in zip(batch_graphs, rebuilt)
        )

    return {
        "type_accuracy": type_correct / max(type_total, 1),
        "pointer_accuracy": pointer_correct / max(pointer_total, 1),
        "exact_match_rate": exact_matches / max(num_graphs, 1),
    }


def write_artifacts(
    output_dir: Path,
    eigenvalues: np.ndarray,
    per_token_eigenvalues: list[np.ndarray],
    latents: torch.Tensor | None,
    sweep: list[tuple[int | None, dict[str, float]]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "latent_pca.npz"
    arrays = {} if latents is None else {"latents": latents.numpy()}
    np.savez_compressed(
        npz_path,
        **arrays,
        eigenvalues=eigenvalues,
        per_token_eigenvalues=np.stack(per_token_eigenvalues),
        truncation_ranks=np.array(
            [-1 if rank is None else rank for rank, _ in sweep]
        ),
        truncation_type_accuracy=np.array(
            [scores["type_accuracy"] for _, scores in sweep]
        ),
        truncation_pointer_accuracy=np.array(
            [scores["pointer_accuracy"] for _, scores in sweep]
        ),
        truncation_exact_match=np.array(
            [scores["exact_match_rate"] for _, scores in sweep]
        ),
    )
    print(f"\nwrote {npz_path}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: skipping spectrum.png")
        return

    cumulative = np.cumsum(eigenvalues) / eigenvalues.sum()
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    ranks = np.arange(1, len(eigenvalues) + 1)
    axes[0].loglog(ranks, eigenvalues / eigenvalues.sum())
    axes[0].set(
        xlabel="component rank",
        ylabel="variance share",
        title="Spectrum (log-log)",
    )
    axes[0].grid(alpha=0.3, which="both")

    axes[1].semilogx(ranks, cumulative)
    for threshold in VARIANCE_THRESHOLDS:
        axes[1].axhline(threshold, color="grey", linestyle=":", linewidth=0.8)
    axes[1].set(
        xlabel="components kept",
        ylabel="cumulative variance",
        title="Cumulative variance",
        ylim=(0, 1.02),
    )
    axes[1].grid(alpha=0.3, which="both")

    truncated = [(rank, scores) for rank, scores in sweep if rank is not None]
    if truncated:
        kept = [rank for rank, _ in truncated]
        for key, label in (
            ("type_accuracy", "trunk type acc"),
            ("pointer_accuracy", "pointer acc"),
            ("exact_match_rate", "exact match"),
        ):
            axes[2].semilogx(kept, [scores[key] for _, scores in truncated], marker="o", label=label)
        full = next((scores for rank, scores in sweep if rank is None), None)
        if full is not None:
            axes[2].axhline(
                full["pointer_accuracy"],
                color="grey",
                linestyle="--",
                linewidth=0.8,
                label="pointer acc, untruncated",
            )
        axes[2].set(
            xlabel="components kept",
            ylabel="accuracy",
            title="Decode quality vs. PCA rank",
            ylim=(0, 1.02),
        )
        axes[2].legend(fontsize=8)
        axes[2].grid(alpha=0.3, which="both")

    figure.tight_layout()
    png_path = output_dir / "latent_pca.png"
    figure.savefig(png_path, dpi=140)
    print(f"wrote {png_path}")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = resolve_device(args.device)
    checkpoint_path = resolve_checkpoint(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model_from_checkpoint(checkpoint["model_state_dict"], device)

    latent_dim = model.num_output_nodes * model.node_embedding_dim
    print(f"checkpoint    {checkpoint_path}")
    print(f"graphs_seen   {checkpoint.get('graphs_seen')}  loss {checkpoint.get('loss')}")
    print(f"device        {device}")
    print(
        f"geometry      D={model.node_embedding_dim} roots={model.num_root_nodes} "
        f"trunk={model.num_trunk_nodes} outputs={model.num_output_nodes} "
        f"nodes={model.num_nodes}"
    )
    print(
        f"              encoder={model.encoder_num_layers}L "
        f"compressor={model.compressor_num_layers}L "
        f"decoder={model.decoder_num_layers}L"
    )
    print(
        f"latent        {model.num_output_nodes} x {model.node_embedding_dim} "
        f"= {latent_dim} dims"
    )
    if model.node_embedding_dim != cfg.NODE_EMBEDDING_DIM:
        print(
            f"NOTE: checkpoint D={model.node_embedding_dim} but "
            f"cfg.NODE_EMBEDDING_DIM={cfg.NODE_EMBEDDING_DIM}; using the "
            "checkpoint's geometry."
        )

    batch_size = args.batch_size or cfg.GRAPH_BATCH_SIZE
    if args.num_graphs < 2 * latent_dim:
        print(
            f"\nWARNING: {args.num_graphs} graphs for a {latent_dim}-dim "
            f"covariance. The spectrum is rank-limited to "
            f"{args.num_graphs - 1} components and the tail is dominated by "
            "sampling noise; pass --num-graphs >= "
            f"{2 * latent_dim} for a trustworthy tail."
        )

    latents, graphs = encode_dataset(model, args.num_graphs, batch_size, device)

    norms = latents.reshape(len(graphs), -1).norm(dim=1)
    per_token_norms = latents.norm(dim=2)
    print(
        f"\nlatent norm   mean {norms.mean():.3f}  std {norms.std():.3f}   "
        f"per-token mean {per_token_norms.mean():.3f} "
        f"(sqrt(D)={model.node_embedding_dim ** 0.5:.1f})"
    )

    holdout_size = int(len(graphs) * args.holdout_fraction)
    fit_size = len(graphs) - holdout_size
    if fit_size < 2:
        raise ValueError("not enough graphs left to fit PCA after the holdout split")
    fit_latents = latents[:fit_size]
    holdout_latents = latents[fit_size:]
    holdout_graphs = graphs[fit_size:]
    print(f"pca fit on    {fit_size} graphs, {holdout_size} held out")

    flat_fit = fit_latents.reshape(fit_size, -1)
    eigenvalues, components = spectrum_of(flat_fit)
    report_spectrum("flattened latent (the full search space)", eigenvalues, latent_dim)

    per_token_eigenvalues = []
    print(f"\n--- per-token spectra (ambient dim {model.node_embedding_dim}) ---")
    for token in range(model.num_output_nodes):
        token_eigenvalues, _ = spectrum_of(fit_latents[:, token])
        per_token_eigenvalues.append(token_eigenvalues)
        counts = components_for_thresholds(token_eigenvalues, (0.95,))
        print(
            f"  token {token}  participation ratio "
            f"{participation_ratio(token_eigenvalues):>7.1f}   "
            f"components for 95% var {counts[0.95]:>4d}"
        )
    pooled_eigenvalues, _ = spectrum_of(
        fit_latents.reshape(-1, model.node_embedding_dim)
    )
    report_spectrum(
        "tokens pooled (all K treated as samples)",
        pooled_eigenvalues,
        model.node_embedding_dim,
    )
    print(
        f"\nmean |cross-token correlation|  {cross_token_correlation(fit_latents):.3f}"
        "   (near 1 => the K tokens are redundant copies)"
    )

    sweep: list[tuple[int | None, dict[str, float]]] = []
    if not args.skip_truncation:
        truncation_count = min(args.truncation_graphs, holdout_size)
        if truncation_count < 1:
            print("\nno held-out graphs: skipping the truncation sweep")
        else:
            eval_latents = holdout_latents[:truncation_count]
            eval_graphs = holdout_graphs[:truncation_count]
            mean = flat_fit.mean(dim=0, keepdim=True)
            max_rank = min(components.shape[1], latent_dim)
            ranks: list[int | None] = [
                rank for rank in DEFAULT_TRUNCATION_RANKS if rank <= max_rank
            ]
            ranks.append(None)

            print(
                f"\n--- decode quality vs. PCA rank "
                f"({truncation_count} held-out graphs) ---"
            )
            print(f"  {'rank':>6}  {'type acc':>9}  {'ptr acc':>9}  {'exact':>7}")
            for rank in ranks:
                scores = truncation_scores(
                    model=model,
                    latents=eval_latents,
                    graphs=eval_graphs,
                    components=components,
                    mean=mean,
                    rank=rank,
                    device=device,
                    batch_size=batch_size,
                )
                sweep.append((rank, scores))
                label = "full" if rank is None else str(rank)
                print(
                    f"  {label:>6}  {scores['type_accuracy']:>9.4f}  "
                    f"{scores['pointer_accuracy']:>9.4f}  "
                    f"{scores['exact_match_rate']:>7.4f}",
                    flush=True,
                )

    output_dir = (
        Path(args.output_dir) if args.output_dir else checkpoint_path.parent
    )
    write_artifacts(
        output_dir,
        eigenvalues,
        per_token_eigenvalues,
        latents if args.save_latents else None,
        sweep,
    )


if __name__ == "__main__":
    main()
