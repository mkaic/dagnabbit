"""Diagnose why primary *root* nodes stall at ~chance classification accuracy.

Hypothesis (see chat): the decoded representation of every root collapses onto
the *centroid* of the (near-orthogonal) frozen/learned root embeddings, so the
type head sees four nearly-identical inputs -> ~1/NUM_ROOT_NODES accuracy. We
also want to separate two candidate causes:

  (A) identifiability  -- an individual child edge pointing at a root cannot
      tell *which* root it points at, so even a single edge's prediction is
      centroid-ish (no aggregator can fix this), versus
  (B) aggregation      -- individual edges *do* carry root identity, but the
      ``sum / sqrt(count)`` combine over a root's many children washes it out.

To tell them apart we measure root identity recovery at two points:
  * per-edge:   each child's decoded prediction of its root parent, BEFORE
                aggregation (recomputed from the decode-side combined buffer).
  * aggregated: the root's ``combined`` decode embedding, AFTER aggregation
                (exactly what the type head is fed).

Identity is scored head-independently as ``argmax_i cos(pred, root_embedding_i)``
so it does not depend on the (possibly under-trained) type head.

Run::

    uv run python -m dagnabbit.scripts.diagnose_root_collapse \
        --ckpt runs/<run>/best.ckpt --num-graphs 256
"""

from __future__ import annotations

# Keep the math pure-eager so we reproduce the decode exactly.
import os

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dagnabbit.dag.autoencoder import DagnabbitAutoEncoder
from dagnabbit.dag.description import make_random_graph_description
from dagnabbit.scripts import config as cfg


def build_model(device: torch.device) -> DagnabbitAutoEncoder:
    return DagnabbitAutoEncoder(
        node_embedding_dim=cfg.NODE_EMBEDDING_DIM,
        trunk_node_type_in_degrees=cfg.TRUNK_NODE_TYPE_IN_DEGREES,
        num_trunk_node_types=cfg.NUM_TRUNK_NODE_TYPES,
        num_root_nodes=cfg.NUM_ROOT_NODES,
        num_output_nodes=cfg.NUM_OUTPUT_NODES,
        mlp_depth=cfg.MLP_DEPTH,
        mlp_expansion_factor=cfg.MLP_EXPANSION_FACTOR,
        transformer_num_layers=cfg.TRANSFORMER_NUM_LAYERS,
        transformer_mlp_depth=cfg.TRANSFORMER_MLP_DEPTH,
        transformer_num_register_tokens=cfg.TRANSFORMER_NUM_REGISTER_TOKENS,
        transformer_num_heads=cfg.TRANSFORMER_NUM_HEADS,
        transformer_dropout=cfg.TRANSFORMER_DROPOUT,
    ).to(device)


def load_checkpoint(model: DagnabbitAutoEncoder, ckpt_path: Path, device) -> int:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [warn] missing keys: {missing}")
    if unexpected:
        print(f"  [warn] unexpected keys: {unexpected}")
    return int(ckpt.get("step", -1))


def cos_to_roots(vecs: torch.Tensor, root_embeddings: torch.Tensor) -> torch.Tensor:
    """Cosine similarity of each row in ``vecs`` [N, D] to each root [R, D] -> [N, R]."""
    vecs = F.normalize(vecs.float(), dim=-1)
    roots = F.normalize(root_embeddings.float(), dim=-1)
    return vecs @ roots.t()


@torch.no_grad()
def diagnose(
    model: DagnabbitAutoEncoder,
    num_graphs: int,
    device: torch.device,
    seed: int,
) -> dict:
    torch.manual_seed(seed)
    model.eval()

    R = model.num_root_nodes
    root_embeddings = model.root_node_embeddings.weight.detach().to(device)  # [R, D]
    centroid = root_embeddings.mean(dim=0, keepdim=True)  # [1, D]

    # Accumulators.
    agg_cos_matrix = torch.zeros(
        R, R, device=device
    )  # mean cos(combined_root_i, root_j)
    agg_cos_count = 0
    per_root_encode_cos_dist = torch.zeros(R, device=device)
    per_root_encode_cos_dist_count = torch.zeros(R, device=device)

    agg_identity_conf = torch.zeros(R, R, dtype=torch.long)  # [true, argmax-cos-pred]
    edge_identity_conf = torch.zeros(R, R, dtype=torch.long)

    agg_cos_to_centroid: list[float] = []
    edge_cos_to_centroid: list[float] = []

    # Type-head confusion over root classes only: [true_root, predicted_root]
    # (predicted-root taken only when the head's argmax lands in the root block).
    head_root_conf = torch.zeros(R, R, dtype=torch.long)
    head_root_nonroot = torch.zeros(R, dtype=torch.long)  # predicted outside root block

    # Out-degree vs per-node type-head correctness (all node types).
    outdeg_correct: list[tuple[int, int]] = []  # (out_degree, correct?)

    root_block_start = model.num_trunk_node_types  # roots occupy [start, start+R)

    for _ in range(num_graphs):
        graph = make_random_graph_description(
            num_root_nodes=cfg.NUM_ROOT_NODES,
            num_trunk_nodes=cfg.NUM_TRUNK_NODES,
            num_output_nodes=cfg.NUM_OUTPUT_NODES,
            trunk_node_in_degrees=cfg.TRUNK_NODE_TYPE_IN_DEGREES,
            num_trunk_node_types=cfg.NUM_TRUNK_NODE_TYPES,
        )

        _losses, encode_buffer, combined = model.training_forward(
            graph, return_buffers=True
        )
        encode_buffer = encode_buffer.to(device)
        combined = combined.to(device)

        # ---- aggregated root identity (what the type head actually sees) ----
        root_combined = combined[:R]  # [R, D]
        cm = cos_to_roots(root_combined, root_embeddings)  # [R, R]
        agg_cos_matrix += cm
        agg_cos_count += 1

        agg_pred = cm.argmax(dim=-1).cpu()  # [R]
        for true_i in range(R):
            agg_identity_conf[true_i, agg_pred[true_i]] += 1

        agg_cos_to_centroid.extend(
            cos_to_roots(root_combined, centroid).squeeze(-1).cpu().tolist()
        )

        # Diagnostic: cosine distance between decode combined and encode embedding.
        encode_cos_dist = 1.0 - F.cosine_similarity(
            root_combined.float(), encode_buffer[:R].float(), dim=-1
        )
        per_root_encode_cos_dist += encode_cos_dist.to(per_root_encode_cos_dist.dtype)
        per_root_encode_cos_dist_count += 1

        # ---- type-head behavior on roots ----
        root_logits = model.node_type_predictor(root_combined)  # [R, num_types]
        head_argmax = root_logits.argmax(dim=-1).cpu()  # [R] (global type idx)
        for true_i in range(R):
            pred_global = int(head_argmax[true_i])
            pred_root = pred_global - root_block_start
            if 0 <= pred_root < R:
                head_root_conf[true_i, pred_root] += 1
            else:
                head_root_nonroot[true_i] += 1

        # ---- per-edge child->root predictions (BEFORE aggregation) ----
        # Re-decode every non-root node from its combined embedding; the decoder
        # output for each parent slot is exactly the prediction that was scattered
        # onto that parent during training_forward. Gather the ones landing on a
        # root and score their identity individually.
        for node_idx in range(R, graph.num_nodes):
            parents = graph.node_inputs_indices[node_idx]
            if not parents:
                continue
            if not any(p < R for p in parents):
                continue
            node_type = graph.node_types[node_idx]
            subtypes = torch.tensor([node_type], device=device)
            preds = model.node_decoder.forward_batch(
                combined[node_idx : node_idx + 1],
                subtypes,
            )
            preds = preds[0]  # [in_degree, D]
            for slot, parent in enumerate(parents):
                if parent < R:  # parent is a root
                    pred_vec = preds[slot : slot + 1]  # [1, D]
                    cm_edge = cos_to_roots(pred_vec, root_embeddings)[0]  # [R]
                    pred_root = int(cm_edge.argmax())
                    edge_identity_conf[parent, pred_root] += 1
                    edge_cos_to_centroid.append(
                        float(cos_to_roots(pred_vec, centroid)[0, 0])
                    )

        # ---- out-degree vs correctness (all node types) ----
        out_degree = [0] * graph.num_nodes
        for parents in graph.node_inputs_indices:
            for p in parents:
                out_degree[p] += 1
        all_logits = model.node_type_predictor(combined)  # [N, num_types]
        all_pred = all_logits.argmax(dim=-1).cpu()
        labels = torch.tensor(graph.node_types, dtype=torch.long)
        for node_idx in range(graph.num_nodes):
            correct = int(all_pred[node_idx] == labels[node_idx])
            outdeg_correct.append((out_degree[node_idx], correct))

    return {
        "R": R,
        "root_embeddings": root_embeddings.cpu(),
        "agg_cos_matrix": (agg_cos_matrix / max(agg_cos_count, 1)).cpu(),
        "per_root_encode_cos_dist": (
            per_root_encode_cos_dist / per_root_encode_cos_dist_count.clamp(min=1)
        ).cpu(),
        "agg_identity_conf": agg_identity_conf,
        "edge_identity_conf": edge_identity_conf,
        "head_root_conf": head_root_conf,
        "head_root_nonroot": head_root_nonroot,
        "agg_cos_to_centroid": np.array(agg_cos_to_centroid),
        "edge_cos_to_centroid": np.array(edge_cos_to_centroid),
        "outdeg_correct": np.array(outdeg_correct),
    }


def _fmt_matrix(mat: torch.Tensor) -> str:
    rows = []
    for i, row in enumerate(mat.tolist()):
        cells = "  ".join(f"{v:6.3f}" for v in row)
        rows.append(f"  root {i} | {cells}")
    header = "          " + "  ".join(f"r{j:<5}" for j in range(mat.shape[1]))
    return header + "\n" + "\n".join(rows)


def _identity_accuracy(conf: torch.Tensor) -> float:
    total = int(conf.sum())
    if total == 0:
        return float("nan")
    return float(conf.diag().sum()) / total


def _conf_rows(conf: torch.Tensor) -> str:
    rows = []
    for i, row in enumerate(conf.tolist()):
        total = sum(row) or 1
        cells = "  ".join(f"{v:5d}" for v in row)
        rows.append(
            f"  true {i} | {cells}   (n={sum(row)}, recall={row[i] / total:.3f})"
        )
    header = "          " + "  ".join(f"p{j:<4}" for j in range(conf.shape[1]))
    return header + "\n" + "\n".join(rows)


def report(d: dict) -> None:
    R = d["R"]
    re = d["root_embeddings"]
    print("\n================ ROOT-COLLAPSE DIAGNOSTIC ================\n")

    # Sanity: how orthogonal are the root embeddings, and what is the
    # centroid-vs-root cosine we'd predict under full collapse?
    re_n = F.normalize(re.float(), dim=-1)
    pairwise = re_n @ re_n.t()
    offdiag = pairwise[~torch.eye(R, dtype=torch.bool)]
    centroid = re.float().mean(dim=0, keepdim=True)
    cos_centroid_root = cos_to_roots(re.float(), centroid).squeeze(-1)
    print("Root embedding geometry (the encode/identity targets):")
    print(f"  mean |pairwise cos| between distinct roots = {offdiag.abs().mean():.3f}")
    print(
        "  cos(root_i, centroid) per root             = "
        + ", ".join(f"{v:.3f}" for v in cos_centroid_root.tolist())
    )
    print(
        "  => under full centroid collapse we expect encode cos dist ~= "
        f"{(1 - cos_centroid_root.mean()).item():.3f} and identity acc ~= {1 / R:.3f}\n"
    )

    print("AGGREGATED: cos(combined_root_i, root_embedding_j)  [mean over graphs]")
    print("  (a clean diagonal = roots recovered; flat rows = collapse)")
    print(_fmt_matrix(d["agg_cos_matrix"]))
    print()

    print("Per-root encode cosine distance (1 - cos(combined, encode target)):")
    for i, v in enumerate(d["per_root_encode_cos_dist"].tolist()):
        print(f"  root {i}: {v:.4f}")
    print()

    print("Centroid pull  cos(pred, centroid)   (closer to 1.0 = more collapsed):")
    agg_c = d["agg_cos_to_centroid"]
    edge_c = d["edge_cos_to_centroid"]
    print(f"  aggregated combined : mean={agg_c.mean():.3f}  std={agg_c.std():.3f}")
    if edge_c.size:
        print(
            f"  per-edge predictions: mean={edge_c.mean():.3f}  std={edge_c.std():.3f}"
        )
    print()

    print("IDENTITY RECOVERY (head-independent: argmax_i cos(pred, root_i))")
    edge_acc = _identity_accuracy(d["edge_identity_conf"])
    agg_acc = _identity_accuracy(d["agg_identity_conf"])
    print(
        f"  per-edge  (before aggregation): acc = {edge_acc:.3f}  (chance {1 / R:.3f})"
    )
    print(
        f"  aggregated(after  aggregation): acc = {agg_acc:.3f}  (chance {1 / R:.3f})"
    )
    print("  --> per-edge >> aggregated  ==> aggregation is washing identity out")
    print(
        "  --> per-edge ~= chance      ==> identifiability problem (aggregator won't help)\n"
    )

    print("Per-edge identity confusion (true root -> argmax-cos root):")
    print(_conf_rows(d["edge_identity_conf"]))
    print()
    print("Aggregated identity confusion (true root -> argmax-cos root):")
    print(_conf_rows(d["agg_identity_conf"]))
    print()

    print("Type-head root confusion (true root -> predicted root class):")
    print(_conf_rows(d["head_root_conf"]))
    nonroot = d["head_root_nonroot"].tolist()
    print(f"  predicted OUTSIDE root block per true root: {nonroot}")
    head_acc = _identity_accuracy(d["head_root_conf"])
    print(f"  type-head root identity acc (within root preds) = {head_acc:.3f}\n")

    # Out-degree vs correctness, bucketed.
    oc = d["outdeg_correct"]
    if oc.size:
        deg = oc[:, 0]
        cor = oc[:, 1]
        print("Out-degree vs type-head correctness (all node types):")
        edges = [0, 1, 2, 3, 5, 9, 17, 33, 10**9]
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (deg >= lo) & (deg < hi)
            if mask.any():
                label = f"[{lo},{hi})" if hi < 10**8 else f">={lo}"
                print(
                    f"  out-degree {label:>9}: acc={cor[mask].mean():.3f}  n={int(mask.sum())}"
                )
        if deg.std() > 0:
            r = np.corrcoef(deg, cor)[0, 1]
            print(f"  correlation(out_degree, correct) = {r:.3f}")
    print("\n=========================================================\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--ckpt",
        default="runs/best.ckpt",
        help="Path to a best.ckpt to diagnose.",
    )
    p.add_argument("--num-graphs", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--device",
        default=None,
        help="Override device (default: config.DEVICE).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or cfg.DEVICE)
    print(f"device={device}  ckpt={args.ckpt}")

    model = build_model(device)
    step = load_checkpoint(model, Path(args.ckpt), device)
    print(f"loaded checkpoint at step={step}")

    d = diagnose(model, num_graphs=args.num_graphs, device=device, seed=args.seed)
    report(d)


if __name__ == "__main__":
    main()
