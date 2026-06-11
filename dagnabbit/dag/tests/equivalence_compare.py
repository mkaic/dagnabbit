"""Compare two normalized harness pickles and report max abs/rel diffs.

Usage:  python compare.py <reference.pkl> <candidate.pkl>
"""

import pickle
import sys

import torch


def _load(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def _report_tensor(name: str, a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    a = a.float()
    b = b.float()
    assert a.shape == b.shape, f"{name}: shape {a.shape} vs {b.shape}"
    abs_diff = (a - b).abs()
    max_abs = float(abs_diff.max()) if abs_diff.numel() else 0.0
    denom = b.abs().clamp_min(1e-8)
    max_rel = float((abs_diff / denom).max()) if abs_diff.numel() else 0.0
    print(
        f"  {name:38s} shape={tuple(a.shape)!s:14s} max_abs={max_abs:.3e}  max_rel={max_rel:.3e}"
    )
    return max_abs, max_rel


def main() -> None:
    ref = _load(sys.argv[1])
    cand = _load(sys.argv[2])

    print(f"reference = {sys.argv[1]}")
    print(f"candidate = {sys.argv[2]}")
    print()

    assert ref["num_nodes"] == cand["num_nodes"], "graph differs (num_nodes)"
    assert ref["node_types"] == cand["node_types"], "graph differs (node_types)"
    print(f"graph identical: num_nodes={ref['num_nodes']}")

    # Param-init fingerprint: catches any weight-init RNG drift between commits.
    print("\n-- parameter-init fingerprint (double-precision param sums) --")
    fp_max = 0.0
    assert ref["param_fingerprint"].keys() == cand["param_fingerprint"].keys()
    for name in ref["param_fingerprint"]:
        d = abs(ref["param_fingerprint"][name] - cand["param_fingerprint"][name])
        fp_max = max(fp_max, d)
    print(
        f"  max |param_sum diff| over {len(ref['param_fingerprint'])} params = {fp_max:.3e}"
    )

    print("\n-- forward activations --")
    worst_abs = 0.0
    worst_rel = 0.0
    for key in ("encode_buffer", "decode_combined", "primary_logits"):
        a, r = _report_tensor(key, ref[key], cand[key])
        worst_abs = max(worst_abs, a)
        worst_rel = max(worst_rel, r)

    print("\n-- per-node loss vectors --")
    a, r = _report_tensor(
        "loss_primary_classification",
        ref["loss_primary_classification"],
        cand["loss_primary_classification"],
    )
    worst_abs = max(worst_abs, a)
    worst_rel = max(worst_rel, r)

    print("\n-- total loss --")
    total_abs = abs(ref["total_loss"] - cand["total_loss"])
    print(
        f"  ref={ref['total_loss']:.8f}  cand={cand['total_loss']:.8f}  "
        f"abs_diff={total_abs:.3e}"
    )

    print("\n-- gradients --")
    assert ref["grads"].keys() == cand["grads"].keys(), "grad param sets differ"
    grad_worst_abs = 0.0
    grad_worst_rel = 0.0
    grad_worst_name = ""
    for name in ref["grads"]:
        g_ref = ref["grads"][name].float()
        g_cand = cand["grads"][name].float()
        assert g_ref.shape == g_cand.shape
        abs_diff = (g_ref - g_cand).abs()
        max_abs = float(abs_diff.max())
        scale = float(g_ref.abs().max().clamp_min(1e-8))
        rel = max_abs / scale
        if max_abs > grad_worst_abs:
            grad_worst_abs = max_abs
            grad_worst_name = name
        grad_worst_rel = max(grad_worst_rel, rel)
    print(f"  num_grad_tensors = {len(ref['grads'])}")
    print(f"  worst grad max_abs = {grad_worst_abs:.3e} (param: {grad_worst_name})")
    print(f"  worst grad max_rel (vs that tensor's max|grad|) = {grad_worst_rel:.3e}")

    print("\n==== SUMMARY ====")
    print(f"  activations: max_abs={worst_abs:.3e}  max_rel={worst_rel:.3e}")
    print(f"  total_loss : abs={total_abs:.3e}")
    print(f"  grads      : max_abs={grad_worst_abs:.3e}  max_rel={grad_worst_rel:.3e}")
    print(f"  param init : max_abs={fp_max:.3e}")


if __name__ == "__main__":
    main()
