"""Standalone training-loop profiler for the DAG autoencoder.

Run this on the CUDA machine. It dumps a single human-readable SUMMARY.txt
(plus a chrome trace and the torch op table) into a timestamped directory under
``profile_out/``. Paste SUMMARY.txt back for analysis.

What it measures, and why:

  * Per-phase HOST vs GPU time. Each training step is split into four phases:
    graph-gen (pure-Python CPU), forward (encode + decode), backward, optimizer.
    For each we record both wall-clock host time (CPU + kernel-launch overhead)
    and actual GPU-timeline time (CUDA events). The gap between "sum of host
    phase times" and "GPU busy time" is the tell: if GPU busy fraction is low,
    the loop is launch-bound / CPU-bound, not compute-bound, and the win is
    fusing/batching Python-driven kernel launches rather than a bigger GPU.

  * Encode vs compress vs decode split. The encoder is the remaining sequential
    path -- a Python loop over topological ranks, one shared-transformer
    invocation per rank -- while compress/decode are single dense passes over
    the whole canonical sequence.

  * A torch.profiler pass: top ops by CUDA and by CPU time, total CUDA kernel
    count per step, and average kernel duration (tiny avg => launch-bound).

  * Peak memory.

Usage:
    uv run python -m dagnabbit.scripts.profile_training
    uv run python -m dagnabbit.scripts.profile_training --steps 30 --warmup 10
    uv run python -m dagnabbit.scripts.profile_training --batch-size 64 --no-trace
"""

import argparse
import platform
import statistics
import time
from contextlib import contextmanager, nullcontext
from datetime import datetime
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from dagnabbit.dag.autoencoder import DagnabbitAutoEncoder
from dagnabbit.dag.description import make_random_graph_description
from dagnabbit.optimizers import build_optimizer
from dagnabbit.scripts import config as cfg
from dagnabbit.scripts.train_encoder import combine_losses


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--steps",
        type=int,
        default=20,
        help="Measured steps for the manual phase timing.",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=8,
        help="Unmeasured warmup steps (allocator/cudnn autotune settle).",
    )
    p.add_argument(
        "--profiler-steps",
        type=int,
        default=6,
        help="Active steps recorded by torch.profiler.",
    )
    p.add_argument(
        "--batch-size", type=int, default=None, help="Override cfg.GRAPH_BATCH_SIZE."
    )
    p.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override cfg.TORCH_COMPILE.",
    )
    p.add_argument(
        "--amp",
        choices=("off", "bf16", "fp16"),
        default="off",
        help=(
            "Wrap the forward in torch.autocast to measure mixed-precision "
            "throughput. Timing-only: no GradScaler is used, so with fp16 the "
            "gradients themselves are not trustworthy -- compare wall/step and "
            "peak memory, not losses."
        ),
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Output dir (default: profile_out/<timestamp>).",
    )
    p.add_argument(
        "--no-trace",
        action="store_true",
        help="Skip the chrome trace export (smaller output).",
    )
    return p.parse_args()


class PhaseTimer:
    """Collects per-step host wall time and (optionally) GPU-event time per phase."""

    def __init__(self, device: torch.device):
        self.device = device
        self.cuda = device.type == "cuda"
        self.host: dict[str, list[float]] = {}
        self._events: dict[str, list[tuple]] = {}

    @contextmanager
    def phase(self, name: str):
        start_evt = end_evt = None
        if self.cuda:
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            start_evt.record()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            t1 = time.perf_counter()
            self.host.setdefault(name, []).append((t1 - t0) * 1e3)
            if self.cuda:
                end_evt.record()
                self._events.setdefault(name, []).append((start_evt, end_evt))

    def gpu_ms(self, name: str) -> list[float]:
        if not self.cuda:
            return []
        return [s.elapsed_time(e) for s, e in self._events.get(name, [])]


def fmt_ms(values: list[float]) -> str:
    if not values:
        return "      n/a"
    mean = statistics.mean(values)
    med = statistics.median(values)
    return f"{mean:8.2f} (median {med:7.2f})"


def main() -> None:
    args = parse_args()
    if args.batch_size is not None:
        cfg.GRAPH_BATCH_SIZE = args.batch_size
    if args.compile is not None:
        cfg.TORCH_COMPILE = args.compile

    torch.manual_seed(cfg.SEED)
    torch.set_float32_matmul_precision("high")
    device = torch.device(cfg.DEVICE)

    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else (Path("profile_out") / datetime.now().strftime("%Y%m%d-%H%M%S"))
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    model = DagnabbitAutoEncoder(
        node_embedding_dim=cfg.NODE_EMBEDDING_DIM,
        trunk_node_type_in_degrees=cfg.TRUNK_NODE_TYPE_IN_DEGREES,
        num_trunk_node_types=cfg.NUM_TRUNK_NODE_TYPES,
        num_root_nodes=cfg.NUM_ROOT_NODES,
        num_trunk_nodes=cfg.NUM_TRUNK_NODES,
        num_output_nodes=cfg.NUM_OUTPUT_NODES,
        mlp_expansion_factor=cfg.MLP_EXPANSION_FACTOR,
        encoder_num_layers=cfg.ENCODER_NUM_LAYERS,
        compressor_num_layers=cfg.COMPRESSOR_NUM_LAYERS,
        decoder_num_layers=cfg.DECODER_NUM_LAYERS,
    ).to(device)

    if cfg.TORCH_COMPILE and device.type == "cuda":
        from dagnabbit.scripts.train_encoder import apply_torch_compile

        apply_torch_compile(model, device)

    optimizer = build_optimizer(cfg.OPTIMIZER_CLASS, model, **cfg.OPTIMIZER_KWARGS)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Split forward into encode / compress / decode by wrapping the bound
    # methods. The wrappers only time; they call straight through to the
    # originals.
    timer = PhaseTimer(device)
    orig_encode = model.evaluate_graph_batch
    orig_compress = model.compress
    orig_decode = model.decode_latent

    def timed_encode(*a, **k):
        with timer.phase("encode"):
            return orig_encode(*a, **k)

    def timed_compress(*a, **k):
        with timer.phase("compress"):
            return orig_compress(*a, **k)

    def timed_decode(*a, **k):
        with timer.phase("decode"):
            return orig_decode(*a, **k)

    model.evaluate_graph_batch = timed_encode
    model.compress = timed_compress
    model.decode_latent = timed_decode

    def make_graphs():
        return [
            make_random_graph_description(
                num_root_nodes=cfg.NUM_ROOT_NODES,
                num_trunk_nodes=cfg.NUM_TRUNK_NODES,
                num_output_nodes=cfg.NUM_OUTPUT_NODES,
                trunk_node_in_degrees=cfg.TRUNK_NODE_TYPE_IN_DEGREES,
                num_trunk_node_types=cfg.NUM_TRUNK_NODE_TYPES,
            )
            for _ in range(cfg.GRAPH_BATCH_SIZE)
        ]

    amp_dtype = {"off": None, "bf16": torch.bfloat16, "fp16": torch.float16}[args.amp]

    def autocast_ctx():
        if amp_dtype is None:
            return nullcontext()
        return torch.autocast(device_type=device.type, dtype=amp_dtype)

    def run_step():
        with timer.phase("gen"):
            graphs = make_graphs()
        with timer.phase("forward"):
            # Backward stays outside the autocast region, as autocast intends.
            with autocast_ctx():
                losses = model.training_forward_batch(graphs)
                total, _ = combine_losses(losses)
        with timer.phase("backward"):
            total.backward()
        with timer.phase("optimizer"):
            if cfg.GRADIENT_CLIP_MAX_NORM is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.GRADIENT_CLIP_MAX_NORM
                )
            optimizer.step()
            optimizer.zero_grad()

    print(
        f"device={device} batch_size={cfg.GRAPH_BATCH_SIZE} "
        f"params={num_params / 1e6:.2f}M compile={cfg.TORCH_COMPILE}"
    )
    print(
        f"warmup={args.warmup} measured={args.steps} "
        f"profiler_steps={args.profiler_steps}"
    )

    # --- warmup (not timed) ---
    for _ in range(args.warmup):
        run_step()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    timer.host.clear()
    timer._events.clear()

    # --- measured manual phase timing ---
    wall = []
    for _ in range(args.steps):
        t0 = time.perf_counter()
        run_step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        wall.append((time.perf_counter() - t0) * 1e3)

    # --- torch.profiler pass for op-level detail ---
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    with profile(activities=activities, record_shapes=False) as prof:
        for _ in range(args.profiler_steps):
            run_step()
        if device.type == "cuda":
            torch.cuda.synchronize()

    # --- assemble report ---
    lines: list[str] = []
    w = lines.append
    w("=" * 78)
    w("DAGNABBIT TRAINING PROFILE")
    w("=" * 78)
    w(f"timestamp        {datetime.now().isoformat(timespec='seconds')}")
    w(f"host             {platform.node()} ({platform.platform()})")
    w(f"torch            {torch.__version__}")
    if device.type == "cuda":
        w(f"gpu              {torch.cuda.get_device_name(0)}")
        cap = torch.cuda.get_device_capability(0)
        w(f"compute_cap      {cap[0]}.{cap[1]}")
    w(f"device           {device}")
    w(f"batch_size       {cfg.GRAPH_BATCH_SIZE}")
    w(
        f"trunk_nodes      {cfg.NUM_TRUNK_NODES}  roots {cfg.NUM_ROOT_NODES}  "
        f"outputs {cfg.NUM_OUTPUT_NODES}"
    )
    w(
        f"embedding_dim    {cfg.NODE_EMBEDDING_DIM}  heads "
        f"{model.node_encoder.sequence_transformer.num_heads}"
    )
    # Spell out all three layer counts: they have churned independently, and a
    # summary that reports only one is not comparable against archived runs.
    w(
        f"layers           encoder {cfg.ENCODER_NUM_LAYERS}  compressor "
        f"{cfg.COMPRESSOR_NUM_LAYERS if model.compressor is not None else 'disabled'}"
        f"  decoder {cfg.DECODER_NUM_LAYERS}"
    )
    w(f"params           {num_params / 1e6:.2f}M")
    w(f"torch_compile    {cfg.TORCH_COMPILE}")
    w(f"amp              {args.amp}")
    w(f"measured_steps   {args.steps} (after {args.warmup} warmup)")
    w("")

    mean_wall = statistics.mean(wall)
    w("STEP THROUGHPUT")
    w(f"  wall/step      {fmt_ms(wall)} ms")
    w(f"  steps/sec      {1000.0 / mean_wall:8.2f}")
    w(f"  graphs/sec     {1000.0 * cfg.GRAPH_BATCH_SIZE / mean_wall:8.2f}")
    w("")

    w("PER-PHASE HOST TIME (CPU + kernel-launch overhead), ms/step")
    w("  phase            host_ms")
    phase_order = [
        "gen",
        "forward",
        "encode",
        "compress",
        "decode",
        "backward",
        "optimizer",
    ]
    host_sum = 0.0
    for name in phase_order:
        if name in timer.host:
            vals = timer.host[name]
            w(f"  {name:14s} {fmt_ms(vals)}")
            if name in ("gen", "forward", "backward", "optimizer"):
                host_sum += statistics.mean(vals)
    w(f"  {'(sum top-level)':14s} {host_sum:8.2f}")
    w(
        "    note: encode/compress/decode are sub-phases of forward, "
        "not added to the sum."
    )
    w("")

    if device.type == "cuda":
        w("PER-PHASE GPU TIME (CUDA-event timeline), ms/step")
        w("  phase             gpu_ms")
        gpu_total = 0.0
        for name in phase_order:
            g = timer.gpu_ms(name)
            if g:
                w(f"  {name:14s} {fmt_ms(g)}")
                if name in ("gen", "forward", "backward", "optimizer"):
                    gpu_total += statistics.mean(g)
        w(f"  {'(sum top-level)':14s} {gpu_total:8.2f}")
        busy = 100.0 * gpu_total / mean_wall
        w("")
        w(
            f"  event-span sum     {busy:5.1f}%  "
            f"(gpu_total {gpu_total:.2f} ms / wall {mean_wall:.2f} ms)"
        )
        w("    NOT a utilization figure: CUDA-event spans enclose the idle")
        w("    bubbles between kernels, so this can exceed 100%. Use TRUE GPU")
        w("    UTILIZATION below instead.")
        w("")

    # op tables
    sort_keys = ["self_cuda_time_total"] if device.type == "cuda" else []
    sort_keys.append("self_cpu_time_total")
    ka = prof.key_averages()
    for sk in sort_keys:
        try:
            table = ka.table(sort_by=sk, row_limit=25)
        except Exception:
            continue
        w(f"TOP OPS BY {sk}  (over {args.profiler_steps} profiler steps)")
        w(table)
        w("")
        (out_dir / f"op_table_{sk}.txt").write_text(table)

    # kernel-launch signal
    if device.type == "cuda":
        kernel_events = [
            e
            for e in ka
            if e.device_type.name == "CUDA" and e.self_device_time_total > 0
        ]
        total_kernel_calls = sum(e.count for e in kernel_events)
        total_kernel_us = sum(e.self_device_time_total for e in kernel_events)
        per_step_calls = total_kernel_calls / max(1, args.profiler_steps)
        avg_kernel_us = total_kernel_us / max(1, total_kernel_calls)
        # Summed kernel self-time is real GPU-busy time with no idle included,
        # so this ratio is the honest utilization number and the one that says
        # whether shrinking kernels (AMP, bigger tensor cores) can pay at all.
        kernel_ms_per_step = total_kernel_us / max(1, args.profiler_steps) / 1e3
        true_util = 100.0 * kernel_ms_per_step / mean_wall
        w("TRUE GPU UTILIZATION")
        w(f"  kernel busy time / step       {kernel_ms_per_step:10.2f} ms")
        w(f"  utilization                   {true_util:10.1f} %")
        if true_util < 60:
            w("  => LAUNCH-BOUND: the GPU is idle most of the step. Cutting kernel")
            w("     *duration* (AMP/bf16, a faster GPU) caps out at the busy time")
            w("     above; the win is fewer/larger launches -- batch the Python")
            w("     rank loops. AMP also adds cast kernels to the dispatch path.")
        else:
            w("  => Approaching compute-bound: kernel duration is now worth")
            w("     attacking, so AMP/bf16 can convert into real wall-clock time.")
        w("")
        w("KERNEL-LAUNCH SIGNAL")
        w(f"  cuda kernel launches / step   {per_step_calls:10.0f}")
        w(f"  avg kernel duration           {avg_kernel_us:10.2f} us")
        w("  (thousands of launches and/or sub-10us avg => launch-bound; the")
        w("   Python rank/graph loops are issuing many tiny kernels.)")
        w("")
        w("MEMORY")
        w(f"  max allocated   {torch.cuda.max_memory_allocated() / 1e9:.3f} GB")
        w(f"  max reserved    {torch.cuda.max_memory_reserved() / 1e9:.3f} GB")
        w("")

    report = "\n".join(lines)
    print("\n" + report)
    (out_dir / "SUMMARY.txt").write_text(report)

    if not args.no_trace and device.type == "cuda":
        trace_path = out_dir / "trace.json"
        prof.export_chrome_trace(str(trace_path))
        print(f"chrome_trace={trace_path}  (open at https://ui.perfetto.dev)")

    print(f"\nwrote {out_dir}/SUMMARY.txt  <- paste this back for analysis")


if __name__ == "__main__":
    main()
