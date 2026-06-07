# Training wallclock optimization notes

Context handoff for a future agent. The goal is **reducing training wallclock
time per step** for a fixed network size and graph node count. Memory has
already been addressed (see "Done" below) and is no longer a concern.

## Model / training shape (why this is unusual)

- `dagnabbit/dag/autoencoder.py` — `DagnabbitAutoEncoder` encodes/decodes
  random-structured DAGs of small MLPs feeding into each other.
- `dagnabbit/scripts/train_encoder.py` — training loop.
- **Batch size is 1 and this is deliberate.** Each step samples a *new* random
  DAG (`make_random_graph_description`), so graph topology, node count, and
  condenser depth vary step to step. Cross-step batching is intentionally not
  attempted.
- A step walks nodes **one at a time in a Python loop** (`evaluate_graph` for
  encode, `_decode_step` for decode), calling small fixed-shape MLPs
  (`StandardizedLinear` = Scaled Weight Standardization, Brock et al. 2021) per
  node. This produces ~1,700 tiny matmuls/step.

## Already done (do NOT redo)

1. **Per-step standardized-weight caching.**
   `StandardizedLinear` memoizes its standardized weight via
   `prime_weight_cache(dtype)` / `clear_weight_cache()`, orchestrated by the
   `DagnabbitAutoEncoder.cached_standardized_weights(dtype)` context manager
   wrapped around each step's forward+backward in `train_encoder.py`. The cache
   is primed in the autocast compute dtype (bf16 on CUDA) so the bf16 cast is
   also done once. Numerically identical to recomputing per call (verified: loss
   bit-identical, max grad diff ~4e-8), so Adam is unaffected.
2. **Deferred GPU syncs.** `combine_losses` returns loss components as tensors;
   `.item()` is only called on logging steps (was 5 syncs/step, now 1).
3. **Precomputed cross-entropy labels.** `_decode_step` takes a precomputed
   on-device `class_label` tensor instead of building one per node.

Result: peak memory 1.91 GB -> 0.21 GB (~9x), self CPU time ~2.6x lower, self
CUDA ~4.2x lower, ~2.3x faster per step.

## Current bottleneck: launch-bound (CPU-bound), not compute-bound

From the latest `--profile` run (8 active steps):

- Self CPU time 3.36 s vs Self CUDA time 0.23 s.
- GPU busy only ~6% of each step (ProfilerStep ~506 ms/step, ~29 ms of real GPU
  work).
- Top remaining costs are now the *irreducible* work: `AddmmBackward0`,
  `aten::mm`, `aten::addmm` (the per-node matmuls + their backward, ~1,716
  launches/step) and a single `Adam.step` (~38 ms for 7.75M params).

The waste is gone; what remains is the cost of launching thousands of tiny
per-node kernels at batch size 1. **Further speedups require reducing the number
of kernel launches.**

## How to profile (already wired up)

> **IMPORTANT (agent, read this):** Do NOT run the profiling yourself. The
> CUDA-capable hardware lives on a separate machine that only the user has
> access to — this dev environment has no usable GPU, so any profiling run you
> attempt will be meaningless or fail. When profiling is needed, **ask the user
> to run the command below on their CUDA machine and paste the results back to
> you** (the `op_table.txt` contents / terminal output, and the
> `analyze_mem.py` output if memory is relevant). Wait for those results before
> drawing conclusions.

```bash
uv run -m dagnabbit.scripts.train_encoder --profile --no-log
# optional: --profile-steps N, --profile-output-dir DIR
```

Artifacts land in `profile_out/`: `op_table.txt`, `trace.json` (open in
chrome://tracing or ui.perfetto.dev), `mem_snapshot.pickle`
(pytorch.org/memory_viz). Memory breakdown helper:

```bash
uv run python profile_out/analyze_mem.py profile_out/mem_snapshot.pickle
```

Sanity check for "launch-bound": Self CPU time >> Self CUDA time, and large gaps
on the CUDA timeline in the trace.

## Future optimizations (prioritized)

### 1. Batch same-type node ops out of the sequential loop (highest payoff)

The per-node Python loop is the root cause. Collapse independent per-node ops
into batched `[N, D]` calls:

- **Classification head + decode losses:** the `node_type_predictor` and the
  classification / similarity losses currently run once per node inside
  `_decode_step`. They depend only on each node's `combined_predicted_embedding`.
  Collect those into one `[num_nodes, D]` tensor and run the predictor +
  `cross_entropy` + similarity once, after the decode loop, instead of ~N times.
- **Encode/decode within a topological level:** nodes at the same depth have no
  data dependencies on each other and share an autoencoder (only 1 trunk type,
  shared condenser AE). They can be processed as a batch. This requires grouping
  nodes by (depth, node_type) and gathering parent embeddings per group — more
  involved, but it's where most of the ~1,716 launches/step go.

This is the prerequisite for #2 and #3 to be effective.

### 2. torch.compile (after #1)

`torch.compile` fuses elementwise chains and cuts Python/dispatch overhead.
Caveat: the data-dependent Python loop causes **graph breaks** — compiled
regions still get re-entered per node, so most launch overhead remains until the
per-node loop is collapsed. Worth applying to the batched ops from #1, where
shapes are larger and control flow is simpler. Expect recompiles on new shapes
(use dynamic shapes).

### 3. CUDA graphs (after #1)

CUDA graphs record a kernel-launch sequence and replay it with ~zero CPU launch
cost — directly targeting the launch-bound bottleneck. Caveats: requires static
shapes and fixed memory addresses, and cannot capture data-dependent Python
control flow. Only viable on fixed-shape sub-pieces, or once #1 makes per-step
work regular enough to capture.

Note: `torch.compile(model, mode="reduce-overhead")` enables CUDA graphs under
the compiler, so #2 and #3 can be combined rather than implemented separately.

## Guardrails

- Preserve the Scaled-WS reparameterization (differentiable standardization in
  forward; Adam optimizes the raw latent weights). Do not switch to in-place
  post-step weight normalization — it changes the optimization math.
- Keep batch size 1 / per-step random graphs.
- Re-profile before and after each change to confirm impact — but **the user
  runs profiling on their CUDA machine and feeds the results back** (see "How to
  profile"); the agent must not run it. Verifying loss/grad parity on a fixed
  seed + fixed graph can be done locally on CPU, so do that yourself to keep
  optimizations numerically faithful.
```
