# dagnabbit

Is it possible to use machine learning to search the space of fixed-in-degree
DAGs for one that computes a given boolean function?

The target is an 8-bit adder: 16 input bits, 8 output bits, built from a budget
of 128 gates. Two hand-wired solutions exist inside that budget — one all-NAND,
one spending XOR — so the search has known optima to measure against, and
probing a trained model on both separates gate vocabulary from depth.

The gate set is **NAND, XOR, XNOR**, chosen by measuring how random circuits
behave as they scale. The metric is *free accuracy*: what a predictor scores by
emitting each output's marginal bit and never reading the graph.

| gate set | 64 gates | 512 | 4096 |
| --- | --- | --- | --- |
| NAND, NOR | 80.7% | 88.4% | 90.4% |
| AND, NAND | 80.3% | 88.8% | 91.0% |
| NAND | 63.8% | 63.5% | 61.2% |
| NAND, XOR | 60.6% | 60.8% | 59.4% |
| **NAND, XOR, XNOR** | **57.9%** | **57.1%** | **56.1%** |
| XOR *(linear, unusable)* | 50.2% | 50.0% | 50.0% |

Two things decide this. Whether constants **cascade** — `NAND(0,y) = 1` absorbs
but `NAND(1,y) = ¬y` does not, so under NAND a constant dies after one hop,
whereas NAND+NOR alternates `0 → 1 → 0` forever and a third of its outputs are
constant at 4096 gates. And the mean-field fixed point of `p ↦ P(output = 1)`:
`1 − p²` for NAND gives the golden ratio 0.618, while adding XOR and XNOR
contributes a symmetric pair that cancels about ½, giving exactly 0.5.

Gates may draw the same parent twice — `NAND(x, x) = NOT x` is the only way an
inverter exists, and the reference adders are ~48% inverters. Coverage is
unaffected, so no node is ever dead.

## The approach

Rather than pretrain a task-agnostic representation of graphs, train one model
on the downstream question directly: **given a DAG, what is its truth table?**
Random graphs are cheap to sample (~35 us each) and exact to evaluate, so this
is an unlimited supply of labeled data and no example is ever seen twice.

That model is a differentiable surrogate for the real evaluator. Inverse design
then means running it backwards: hold the surrogate fixed, and optimize a
circuit until its *predicted* truth table matches the one you want.

### Phase 0 — the simulator (`dagnabbit/scripts/train_simulator.py`)

Three stages, in `dagnabbit/dag/model.py`:

1. Each node becomes one token by summing embeddings of its type, its own node
   index, and its parents' node indices. No transformer, no pooling — a gather
   and three matmuls, fully parallel across nodes. Node index order is already a
   valid topological order, so there is no canonicalization step; an earlier
   structure-derived ordering leaked structure into the index itself.
2. An unmasked transformer runs over the 152-token sequence, optionally plus a
   configurable bank of register tokens belonging to no node. This is where the
   work happens: composing a gate with its ancestors' values is a hop of message
   passing, and the layer count bounds how many hops are available.
3. A grid of learned patch queries cross-attends the result. Patch `p` owns a
   contiguous block of truth-table rows across every output, so one query
   decodes thousands of bits and the loss can be taken on a random subset of
   patches per step.

There is no reconstruction loss and nothing decodes back to a graph. The token
space is never inverted — Phase 1 emits categorical choices and *constructs*
tokens from them, so what the simulator reads is always a real graph's encoding.

**The gate.** Average bit accuracy is not the number to watch: a model can score
well above chance on statistical regularities of random circuits without
simulating anything, and such a model has a useless loss landscape near an
actual target. Two metrics separate the cases.

`eval/mcc` is Matthews correlation between predicted and exact bits. It is 0 for
any constant predictor regardless of class balance, so it only moves when a
prediction tracks the *specific* circuit. The target distribution is close to
balanced — entropy 0.660 nats against a fair coin's 0.693, so a marginal-only
predictor bottoms out at **BCE 0.660 and 57.3% accuracy**. That is the real zero
point on the loss, not 0.693. Outputs with a constant target are excluded from
the MCC average (there is nothing to correlate against) and reported as
`eval/constant_target_fraction`, now 0.3%.

`eval/accuracy_by_rank` and `eval/mcc_by_rank` are the same two bucketed by each
output node's longest-path depth. Output nodes sit at median depth 11 (p95 15,
max 22), so if either falls off a cliff at the simulator's layer count, the
receptive field is binding and the fix is more layers or a weight-tied recurrent
simulator. If they degrade smoothly past the layer count, the model found
something cheaper than hop-by-hop propagation.

### Phase 1 — inverse design, not yet built

Optimize a batch of independent candidate circuits by gradient descent through
the frozen simulator, with Gumbel-softmax and a straight-through estimator so
the forward value is always a real discrete graph. Periodically re-evaluate the
hard-sampled graphs with the exact evaluator and fine-tune the simulator on the
result, which is what keeps it from being exploited.

The diagnostic that matters: the gap between predicted and true truth-table
loss. If predicted loss goes to zero while true loss does not, the optimizer is
exploiting the surrogate rather than finding a circuit.

### Phase 2 — amortization, only if Phase 1 earns it

Replace the population of independent candidates with a generator mapping noise
to circuits. That buys parameter sharing and a smoother search landscape, but it
sits downstream of the same surrogate, so it can fix an optimization failure and
not a model failure.

## Layout

| path | what |
| --- | --- |
| `dag/generate.py` | compiled (numba) random DAG sampler |
| `dag/graphs.py` | the one graph representation: types, parent indices, ranks |
| `dag/model.py` | node tokens, simulator, truth-table patch decoder |
| `dag/metrics.py` | bit accuracy, Matthews correlation, depth bucketing |
| `tasks/logic_gates/evaluate.py` | exact bitpacked circuit evaluation and scoring |
| `tasks/logic_gates/reference_circuits.py` | two hand-wired adders (all-NAND and NAND+XOR), as known optima |
| `scripts/train_simulator.py` | Phase 0 training loop |
| `scripts/config.py` | geometry, model, and training knobs |

## Running

```bash
uv run python -m dagnabbit.scripts.train_simulator --name my-run
```

```bash
uv run python -m pytest dagnabbit -q
```
