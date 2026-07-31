# dagnabbit

Is it possible to use machine learning to search the space of fixed-in-degree
DAGs for one that computes a given boolean function?

The target is an 8-bit adder: 16 input bits, 8 output bits, built from a budget
of 128 NAND/NOR gates. A hand-wired solution exists inside that budget
(`nand_ripple_carry_adder`), so the search has a known optimum to measure
against.

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

1. Each node becomes one token by summing embeddings of its type, its own
   canonical position, and its parents' canonical positions. No transformer, no
   pooling — a gather and three matmuls, fully parallel across nodes.
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
well above chance on statistical regularities of random NAND/NOR circuits
without simulating anything, and such a model has a useless loss landscape near
an actual target. Two metrics separate the cases.

`eval/mcc` is Matthews correlation between predicted and exact bits. It is 0 for
any constant predictor regardless of class balance, so it only moves when a
prediction tracks the *specific* circuit. This matters a lot here because random
circuit outputs are badly imbalanced — measured over 2048 sampled circuits,
**17% of output nodes are constant over the entire truth table and 47% are more
lopsided than 90/10**. Accuracy climbing while MCC sits at zero is the signature
of a model that has learned the marginal bit. Outputs with a constant target are
excluded from the MCC average (there is nothing to correlate against) and
reported as `eval/constant_target_fraction`.

`eval/accuracy_by_rank` and `eval/mcc_by_rank` are the same two bucketed by each
output node's longest-path depth. Output nodes sit at median depth 11 (p95 15,
max 23), so if either falls off a cliff at the simulator's layer count, the
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
| `dag/canonical.py` | the one graph representation: types, parent positions, ranks |
| `dag/model.py` | node tokens, simulator, truth-table patch decoder |
| `tasks/logic_gates/evaluate.py` | exact bitpacked circuit evaluation and scoring |
| `tasks/logic_gates/reference_circuits.py` | the hand-wired adder, as a known optimum |
| `scripts/train_simulator.py` | Phase 0 training loop |
| `scripts/config.py` | geometry, model, and training knobs |

## Running

```bash
uv run python -m dagnabbit.scripts.train_simulator --name my-run
```

```bash
uv run python -m pytest dagnabbit -q
```
