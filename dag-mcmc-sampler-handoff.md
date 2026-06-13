# Handoff: Sampling DAGs by MCMC Edge-Rewiring

This document is a self-contained spec for a method to sample random DAGs with a
**fixed number of root, trunk, and leaf nodes**, with a distribution that is
uniform (or close to it) over a clearly-defined space — i.e. without the severe
structural bias that growth-style construction produces. It is written so a fresh
Claude Code instance can implement it cold. Read all of it before coding; the
correctness rests on a few non-obvious points that are easy to get wrong.

If you also have an earlier "edge-splitting construction" handoff: **this
supersedes it as the generation method.** That growth process turned out to be
badly biased (see §1). The reachability discussion in it is still useful background.

---

## 1. Why we're not using a growth/construction process

The previous approach grew the graph by inserting nodes one at a time (splitting a
random edge, attaching a new node to a random earlier node). It produced graphs
that "don't feel random": a rich-get-richer effect meant some outputs had ~98% of
nodes as ancestors of a single node while others had ~1%. That bias is intrinsic
to any "grow by local random choices" rule — early nodes accumulate an
advantage, and the final graph is weighted by how many build-orders could have
produced it, not uniformly.

Rejection sampling (generate a much larger graph, prune to the desired leaf count)
removes the bias but costs ~N² work to produce a size-N graph.

MCMC fixes both: it works at exactly size N, holds the counts exactly, and has **no
growth dynamic at all** — so nothing is ever "early" and nothing accumulates an
advantage.

---

## 2. What MCMC does, in one paragraph

Like shuffling a deck of cards: you don't *build* a shuffled deck card-by-card
(that smuggles in patterns), you start from any order and make many small random
swaps until the start order is forgotten. Here: start from **any** valid DAG that
already has the right root/trunk/leaf counts (even a biased one, or a trivial
hand-built one), then repeatedly make one tiny **legal, reversible** edit. After
enough edits the graph is a fair sample and the seed has washed out.

The edit: grab one edge `p → z` and re-point its tail, making it `p_new → z`,
provided the result is still a valid DAG with the same counts.

---

## 3. Define the target space FIRST (important)

"Uniform over DAGs with R/T/L" is ambiguous until the degree rules are fixed. The
single-edge rewiring below **preserves the in-degree of every node** and lets
out-degrees float. So it samples uniformly over:

> **All labeled DAGs with a fixed in-degree sequence and fixed role assignment** —
> R designated roots (in-degree 0), T trunks, L leaves — where each node keeps the
> in-degree it started with.

Match the starting in-degrees to your structure. The original construction gave
**trunks in-degree 2, leaves in-degree 1, roots in-degree 0**; if that's your
intent, seed the chain with a graph having exactly those in-degrees and they stay
fixed forever. Consequences worth stating explicitly:

- Number of edges `|E| = 2·T + 1·L` (sum of in-degrees) is **constant** across the
  whole space. (This matters for §5.)
- Number of non-leaf nodes `R + T` is **constant**. (Also matters for §5.)

If you instead want only the *counts* fixed with in-degrees free to vary, you need
the extra moves in §8 and a real Metropolis-Hastings ratio — do not use the simple
"accept if legal" rule in that case.

---

## 4. The algorithm

```
# ---- setup (once) ----
designate R nodes as roots, T as trunks, L as leaves
G = any valid DAG honoring those roles and in-degrees   # a biased seed is fine
NONLEAF = roots ∪ trunks            # fixed-size set (R + T), never changes

# ---- one MCMC step ----
def step(G):
    (p, z) = a uniformly random edge of G       # re-point this edge's tail
    p_new  = a uniformly random node from NONLEAF
    if legal(G, p, z, p_new):
        G.remove_edge(p, z)
        G.add_edge(p_new, z)                     # ACCEPT
    # else: leave G unchanged                    # REJECT — staying put is normal

def legal(G, p, z, p_new):
    if p_new == z or p_new == p:   return False  # no self-loop, no no-op
    if G.has_edge(p_new, z):       return False  # no duplicate edge
    if G.out_degree(p) == 1:       return False  # p would lose its last out-edge -> become a leaf
    if can_reach(G, z, p_new):     return False  # p_new -> z would close a cycle
    return True

# ---- drawing samples ----
for _ in range(BURN_IN):   step(G)               # mix away from the seed
emit snapshot(G)
loop:
    for _ in range(THIN):  step(G)               # decorrelate between samples
    emit snapshot(G)
```

`can_reach(G, z, p_new)` = "is there already a path from z to p_new?" A DFS/BFS
forward from `z` that stops the instant it reaches `p_new`. This is the only
non-trivial per-step cost; on sparse graphs it is cheap and usually short-circuits
fast.

---

## 5. Why "accept if legal" is correct (do not add an MH ratio here)

In general, Metropolis-Hastings requires accepting a proposed move with probability
`min(1, ...)` to avoid favoring states that offer more moves. We sidestep that by
making the proposal **symmetric** — the move and its exact reverse are always
equally likely to be proposed:

- The forward edit is uniquely specified by (the edge `p→z`, the new parent
  `p_new`). Its probability is `1 / (|E| · |NONLEAF|)`.
- The reverse edit (turning the new graph back into the old one) is specified by
  (the edge `p_new→z`, the new parent `p`). Its probability is also
  `1 / (|E| · |NONLEAF|)`, because `|E|` and `|NONLEAF|` are the same constants in
  every valid state (§3), and **`p` is always a non-leaf** (it had the out-edge
  `p→z`, so its out-degree was ≥ 1), so `p` is a legitimate draw from `NONLEAF` on
  the way back.

Symmetric proposal + uniform target ⇒ the acceptance probability collapses to
`min(1, 1) = 1` for every legal move. So: accept iff the result is a valid state.
**Drawing `p_new` from `NONLEAF` (not from all nodes) is what keeps the proposal
set a constant size and the proposal symmetric — keep it that way.**

---

## 6. How the guards keep the counts pinned

Each `legal` check maps to a category that must not flip:

- Removing `p→z` and adding `p_new→z` leaves **z's in-degree unchanged** — so z
  never changes category, and the whole in-degree sequence is preserved.
- Roots (in-degree 0) are never the head of an edge, so they're never chosen as
  `z` and never gain an in-edge. They stay roots.
- Leaves (out-degree 0) are excluded from `NONLEAF`, so they're never chosen as
  `p_new` and never gain an out-edge. They stay leaves.
- The `out_degree(p) == 1` guard stops a trunk or root from losing its last
  out-edge and collapsing into a leaf. It stays a non-leaf.
- `p_new` is a non-leaf gaining an out-edge — still a non-leaf.

Net effect: every node keeps its role on every step, so every state the chain ever
visits — and every sample you emit — has exactly R roots, T trunks, L leaves.

---

## 7. Why we recompute reachability instead of maintaining it

If you saw the incremental-transitive-closure trick in the earlier handoff: it does
**not** apply here. That trick relies on edges only ever being *added* (monotone
reachability). MCMC *removes* an edge on every accepted step, which can destroy
reachability and breaks monotonicity. Maintaining reachability under deletions is
much harder, so the practical choice is a **fresh single-pair reachability probe
each step** (the `can_reach` DFS/BFS with early termination). Graphs here are sparse
(small in-degree), so this is inexpensive.

---

## 8. Irreducibility and an optional second move

For the sampler to be correct, the moves must be able to reach **every** valid DAG
in the space from any starting point (irreducibility). Single-edge rewiring usually
suffices, but the `out_degree(p) == 1` guard can strand configurations (a node
whose single out-edge can never be moved). If exploration looks stuck, add a
**double edge swap** as a second move type, chosen with some probability each step:

```
pick two distinct edges (a -> b) and (c -> d) with b != d
if a != c and not G.has_edge(a, d) and not G.has_edge(c, b)
   and swapping creates no cycle:
       replace them with (a -> d) and (c -> b)
```

The double swap preserves **both** in- and out-degrees, is also symmetric (so it
keeps the simple accept-if-legal rule), and — crucially — it doesn't reduce
anyone's out-degree, so it can move edges that the single rewire is forbidden to
touch. Single rewire (which *does* change out-degrees) plus the occasional double
swap together explore the full fixed-in-degree space.

(If you adopt the §3 "counts-only, degrees free" target instead, you'd add
edge-insertion / edge-deletion moves with analogous category-preserving guards.
Those change `|E|`, so the proposal is no longer symmetric and you must use a real
`min(1, N_cur / N_prop)` acceptance ratio. Keep that case separate from the simple
recipe above.)

---

## 9. Practical / implementation notes

- **Data structures.** Adjacency lists for the forward search; per-node **parent
  set** (for O(1) `has_edge(p_new, z)` and duplicate checks); per-node out-degree
  counter (for the `out_degree(p) == 1` guard); an edge list or array for uniform
  edge sampling in O(1).
- **Cost per step** is dominated by `can_reach`, worst case O(descendants of z) but
  usually far less due to early termination. Sparsity is your friend.
- **Seeding.** Start the chain from one of your existing rejection-sampled graphs
  (or any valid graph). A realistic seed shortens burn-in.
- **Correlated samples.** Consecutive graphs are nearly identical. Burn in first,
  then thin (skip a block of steps between emitted samples) so samples are roughly
  independent.
- **Mixing time is not known in closed form.** Tune `BURN_IN` and `THIN`
  empirically (§10). Budget for "many cheap steps" rather than "one expensive
  build."

---

## 10. Validation plan

1. **Tiny-case uniformity.** For small R/T/L, brute-force enumerate every valid DAG
   in the space. Run the chain a long time, histogram which graph it's in, and
   confirm frequencies are ~equal (chi-square goodness-of-fit). This directly
   verifies both correctness of `legal` and uniformity.
2. **Bias diagnostic at scale.** Track the statistic that exposed the original
   problem — e.g. the maximum over nodes of (ancestor count / N). Under the old
   growth process it was wildly state-dependent. Here it should settle into a
   stable distribution and stop trending; use that to pick `BURN_IN` (run until the
   statistic stops drifting) and `THIN` (run until its autocorrelation decays).
3. **Invariant assertions.** After every emitted sample, assert: exact R/T/L
   counts, acyclic (a topological sort succeeds), in-degree sequence unchanged,
   no duplicate edges, no self-loops.

---

## 11. Decisions left for you

- Exact in-degree rule for trunks and leaves (the construction implied trunks = 2,
  leaves = 1) — this defines the space and the seed.
- Whether out-degrees should be constrained or free (§3 / §8).
- Whether you need *exact* uniform (consider coupling-from-the-past, much harder) or
  *approximately* uniform after mixing (the normal MCMC outcome) is acceptable.
- `BURN_IN`, `THIN`, and whether to include the double-swap move — all tuned via
  §10 against your target size and throughput.
- Whether the *identity* of which nodes are roots should also be randomized; if so,
  apply a role-respecting relabeling to each emitted sample.
