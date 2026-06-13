# Handoff: Random DAG Construction by Edge-Splitting

This document is a self-contained spec for an algorithm we designed in a prior
conversation. It is written so a fresh Claude Code instance can implement it
without that history. Read the whole thing before coding — several "obvious"
shortcuts are wrong, and the reasons are spelled out so they aren't reinvented.

---

## 1. Goal

Generate random DAGs with a **fixed number of root, trunk, and leaf nodes**.

- **Roots**: in-degree 0 (sources).
- **Leaves**: out-degree 0 (sinks).
- **Trunks**: everything else (in-degree 2, out-degree ≥ 1).

Performance target: graphs of **a few thousand nodes**, generated **many times per
second**. So per-build cost in the low millions of operations, in a compiled
language or with vectorized/bitset ops.

---

## 2. The construction process

1. **Initialize.** Create the desired roots and leaves, with each leaf connected
   to a root. (Exact initial wiring is a design choice — e.g. each leaf to a
   random root. Keep it simple; the splitting step does the real work.)

2. **Insert trunk nodes by splitting edges.** Repeat until the trunk count is hit.
   Pick a random existing edge `u → v`. Insert a new trunk node `w`, replacing
   that edge with `u → w → v`. Then give `w` a **second parent**: pick a node `x`
   and add `x → w`.

   So every trunk `w` ends up with in-degree 2 (`u` from the split, `x` from the
   second parent) and out-degree ≥ 1.

3. The node `x` must be chosen **uniformly at random over all valid candidates**,
   where "valid" means the edge `x → w` does not create a cycle *and* does not
   break the fixed root/trunk/leaf counts (see §3).

---

## 3. What "valid candidate for x" means (read carefully)

`x → w` creates a cycle iff `w` can already reach `x`, i.e. iff `x` is a
**descendant of `w`**. Since `w`'s only descendants at insertion time are `v` and
`v`'s descendants:

> **forbidden(x) = {v} ∪ descendants(v)**   (cycle avoidance)

On top of that, to keep the counts fixed:

- **Exclude leaves from being `x`.** Giving a leaf an outgoing edge turns it into
  a trunk and breaks the leaf count. Eligible sources are **roots ∪ trunks** only.
- **Exclude `x = u`** to avoid a parallel edge into `w`.
- Roots are fine as `x` (they just accumulate out-edges; they keep in-degree 0).

> **valid(x) = (roots ∪ trunks) \ ({u, v} ∪ descendants(v))**

`x` is sampled **uniformly** from `valid(x)`. Getting this uniformity right is the
crux of the whole problem.

---

## 4. The trap to avoid (do NOT use a static topological order)

The tempting shortcut: maintain a topological order as a list, insert `w` near its
parent, and only allow `x` to be a node earlier in the list. This guarantees
acyclicity but **is biased and must not be used.**

Why: a linear order conflates two different relationships — "`x` is an ancestor of
`w`" vs. "`x` is merely *incomparable* to `w` but happens to sit before it." The
set we actually want is `ancestors(w) ∪ all incomparable nodes`. No single linear
order can place *all* incomparable nodes before `w`, because incomparable nodes are
exactly the ones free to go on either side, and a fixed order commits each to one
side. So "predecessors in the order" is always a strict, biased subset of
"non-descendants," and the dropped nodes are precisely the ones that distinguish
true uniformity from the shortcut.

Takeaway: the candidate set is *defined by reachability*, so any exactly-uniform
sampler must consult reachability structure. There is no ordering/labeling trick
that avoids this. Don't spend time looking for one.

---

## 5. The insight that makes it efficient

You do **not** have to recompute descendants from scratch each split.

**Splitting `u → v` into `u → w → v` changes reachability among existing nodes by
nothing** — every old path through that edge just detours through `w`. The *only*
reachability change in the entire operation is the single new cross-edge `x → w`.
And it is monotone: edges/pairs are only ever added, never removed.

That is the ideal setting for **incremental transitive closure**. Maintain a
descendant set `desc[a]` per node, and update only when the cross-edge is added.
Adding `x → w` means `x` and everything that can reach `x` newly reach
`{w} ∪ desc[w]`, where `desc[w] = {v} ∪ desc[v]`. Propagate upward through parent
pointers, pruning wherever nothing is new:

```
# After choosing x, before/while wiring x -> w:
new = {w} ∪ desc[v]               # newly reachable set contributed by this edge
desc[w] = {v} ∪ desc[v]
stack = [x]
while stack:
    a = stack.pop()
    added = new - desc[a]         # genuinely new descendants for this ancestor
    if not added:
        continue                  # prune: a (and its ancestors) already had these
    desc[a] |= added
    stack.extend(parents[a])      # parents recheck against their own desc
```

The prune is what keeps it cheap and correct: if `a` already had all of `new`, it
received those descendants earlier, and by induction so did its ancestors. Total
cost across a full build is **output-sensitive** — proportional to the number of
(node, descendant) reachability pairs actually created, not to n². For sparse,
in-degree-2 graphs the descendant sets tend to stay modest, so this is far below
the n² worst case. (This is the practical core of incremental-TC; refs: Italiano
1986; La Poutré & van Leeuwen.)

---

## 6. Sampling x once you have desc[v]

forbidden = `{u, v} ∪ desc[v]`; eligible pool = roots ∪ trunks.

- **Descendant set small (common):** rejection-sample. Draw a random eligible node,
  reject if it's in `desc[v]` or equals `u`. Expected attempts ≈ |eligible| / |valid|
  ≈ O(1).
- **Descendant set large (v near a root):** materialize the complement and take the
  k-th eligible-and-not-forbidden element. With bitsets this is O(n / 64).

Pick the branch based on |desc[v]| at split time.

---

## 7. Data structure options

| Approach | Reachability query | Memory | Best when |
|---|---|---|---|
| Per-node **bitset** `desc[]` | O(1) | O(n²) bits (~12 MB at n=10⁴) | dense reachability; want trivial complement sampling |
| Per-node **hash set** `desc[]` | O(1) avg | O(actual pairs) | descendant sets small; truly output-sensitive |
| **Roaring bitmaps** | O(1) | adaptive | middle ground; good default if a lib is handy |
| **Lazy BFS per split** (no maintenance) | — recompute | O(n) | simplest; graph is sparse so BFS only touches descendants; competitive unless many splits hit large-descendant nodes |

Recommended starting point: **lazy BFS per split** for a correct baseline (simple,
low memory, sparse graph ⇒ cheap), then switch to **incremental TC** (bitset or
roaring) if profiling shows splits are repeatedly hitting nodes with large
descendant sets. The two are duals; incremental TC wins specifically in that case.

Keep **parent pointers** (≤ 2 per node) regardless — needed for the upward
propagation in §5 and cheap to maintain.

---

## 8. Edge cases / gotchas checklist

- [ ] Eligible sources for `x` are roots ∪ trunks only — **never leaves** (count invariant).
- [ ] Exclude `x = u` (no parallel edge into `w`).
- [ ] Roots must never gain an in-edge — verify the init wiring and that `w` is always a *new* node, never a root, so roots stay in-degree 0.
- [ ] Leaves must never gain an out-edge — guaranteed by excluding them as `x`; splitting an edge *into* a leaf keeps it a leaf (it still has out-degree 0).
- [ ] Handle the case where `valid(x)` is empty (very small graphs / extreme parameter ratios) — decide whether to retry a different edge, skip, or error.
- [ ] If you keep an explicit edge list for "pick a random edge," remember a split removes one edge and adds two (`u→w`, `w→v`) plus the cross-edge `x→w`.
- [ ] `desc[w]` must be initialized (`{v} ∪ desc[v]`) so future splits see `w` correctly.

---

## 9. Caveat on "truly random" (worth a design decision)

Uniform-per-split-over-non-descendants defines **a** specific, reasonable
distribution over DAGs — but it is **not** the uniform distribution over all DAGs
with the given root/trunk/leaf counts. The order in which edges happen to be split
biases which graphs are reachable and how often. Before optimizing hard, confirm
this per-split-uniform process is the target distribution you actually want; if you
need uniform-over-all-DAGs or some other measure, the whole approach changes.

---

## 10. Suggested first steps for implementation

1. Confirm language and the exact root/trunk/leaf parameters and init wiring.
2. Implement the **lazy-BFS baseline** end to end; assert acyclicity (e.g. a topo
   sort succeeds) and that final in/out-degree counts match the root/trunk/leaf spec.
3. Add a uniformity sanity check on small graphs: enumerate `valid(x)` by brute
   force and confirm the sampler's empirical distribution is flat.
4. Profile at the target n; if splits on large-descendant nodes dominate, swap in
   incremental TC (§5) with bitsets or roaring bitmaps.
5. Benchmark builds/sec against the target.
