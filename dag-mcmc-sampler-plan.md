# Plan: Replace edge-split DAG sampling with MCMC edge-rewiring

## Goal

Replace the random-DAG generator used in training so that graphs are produced by
**MCMC edge-rewiring** (per `dag-mcmc-sampler-handoff.md`) instead of relying on
the raw edge-split construction. The edge-split construction stays — but only as
the **seed** for the Markov chain, which then mixes toward a (near-)uniform sample
over DAGs with the same fixed counts and in-degrees.

The **output format must not change**: the public entry point
`make_random_graph_description(...)` must keep returning a
`FixedInDegreeDAGDescription` with the same invariants. Only the *distribution* of
graphs changes.

## Terminology mapping (read this first)

The handoff doc says "roots / trunks / leaves". This codebase calls them:

| handoff term | this repo's term | in-degree | out-degree |
|---|---|---|---|
| root | `ROOT` node | 0 | ≥ 0 |
| trunk | `TRUNK` node | fixed per trunk type (`trunk_node_in_degrees`) | ≥ 1 |
| leaf | `OUTPUT` node | 1 | 0 |

So "leaves" in the handoff are **OUTPUT nodes** here. `NONLEAF` = ROOT ∪ TRUNK.

## Decisions locked in for this implementation (override the handoff where they conflict)

1. **Multi-edges are legal.** A node may list the same parent more than once
   (parents are a *multiset*, not a set). Therefore **drop the handoff's
   "no duplicate edge" guard** (`G.has_edge(p_new, z)`). The sampling space is
   "labeled multigraph DAGs with a fixed per-node in-degree and fixed
   root/trunk/output role assignment." This is already legal in the current
   format — see `test_duplicate_parents_are_accepted` in
   `dagnabbit/dag/tests/test_edge_split_generation.py`.
2. **Per-trunk-type fixed in-degree.** Each trunk type keeps its own in-degree
   (`trunk_node_in_degrees[trunk_type]`). MCMC rewiring preserves every node's
   in-degree exactly, so this is honored automatically — no special handling.
3. **Seed the chain with the existing edge-split graph.** Use
   `_build_edge_split_graph(...)` to get a valid starting DAG, mix it, then assemble.
4. **Default mixing steps = number of trunk nodes** (`num_trunk_nodes`). Expose it
   as a parameter so it can be tuned/overridden.
5. Only the **single-edge rewire** move is required for v1. The optional
   double-swap (handoff §8) is a follow-up, not part of the core deliverable.

## Why this is correct (keep in mind while implementing)

- Rewiring removes `p→z` and adds `p_new→z`, so **z's in-degree is unchanged** ⇒
  every node keeps its role and in-degree forever (handoff §6).
- `p_new` is drawn only from `NONLEAF`, so outputs never gain an out-edge and stay
  leaves; the `out_degree(p) == 1` guard stops a trunk/root from collapsing to a leaf.
- Proposal is symmetric and the target is uniform ⇒ **accept iff legal** (no MH
  ratio). Multi-edges do not break symmetry: edges are sampled as *instances* from
  a constant-size edge list, and the reverse move `(p_new→z, new parent p)` always
  exists because `p` was a non-leaf with out-edge `p→z`. (handoff §5.)
- Reachability must be **recomputed each step** (a fresh `can_reach` DFS), not
  maintained incrementally — MCMC deletes edges, which breaks the monotone-TC trick
  used by the edge-split path. (handoff §7.)

---

## Phase 1 — Implement the MCMC rewiring core

All work in `dagnabbit/dag/description.py` (keeps the single-module pattern that
`_build_edge_split_graph` / `_assemble_edge_split_description` already use).

- [ ] **1.1 Add out-degree-aware reachability probe.** Add a helper
  `_can_reach(children: list[list[int]], src: int, dst: int) -> bool` that does a
  DFS/BFS forward from `src` over `children`, returning `True` as soon as it hits
  `dst`, `False` if exhausted. Early-terminate. (There is already
  `_descendants_from_children`; do **not** reuse it for the hot path — it computes
  the full descendant set with no early exit. A dedicated short-circuiting probe is
  the point of handoff §7.)

- [ ] **1.2 Add the MCMC driver** operating directly on an `_EdgeSplitGraph`:

  ```python
  def _mcmc_rewire(
      graph: _EdgeSplitGraph,
      num_steps: int,
      rng: random.Random,
  ) -> None:
      """In-place MCMC edge-rewiring (single-edge move). Preserves every node's
      role and in-degree; mixes toward a near-uniform sample over fixed-in-degree
      multigraph DAGs with the same root/trunk/leaf assignment."""
  ```

  Implementation notes:
  - Build `nonleaf = [n for n, r in enumerate(graph.roles) if r is not _EdgeSplitRole.LEAF]`.
  - Build a flat edge-instance list `edges: list[tuple[int, int]]` from
    `graph.parents`: for each node `z`, for each `p` in `graph.parents[z]`, append
    `(p, z)`. `len(edges) == sum of in-degrees` and is **constant** for the whole run.
  - Loop `num_steps` times:
    1. `i = rng.randrange(len(edges)); (p, z) = edges[i]`.
    2. `p_new = rng.choice(nonleaf)`.
    3. Reject (continue) if **any** of:
       - `p_new == z` (self-loop) or `p_new == p` (no-op).
       - `len(graph.children[p]) == 1` (p's last out-edge → would become a leaf).
       - `_can_reach(graph.children, z, p_new)` (adding `p_new→z` would close a cycle).
       - **NOTE:** do **not** add a duplicate-edge guard — multi-edges are legal (Decision 1).
    4. Accept: mutate consistently so `parents`, `children`, and `edges` stay in sync:
       - `edges[i] = (p_new, z)`
       - in `graph.parents[z]`, replace one occurrence of `p` with `p_new`
         (`graph.parents[z][graph.parents[z].index(p)] = p_new`).
       - `graph.children[p].remove(z)` (removes one instance).
       - `graph.children[p_new].append(z)`.
  - Out-degree is read as `len(graph.children[node])` (counts multiplicity) — no
    separate counter needed.

  Correctness invariant to preserve on every accepted move: for every `z`,
  `multiset(graph.parents[z]) == multiset of p over edges with child z`, and
  `children` is the transpose of `parents` with matching multiplicity. Because all
  instances of `p→z` are interchangeable, replacing "one occurrence" is sufficient.

- [ ] **1.3 Do not maintain `desc_masks` / incremental TC during MCMC.** That
  machinery is edge-split-only. MCMC uses the fresh `_can_reach` probe.

## Phase 2 — Wire MCMC into the public entry point

- [ ] **2.1 Extend `make_random_graph_description`** with a new optional parameter:

  ```python
  def make_random_graph_description(
      num_root_nodes: int,
      num_trunk_nodes: int,
      num_output_nodes: int,
      trunk_node_in_degrees: int | list[int],
      num_trunk_node_types: int,
      num_mixing_steps: int | None = None,   # NEW
  ) -> FixedInDegreeDAGDescription:
  ```

  Body changes:
  - After `_build_edge_split_graph(...)` produces `raw_graph`, default the step count:
    `steps = num_mixing_steps if num_mixing_steps is not None else num_trunk_nodes`
    (Decision 4).
  - Call `_mcmc_rewire(raw_graph, num_steps=steps, rng=rng)` **before**
    `_assemble_edge_split_description(...)`. Reuse the same `rng` so determinism is
    still driven entirely by the torch seed (`torch.manual_seed`).
  - Leave `_assemble_edge_split_description(...)` unchanged — it already topo-sorts
    from `parents`/`children` (multiplicity-aware) and relabels roots-first /
    trunks / outputs-last, producing the unchanged output format.

- [ ] **2.2 Add a config knob.** In `dagnabbit/scripts/config.py`, under
  `# --- DAG sampling ---`, add `NUM_MCMC_MIXING_STEPS: int | None = None`
  (None ⇒ use `num_trunk_nodes`). Pass it through at the training callsite
  `dagnabbit/scripts/train_encoder.py:230` as
  `num_mixing_steps=cfg.NUM_MCMC_MIXING_STEPS`.

- [ ] **2.3 Leave other callsites alone.** `make_random_graph_description` keeps
  working with no new arg (defaults to `num_trunk_nodes` mixing steps), so
  `roundtrip_blind_decode.py`, `diagnose_root_collapse.py`,
  `analyze_graph_distributions.py`, and the test harnesses need no changes to run.

## Phase 3 — Tests

Add to `dagnabbit/dag/tests/test_edge_split_generation.py` (or a sibling
`test_mcmc_rewire.py` following the same `uv run python -m ...` direct-run pattern).

- [ ] **3.1 Invariants survive mixing.** For several `(R, T, L, in-degrees)` configs
  and seeds, build a graph with mixing enabled and assert the **existing**
  `_assert_edge_split_invariants(graph)` still passes (exact R/T/L counts, root
  in-degree 0, per-trunk-type in-degree, outputs in-degree 1 / out-degree 0,
  topological ordering with `parent < node_idx`). This is the cheapest regression net.

- [ ] **3.2 In-degree sequence is preserved by `_mcmc_rewire`.** Build a raw
  `_EdgeSplitGraph`, snapshot `[len(p) for p in parents]`, run `_mcmc_rewire`, assert
  the per-node in-degree list is **identical**. Also assert roles are unchanged and
  every output node still has `children == []`.

- [ ] **3.3 Acyclicity.** After mixing, assert a topological sort of the raw graph
  succeeds (no cycle) — reuse the Kahn's-algorithm logic already in
  `_assemble_edge_split_description`, or just assert it doesn't raise.

- [ ] **3.4 Determinism.** Same torch seed ⇒ identical
  `(node_inputs_indices, node_types)`; different seed ⇒ different. (Mirror the
  existing `test_torch_seed_determinism`, but with mixing on.)

- [ ] **3.5 Multi-edges remain legal.** Construct a tiny case and confirm mixing
  neither crashes on nor forbids duplicate parents (the duplicate-edge guard must be
  absent).

- [ ] **3.6 The chain actually moves.** With `num_mixing_steps` large on a config
  that has room to rewire (e.g. several roots/trunks), assert the post-mix graph
  differs from the seed for at least some seeds (sanity that accepts happen).

## Phase 4 — Validation against uniformity (handoff §10)

Lightweight, not a training dependency. Put in a script, e.g.
`dagnabbit/scripts/validate_mcmc_uniformity.py`.

- [ ] **4.1 Tiny-case uniformity.** For a small config (e.g. R=2, T=2, L=2,
  in-degree 2), brute-force enumerate every valid fixed-in-degree multigraph DAG in
  the space (respecting the multiset-parents rule), run a long chain from one seed,
  histogram visited canonical graphs (reuse `canonicalize` /  a role-respecting key),
  and chi-square against uniform. Document the observed p-value in the script's output.

- [ ] **4.2 Bias diagnostic at scale.** Track `max over nodes of (ancestor_count / N)`
  across the chain; confirm it stops trending (use it to eyeball a reasonable
  `num_mixing_steps` for the production config in `config.py`). This is the statistic
  that exposed the original edge-split bias.

## Out of scope / follow-ups (do not implement now)

- Double-swap second move (handoff §8) for irreducibility — add only if §4.1 shows
  the single rewire fails to reach parts of the space.
- Counts-only / free-in-degree target (would require real MH acceptance ratios).
- Randomizing *which* nodes are roots beyond what relabeling already does.

## Acceptance criteria

- `make_random_graph_description(...)` returns a `FixedInDegreeDAGDescription` with
  all current invariants intact, now produced via edge-split-seed → MCMC mix → assemble.
- Default mixing-step count is `num_trunk_nodes`; overridable via parameter and
  `cfg.NUM_MCMC_MIXING_STEPS`.
- All existing tests still pass; new Phase 3 tests pass.
- `_mcmc_rewire` provably preserves per-node in-degree and role (verified by 3.2),
  and never creates a cycle (verified by 3.3).
