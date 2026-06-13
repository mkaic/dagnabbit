"""Diagnostics for the MCMC edge-rewiring DAG sampler.

Run directly:

    uv run python -m dagnabbit.scripts.validate_mcmc_uniformity
"""

import argparse
import math
import random
from collections import Counter, deque
from itertools import product

from dagnabbit.dag.description import (
    _EdgeSplitGraph,
    _build_edge_split_graph,
    _mcmc_rewire,
)


def _raw_key(graph: _EdgeSplitGraph) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(parents) for parents in graph.parents)


def _children_from_parents(parents: list[list[int]]) -> list[list[int]]:
    children: list[list[int]] = [[] for _ in parents]
    for node, parent_list in enumerate(parents):
        for parent in parent_list:
            children[parent].append(node)
    return children


def _is_acyclic(parents: list[list[int]], children: list[list[int]]) -> bool:
    in_degrees = [len(parent_list) for parent_list in parents]
    ready = deque(node for node, in_degree in enumerate(in_degrees) if in_degree == 0)
    visited = 0

    while ready:
        node = ready.popleft()
        visited += 1
        for child in children[node]:
            in_degrees[child] -= 1
            if in_degrees[child] == 0:
                ready.append(child)

    return visited == len(parents)


def _enumerate_tiny_states(
    num_root_nodes: int,
    num_trunk_nodes: int,
    num_output_nodes: int,
    trunk_in_degree: int,
) -> set[tuple[tuple[int, ...], ...]]:
    num_nodes = num_root_nodes + num_trunk_nodes + num_output_nodes
    leaf_nodes = list(range(num_root_nodes, num_root_nodes + num_output_nodes))
    trunk_nodes = list(range(num_root_nodes + num_output_nodes, num_nodes))
    nonleaf = list(range(num_root_nodes)) + trunk_nodes

    parent_choices: list[list[tuple[int, ...]]] = []
    variable_nodes = leaf_nodes + trunk_nodes
    for node in variable_nodes:
        in_degree = 1 if node in leaf_nodes else trunk_in_degree
        candidates = [parent for parent in nonleaf if parent != node]
        parent_choices.append(list(product(candidates, repeat=in_degree)))

    states: set[tuple[tuple[int, ...], ...]] = set()
    for assignment in product(*parent_choices):
        parents: list[list[int]] = [[] for _ in range(num_nodes)]
        for node, parent_tuple in zip(variable_nodes, assignment):
            parents[node] = list(parent_tuple)

        children = _children_from_parents(parents)
        if any(children[leaf] for leaf in leaf_nodes):
            continue
        if any(not children[node] for node in nonleaf):
            continue
        if not _is_acyclic(parents, children):
            continue
        states.add(tuple(tuple(parent_list) for parent_list in parents))

    return states


def _regularized_gamma_q(a: float, x: float) -> float:
    if x <= 0.0:
        return 1.0

    eps = 3.0e-14
    fpmin = 1.0e-300
    gln = math.lgamma(a)

    if x < a + 1.0:
        term = 1.0 / a
        total = term
        ap = a
        for _ in range(1_000):
            ap += 1.0
            term *= x / ap
            total += term
            if abs(term) < abs(total) * eps:
                p = total * math.exp(-x + a * math.log(x) - gln)
                return max(0.0, min(1.0, 1.0 - p))
        p = total * math.exp(-x + a * math.log(x) - gln)
        return max(0.0, min(1.0, 1.0 - p))

    b = x + 1.0 - a
    c = 1.0 / fpmin
    d = 1.0 / max(b, fpmin)
    h = d
    for i in range(1, 1_000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < fpmin:
            d = fpmin
        c = b + an / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break

    q = math.exp(-x + a * math.log(x) - gln) * h
    return max(0.0, min(1.0, q))


def _chi_square_p_value(observed: list[int]) -> tuple[float, float]:
    expected = sum(observed) / len(observed)
    if expected == 0.0:
        return 0.0, 1.0
    chi_square = sum((count - expected) ** 2 / expected for count in observed)
    degrees_of_freedom = len(observed) - 1
    p_value = _regularized_gamma_q(degrees_of_freedom / 2.0, chi_square / 2.0)
    return chi_square, p_value


def _tiny_uniformity(args: argparse.Namespace) -> None:
    states = _enumerate_tiny_states(
        num_root_nodes=args.tiny_roots,
        num_trunk_nodes=args.tiny_trunks,
        num_output_nodes=args.tiny_outputs,
        trunk_in_degree=args.tiny_in_degree,
    )
    rng = random.Random(args.seed)
    raw = _build_edge_split_graph(
        num_root_nodes=args.tiny_roots,
        num_trunk_nodes=args.tiny_trunks,
        num_output_nodes=args.tiny_outputs,
        num_trunk_node_types=1,
        trunk_node_in_degrees=[args.tiny_in_degree],
        rng=rng,
    )

    histogram: Counter[tuple[tuple[int, ...], ...]] = Counter()
    for step in range(args.tiny_steps):
        _mcmc_rewire(raw, num_steps=1, rng=rng)
        if step >= args.tiny_burn_in and (step - args.tiny_burn_in) % args.tiny_thin == 0:
            histogram[_raw_key(raw)] += 1

    missing = states - set(histogram)
    extra = set(histogram) - states
    observed = [histogram[state] for state in sorted(states)]
    chi_square, p_value = _chi_square_p_value(observed)

    print("Tiny-case slot-aware uniformity")
    print(f"  enumerated states: {len(states)}")
    print(f"  sampled states:    {len(histogram)}")
    print(f"  missing states:    {len(missing)}")
    print(f"  out-of-space keys: {len(extra)}")
    print(f"  samples:           {sum(observed)}")
    print(f"  chi-square:        {chi_square:.3f}")
    print(f"  p-value:           {p_value:.6f}")


def _ancestor_count(parents: list[list[int]], node: int) -> int:
    seen: set[int] = set()
    stack = list(parents[node])
    while stack:
        parent = stack.pop()
        if parent in seen:
            continue
        seen.add(parent)
        stack.extend(parents[parent])
    return len(seen)


def _max_ancestor_fraction(graph: _EdgeSplitGraph) -> float:
    num_nodes = len(graph.parents)
    return max(_ancestor_count(graph.parents, node) / num_nodes for node in range(num_nodes))


def _bias_diagnostic(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    raw = _build_edge_split_graph(
        num_root_nodes=args.scale_roots,
        num_trunk_nodes=args.scale_trunks,
        num_output_nodes=args.scale_outputs,
        num_trunk_node_types=1,
        trunk_node_in_degrees=[args.scale_in_degree],
        rng=rng,
    )

    print("\nScale bias diagnostic")
    print("  step,max_ancestor_fraction")
    for step in range(0, args.scale_steps + 1, args.scale_interval):
        if step:
            _mcmc_rewire(raw, num_steps=args.scale_interval, rng=rng)
        print(f"  {step},{_max_ancestor_fraction(raw):.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--tiny-roots", type=int, default=2)
    parser.add_argument("--tiny-trunks", type=int, default=2)
    parser.add_argument("--tiny-outputs", type=int, default=2)
    parser.add_argument("--tiny-in-degree", type=int, default=2)
    parser.add_argument("--tiny-steps", type=int, default=1_000_000)
    parser.add_argument("--tiny-burn-in", type=int, default=100_000)
    parser.add_argument("--tiny-thin", type=int, default=10)

    parser.add_argument("--scale-roots", type=int, default=8)
    parser.add_argument("--scale-trunks", type=int, default=128)
    parser.add_argument("--scale-outputs", type=int, default=8)
    parser.add_argument("--scale-in-degree", type=int, default=2)
    parser.add_argument("--scale-steps", type=int, default=20_000)
    parser.add_argument("--scale-interval", type=int, default=2_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _tiny_uniformity(args)
    _bias_diagnostic(args)


if __name__ == "__main__":
    main()
