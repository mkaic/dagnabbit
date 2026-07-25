"""Bitwise logic gates, indexed by trunk node type.

A gate takes the stacked values of a node's parents -- ``[rows, in_degree,
num_words]`` of packed bits -- and reduces the ``in_degree`` axis to a single
``[rows, num_words]`` value. Writing gates as reductions rather than binary
functions keeps them correct for any in-degree, so a trunk type whose
in-degree is not 2 needs no special casing at the callsite.

The tuple index *is* the trunk node type: ``GATE_OPERATORS[t]`` is the gate for
trunk type ``t``, matching the ``[0, num_trunk_node_types)`` slice of the type
layout in :mod:`dagnabbit.dag.description`. Order is therefore load-bearing --
appending is safe, reordering silently relabels every graph ever generated.
"""

from collections.abc import Callable

import torch
from torch import Tensor

GateOperator = Callable[[Tensor], Tensor]


def _reduce(values: Tensor, operator: Callable[[Tensor, Tensor], Tensor]) -> Tensor:
    """Fold ``operator`` across the in-degree axis of a [rows, K, words] tensor."""
    if values.shape[1] == 0:
        raise ValueError("cannot reduce a gate with in-degree 0")
    accumulator = values[:, 0]
    for slot in range(1, values.shape[1]):
        accumulator = operator(accumulator, values[:, slot])
    return accumulator


def nand(values: Tensor) -> Tensor:
    return torch.bitwise_not(_reduce(values, torch.bitwise_and))


def nor(values: Tensor) -> Tensor:
    return torch.bitwise_not(_reduce(values, torch.bitwise_or))


# Index == trunk node type. See the module docstring before touching the order.
GATE_OPERATORS: tuple[GateOperator, ...] = (nand, nor)
GATE_NAMES: tuple[str, ...] = ("NAND", "NOR")

# Retained for readability at callsites that want a gate by name (rendering,
# hand-built test circuits). Evaluation dispatches through GATE_OPERATORS.
VALID_OPERATORS: dict[str, GateOperator] = dict(zip(GATE_NAMES, GATE_OPERATORS))
