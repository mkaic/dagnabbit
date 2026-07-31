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
    """Not in :data:`GATE_OPERATORS`. Kept for tests and for experiments."""
    return torch.bitwise_not(_reduce(values, torch.bitwise_or))


def xor(values: Tensor) -> Tensor:
    return _reduce(values, torch.bitwise_xor)


def xnor(values: Tensor) -> Tensor:
    return torch.bitwise_not(_reduce(values, torch.bitwise_xor))


# NAND, XOR, XNOR -- chosen by measurement, for behaviour at scale.
#
# A random circuit is only useful training data if its outputs stay informative.
# The number to watch is "free accuracy": what a predictor scores by emitting
# each output's marginal bit and never reading the graph. Measured over trunk
# sizes from 64 to 4096 gates:
#
#     gate set             64      512     4096
#     NAND, NOR         80.7%    88.4%    90.4%   <- degrades with scale
#     AND, NAND         80.3%    88.8%    91.0%
#     AND,NAND,XOR,XNOR 65.6%    66.8%    67.2%
#     NAND              63.8%    63.5%    61.2%
#     NAND, XOR         60.6%    60.8%    59.4%
#     NAND, XOR, XNOR   57.9%    57.1%    56.1%   <- improves with scale
#
# Two mechanisms decide this. First, whether constants *cascade*: NAND(0, y) = 1
# absorbs, but NAND(1, y) = NOT y does not, so under NAND a constant dies after
# one hop -- whereas NAND+NOR alternates 0 -> 1 -> 0 forever, which is why a
# third of its outputs are constant at 4096 gates. XOR and XNOR have no
# absorbing element at all, so adding them can only shorten cascades.
#
# Second, the mean-field fixed point of p -> P(gate output is 1). For NAND alone
# that is p = 1 - p^2, the golden ratio 0.618. Adding XOR (2p(1-p)) and XNOR
# (1 - 2p(1-p)) contributes a symmetric pair that cancels about one half, giving
# p = (2/3)(1 - p^2) = 0.5 exactly. Measured marginals land near 0.56 rather than
# 0.50 because reconvergent fan-in correlates a gate's two inputs, but the
# ordering the theory predicts is exactly the ordering observed.
#
# Index == trunk node type. See the module docstring before touching the order.
GATE_OPERATORS: tuple[GateOperator, ...] = (nand, xor, xnor)
GATE_NAMES: tuple[str, ...] = ("NAND", "XOR", "XNOR")

# Retained for readability at callsites that want a gate by name (rendering,
# hand-built test circuits). Evaluation dispatches through GATE_OPERATORS.
VALID_OPERATORS: dict[str, GateOperator] = dict(zip(GATE_NAMES, GATE_OPERATORS))
