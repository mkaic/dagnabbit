"""Bitwise logic gates, indexed by trunk node type.

A gate takes the stacked values of a node's parents -- ``[rows, in_degree,
num_words]`` of packed bits -- and reduces the ``in_degree`` axis to a single
``[rows, num_words]`` value. Writing gates as reductions rather than binary
functions keeps them correct for any in-degree, so a trunk type whose
in-degree is not 2 needs no special casing at the callsite.

The tuple index *is* the trunk node type: ``GATE_OPERATORS[t]`` is the gate for
trunk type ``t``, matching the ``[0, num_trunk_node_types)`` slice of the type
layout in :mod:`dagnabbit.dag.graphs`. Order is therefore load-bearing --
appending is safe, reordering silently relabels every graph ever generated.

Which gates are in play is a configuration choice, made once in
:mod:`dagnabbit.scripts.config` as a :class:`GateSet`. Everything that has to
agree about the gate set derives from that one object rather than restating it:
the geometry's ``num_trunk_node_types`` and ``trunk_node_in_degrees``, the
operators evaluation dispatches through, the names rendering prints, and the
type ids the hand-built circuits are written against. Those used to be four
hand-maintained constants that a reordered gate set would silently desynchronize
-- the reference adder in particular hardcoded "NAND is type 0, XOR is type 1"
and would have quietly built a different circuit.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

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


def conjunction(values: Tensor) -> Tensor:
    """AND. Named in full because ``and`` is a keyword."""
    return _reduce(values, torch.bitwise_and)


def disjunction(values: Tensor) -> Tensor:
    """OR. Named in full because ``or`` is a keyword."""
    return _reduce(values, torch.bitwise_or)


def nand(values: Tensor) -> Tensor:
    return torch.bitwise_not(_reduce(values, torch.bitwise_and))


def nor(values: Tensor) -> Tensor:
    return torch.bitwise_not(_reduce(values, torch.bitwise_or))


def xor(values: Tensor) -> Tensor:
    return _reduce(values, torch.bitwise_xor)


def xnor(values: Tensor) -> Tensor:
    return torch.bitwise_not(_reduce(values, torch.bitwise_xor))


# Every gate that can be named in a config. Being in the catalogue says nothing
# about being in use -- see GateSet and the measurement table below.
GATE_CATALOGUE: dict[str, GateOperator] = {
    "AND": conjunction,
    "OR": disjunction,
    "NAND": nand,
    "NOR": nor,
    "XOR": xor,
    "XNOR": xnor,
}


@dataclass(frozen=True)
class GateSet:
    """The gates in play, and the trunk type each one owns.

    Holds names rather than function objects so it stays a plain, comparable,
    picklable description -- the operators are looked up from
    :data:`GATE_CATALOGUE` on demand. Position in ``names`` *is* the trunk type
    id, so :meth:`index_of` is how anything that needs a specific gate (the
    reference adder wanting NAND) asks for it without assuming an order.

    Use :func:`gate_set` to build one; it fills in the usual in-degree.
    """

    names: tuple[str, ...]
    in_degrees: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.names:
            raise ValueError("a gate set needs at least one gate")
        if len(self.names) != len(self.in_degrees):
            raise ValueError(
                f"{len(self.in_degrees)} in-degrees for {len(self.names)} gates"
            )
        unknown = [name for name in self.names if name not in GATE_CATALOGUE]
        if unknown:
            raise ValueError(
                f"unknown gate(s) {unknown}; available: {sorted(GATE_CATALOGUE)}"
            )
        if len(set(self.names)) != len(self.names):
            raise ValueError(f"duplicate gate in {self.names}")
        if any(in_degree < 1 for in_degree in self.in_degrees):
            raise ValueError("every gate needs in-degree >= 1")

    @property
    def operators(self) -> tuple[GateOperator, ...]:
        """What :func:`~dagnabbit.tasks.logic_gates.evaluate.evaluate_choices` wants."""
        return tuple(GATE_CATALOGUE[name] for name in self.names)

    @property
    def num_types(self) -> int:
        """The geometry's ``num_trunk_node_types``."""
        return len(self.names)

    def index_of(self, name: str) -> int:
        """The trunk type id of ``name``, or a clear error if it is not in use.

        The error matters more than the lookup: a circuit written in terms of a
        gate the configured set does not contain cannot be built at all, and
        saying so beats emitting a wrong circuit against whatever sits at that
        index.
        """
        try:
            return self.names.index(name)
        except ValueError:
            raise ValueError(
                f"this configuration has no {name} gate; its gates are "
                f"{list(self.names)}"
            ) from None


def gate_set(*names: str, in_degrees: Sequence[int] | None = None) -> GateSet:
    """``gate_set("NAND", "XOR", "XNOR")``. In-degree defaults to 2 per gate."""
    return GateSet(
        names=tuple(names),
        in_degrees=tuple(in_degrees) if in_degrees is not None else (2,) * len(names),
    )


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
# The table is what "configurable" is for: these rows are reachable by editing
# one line of config, e.g. gate_set("NAND", "NOR") or gate_set("AND", "NAND").
#
# Index == trunk node type. See the module docstring before touching the order.
DEFAULT_GATE_SET = gate_set("NAND", "XOR", "XNOR")

# Library-level defaults, for tests and for callers with no config in hand.
# Training does not read these -- it reads ``config.GATES`` -- so changing the
# configured set does not silently leave a call site on the old one.
GATE_OPERATORS: tuple[GateOperator, ...] = DEFAULT_GATE_SET.operators
GATE_NAMES: tuple[str, ...] = DEFAULT_GATE_SET.names
