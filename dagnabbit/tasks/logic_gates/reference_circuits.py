"""Hand-built circuits with known behaviour, for testing and as search targets.

These are constructed as ragged parent lists and canonicalized on the way out,
so their semantics are known independently of anything the model or the
evaluator does. :func:`nand_ripple_carry_adder` in particular is a
ground-truth optimum for the adder task: any search that works should be able
to reach a fitness of 1.0, because a circuit achieving it demonstrably exists
inside the graph budget.

Padding
-------
The model's geometry fixes the trunk node count (128 by default) while the
adder core needs only 67 gates. The spare capacity is spent on **identity
buffers** rather than dead gates, which matters whenever the circuit is used to
probe a trained model: the sampler's coverage pass guarantees every trunk node
has at least one child, so a circuit padded with
dead gates is off-distribution in a way that has nothing to do with what it
computes, and its padding gates are structural twins that the canonical
ordering cannot distinguish.

A buffer is transparent because ``NAND(x, x) = NOT x``, so an inverter feeding
a restorer returns the original signal. One inverter can be shared by several
restorers, which makes padding cost ``1 + (number of rerouted consumers)`` and
therefore reachable at either parity -- a chain of pure pairs could only ever
spend an even number of gates.

Because buffers are spliced into existing wires, gates cannot simply be
appended in index order; the circuit is built against symbolic ids and
topologically sorted at the end.
"""

from dataclasses import dataclass, field

from dagnabbit.dag.canonical import CanonicalGraphs, Geometry, from_lists

NAND_TYPE = 0


@dataclass
class AdderAnnotations:
    """Which gate does what, for rendering and for reading the circuit.

    Every index here is a *node storage index*, not a canonical position. The
    two coincide only for roots and outputs, whose positions are pinned.
    """

    bit_of_node: dict[int, int] = field(default_factory=dict)
    role_of_node: dict[int, str] = field(default_factory=dict)
    sum_node_of_bit: dict[int, int] = field(default_factory=dict)
    carry_node_of_bit: dict[int, int] = field(default_factory=dict)
    root_of_input: dict[tuple[str, int], int] = field(default_factory=dict)
    output_of_bit: dict[int, int] = field(default_factory=dict)
    buffer_nodes: set[int] = field(default_factory=set)
    core_gates: int = 0
    buffer_gates: int = 0

    @property
    def gates_used(self) -> int:
        return self.core_gates + self.buffer_gates

    @property
    def live_nodes(self) -> set[int]:
        return set(self.bit_of_node)

    @property
    def core_nodes(self) -> set[int]:
        return set(self.bit_of_node) - self.buffer_nodes


class _SymbolicCircuit:
    """Gates addressed by symbolic id, so wires can be rewired after the fact."""

    def __init__(self, num_root_nodes: int):
        self.num_root_nodes = num_root_nodes
        self.parents: dict[int, list[int]] = {}
        self.role: dict[int, str] = {}
        self.bit: dict[int, int] = {}
        self._next_id = num_root_nodes

    def gate(self, left: int, right: int, role: str, bit: int) -> int:
        gate_id = self._next_id
        self._next_id += 1
        self.parents[gate_id] = [left, right]
        self.role[gate_id] = role
        self.bit[gate_id] = bit
        return gate_id

    def is_root(self, node: int) -> bool:
        return node < self.num_root_nodes

    def consumers_of(self, source: int) -> list[tuple[int, int]]:
        """Every ``(gate_id, slot)`` currently reading ``source``."""
        return [
            (gate_id, slot)
            for gate_id, slots in self.parents.items()
            for slot, parent in enumerate(slots)
            if parent == source
        ]

    def topological_order(self) -> list[int]:
        """Gate ids in a deterministic order where parents precede children."""
        emitted: set[int] = set()
        order: list[int] = []
        pending = sorted(self.parents)
        while pending:
            progressed = False
            still_pending = []
            for gate_id in pending:
                if all(
                    self.is_root(parent) or parent in emitted
                    for parent in self.parents[gate_id]
                ):
                    emitted.add(gate_id)
                    order.append(gate_id)
                    progressed = True
                else:
                    still_pending.append(gate_id)
            if not progressed:
                raise AssertionError("cycle in a hand-built circuit")
            pending = still_pending
        return order


def _insert_buffer(
    circuit: _SymbolicCircuit,
    source: int,
    consumers: list[tuple[int, int]],
) -> int:
    """Splice a shared-inverter buffer between ``source`` and ``consumers``.

    Returns the number of gates spent: one inverter plus one restorer per
    rerouted consumer. Every new gate has a child by construction, and
    ``source`` keeps one (the inverter).
    """
    bit = circuit.bit.get(source, 0)
    inverter = circuit.gate(source, source, "buffer (invert)", bit)
    for gate_id, slot in consumers:
        restorer = circuit.gate(inverter, inverter, "buffer (restore)", bit)
        circuit.parents[gate_id][slot] = restorer
    return 1 + len(consumers)


def build_nand_ripple_carry_adder(
    num_root_nodes: int = 16,
    num_trunk_nodes: int = 128,
    num_output_nodes: int = 8,
    num_trunk_node_types: int = 2,
) -> tuple[CanonicalGraphs, AdderAnnotations]:
    """An 8-bit ripple-carry adder, padded to the trunk budget with buffers.

    Scores exactly 1.0 on :func:`~dagnabbit.tasks.logic_gates.evaluate.adder_task`
    and contains **no dead gates**.

    Bit-order convention follows the truth table: roots 0-7 are the bits of
    ``a`` most significant first, roots 8-15 the bits of ``b``, and output j is
    bit ``7 - j`` of the sum. So for bit position ``k`` counting from the LSB,
    ``a_k`` is root ``7 - k``, ``b_k`` is root ``15 - k``, and the sum bit goes
    to output ``7 - k``.

    Each stage is the standard 9-NAND full adder; the least significant bit
    needs no carry-in and uses a 5-NAND half adder, and the most significant
    bit omits its carry-out, which ``(a + b) mod 256`` discards -- that gate
    would otherwise be the circuit's only childless node.
    """
    if num_root_nodes % 2 != 0:
        raise ValueError("an adder needs an even number of input bits")
    width = num_root_nodes // 2
    if num_output_nodes != width:
        raise ValueError(
            f"{num_root_nodes} inputs implies a {width}-bit adder, but "
            f"{num_output_nodes} outputs were requested"
        )

    circuit = _SymbolicCircuit(num_root_nodes)
    annotations = AdderAnnotations()
    sum_bits: dict[int, int] = {}
    carry_bits: dict[int, int] = {}
    carry: int | None = None

    for bit in range(width):
        a = (width - 1) - bit
        b = (num_root_nodes - 1) - bit
        annotations.root_of_input[("a", bit)] = a
        annotations.root_of_input[("b", bit)] = b
        is_last = bit == width - 1

        # Half adder: xor_ab = a XOR b, and n1 = NOT (a AND b).
        n1 = circuit.gate(a, b, "¬(a∧b)", bit)
        xor_ab = circuit.gate(
            circuit.gate(a, n1, "a·¬(a∧b)", bit),
            circuit.gate(b, n1, "b·¬(a∧b)", bit),
            "a⊕b",
            bit,
        )

        if carry is None:
            sum_bits[bit] = xor_ab
            circuit.role[xor_ab] = "sum = a⊕b"
            if not is_last:
                carry = circuit.gate(n1, n1, "carry out = a∧b", bit)
        else:
            n4 = circuit.gate(xor_ab, carry, "¬((a⊕b)∧cin)", bit)
            sum_bits[bit] = circuit.gate(
                circuit.gate(xor_ab, n4, "(a⊕b)·¬(…)", bit),
                circuit.gate(carry, n4, "cin·¬(…)", bit),
                "sum = a⊕b⊕cin",
                bit,
            )
            if not is_last:
                # NAND(n1, n4) == (a AND b) OR (xor_ab AND carry): the carry-out.
                carry = circuit.gate(n1, n4, "carry out", bit)

        if not is_last:
            carry_bits[bit] = carry

    annotations.core_gates = len(circuit.parents)
    output_sources = [sum_bits[(width - 1) - slot] for slot in range(num_output_nodes)]

    graph, final_index = _pad_and_emit(
        circuit=circuit,
        annotations=annotations,
        output_sources=output_sources,
        num_root_nodes=num_root_nodes,
        num_trunk_nodes=num_trunk_nodes,
        num_output_nodes=num_output_nodes,
        num_trunk_node_types=num_trunk_node_types,
    )
    for bit, gate_id in sum_bits.items():
        annotations.sum_node_of_bit[bit] = final_index[gate_id]
    for bit, gate_id in carry_bits.items():
        annotations.carry_node_of_bit[bit] = final_index[gate_id]
    return graph, annotations


def build_nand_bitwise_xor(
    num_root_nodes: int = 16,
    num_trunk_nodes: int = 128,
    num_output_nodes: int = 8,
    num_trunk_node_types: int = 2,
) -> tuple[CanonicalGraphs, AdderAnnotations]:
    """Bitwise ``a XOR b``: the same inputs and outputs, no long-range structure.

    Deliberately the adder's opposite. Output j depends on exactly two input
    bits through four NAND gates, so nothing has to travel and there is no
    carry chain -- the whole circuit is three ranks deep before padding.

    It exists as a contrast for the round-trip probe. If a checkpoint can
    round-trip this but not the adder, the failure is specifically about
    long-range dependencies; if it can round-trip neither, the problem is
    structured circuits in general.
    """
    if num_root_nodes % 2 != 0:
        raise ValueError("bitwise XOR needs an even number of input bits")
    width = num_root_nodes // 2
    if num_output_nodes != width:
        raise ValueError(
            f"{num_root_nodes} inputs implies {width} output bits, but "
            f"{num_output_nodes} were requested"
        )

    circuit = _SymbolicCircuit(num_root_nodes)
    annotations = AdderAnnotations()
    xor_bits: dict[int, int] = {}

    for bit in range(width):
        a = (width - 1) - bit
        b = (num_root_nodes - 1) - bit
        annotations.root_of_input[("a", bit)] = a
        annotations.root_of_input[("b", bit)] = b
        n1 = circuit.gate(a, b, "¬(a∧b)", bit)
        xor_bits[bit] = circuit.gate(
            circuit.gate(a, n1, "a·¬(a∧b)", bit),
            circuit.gate(b, n1, "b·¬(a∧b)", bit),
            "a⊕b",
            bit,
        )

    annotations.core_gates = len(circuit.parents)
    output_sources = [xor_bits[(width - 1) - slot] for slot in range(num_output_nodes)]
    graph, final_index = _pad_and_emit(
        circuit=circuit,
        annotations=annotations,
        output_sources=output_sources,
        num_root_nodes=num_root_nodes,
        num_trunk_nodes=num_trunk_nodes,
        num_output_nodes=num_output_nodes,
        num_trunk_node_types=num_trunk_node_types,
    )
    for bit, gate_id in xor_bits.items():
        annotations.sum_node_of_bit[bit] = final_index[gate_id]
    return graph, annotations


def _pad_and_emit(
    circuit: _SymbolicCircuit,
    annotations: AdderAnnotations,
    output_sources: list[int],
    num_root_nodes: int,
    num_trunk_nodes: int,
    num_output_nodes: int,
    num_trunk_node_types: int,
) -> tuple[CanonicalGraphs, dict[int, int]]:
    """Pad a core circuit to the trunk budget with buffers, then emit it.

    Returns the description and the symbolic-id -> final-index map, which the
    caller needs to translate its own annotations.
    """
    spare = num_trunk_nodes - annotations.core_gates
    if spare < 0:
        raise ValueError(
            f"the circuit needs {annotations.core_gates} gates but only "
            f"{num_trunk_nodes} trunk nodes are available"
        )
    if spare == 1:
        raise ValueError(
            "cannot spend exactly one spare trunk node transparently; the "
            "cheapest buffer costs two gates"
        )

    # Interleave candidate wires across bit positions. Walking gates in id
    # order would spend the whole budget on the first few stages, leaving the
    # circuit lopsided -- deep and buffer-heavy at the LSB, bare at the MSB.
    gates_by_bit: dict[int, list[int]] = {}
    for gate in sorted(circuit.parents):
        if circuit.consumers_of(gate):
            gates_by_bit.setdefault(circuit.bit[gate], []).append(gate)
    buffer_sources = [
        gates_by_bit[bit][depth]
        for depth in range(max(len(gates) for gates in gates_by_bit.values()))
        for bit in sorted(gates_by_bit)
        if depth < len(gates_by_bit[bit])
    ]
    source_index = 0
    while spare > 0:
        source = buffer_sources[source_index % len(buffer_sources)]
        source_index += 1
        consumers = circuit.consumers_of(source)
        if not consumers:
            continue
        # Cost is 1 + len(rerouted). Leaving exactly one gate over would be
        # unspendable, so never take a step that lands there.
        take = min(len(consumers), max(1, spare - 1))
        if spare - (1 + take) == 1:
            take -= 1
            if take < 1:
                continue
        spare -= _insert_buffer(circuit, source, consumers[:take])

    annotations.buffer_gates = len(circuit.parents) - annotations.core_gates

    order = circuit.topological_order()
    if len(order) != num_trunk_nodes:
        raise AssertionError(
            f"built {len(order)} trunk gates, expected {num_trunk_nodes}"
        )
    final_index = {
        gate_id: num_root_nodes + position for position, gate_id in enumerate(order)
    }

    def resolve(node: int) -> int:
        return node if circuit.is_root(node) else final_index[node]

    node_types = [num_trunk_node_types + i for i in range(num_root_nodes)]
    node_inputs_indices: list[list[int]] = [[] for _ in range(num_root_nodes)]
    for gate_id in order:
        node_inputs_indices.append([resolve(p) for p in circuit.parents[gate_id]])
        node_types.append(NAND_TYPE)
        index = final_index[gate_id]
        annotations.bit_of_node[index] = circuit.bit[gate_id]
        annotations.role_of_node[index] = circuit.role[gate_id]
        if circuit.role[gate_id].startswith("buffer"):
            annotations.buffer_nodes.add(index)

    output_type = num_trunk_node_types + num_root_nodes
    width = num_output_nodes
    for slot, source in enumerate(output_sources):
        node_inputs_indices.append([resolve(source)])
        node_types.append(output_type)
        annotations.output_of_bit[(width - 1) - slot] = len(node_inputs_indices) - 1

    geometry = Geometry(
        num_root_nodes=num_root_nodes,
        num_trunk_nodes=num_trunk_nodes,
        num_output_nodes=num_output_nodes,
        num_trunk_node_types=num_trunk_node_types,
        trunk_node_in_degrees=(2,) * num_trunk_node_types,
    )
    return from_lists(node_inputs_indices, node_types, geometry), final_index


def nand_ripple_carry_adder(geometry: Geometry) -> CanonicalGraphs:
    """The adder circuit alone, as a batch of one.

    See :func:`build_nand_ripple_carry_adder` for the construction.
    """
    return build_nand_ripple_carry_adder(
        num_root_nodes=geometry.num_root_nodes,
        num_trunk_nodes=geometry.num_trunk_nodes,
        num_output_nodes=geometry.num_output_nodes,
        num_trunk_node_types=geometry.num_trunk_node_types,
    )[0]
