"""Hand-built circuits with known behaviour, for testing and as search targets.

These are constructed directly as :class:`FixedInDegreeDAGDescription`s rather
than generated, so their semantics are known independently of anything the
model or the evaluator does. :func:`nand_ripple_carry_adder` in particular is a
ground-truth optimum for the adder task: any search that works should be able
to reach a fitness of 1.0, because a circuit achieving it demonstrably exists
inside the graph budget.
"""

from dagnabbit.dag.description import FixedInDegreeDAGDescription

NAND_TYPE = 0


def nand_ripple_carry_adder(
    num_root_nodes: int = 16,
    num_trunk_nodes: int = 128,
    num_output_nodes: int = 8,
    num_trunk_node_types: int = 2,
) -> FixedInDegreeDAGDescription:
    """An 8-bit ripple-carry adder built from 68 NAND gates.

    Scores exactly 1.0 on :func:`~dagnabbit.tasks.logic_gates.evaluate.adder_task`.

    Bit-order convention follows the truth table: roots 0-7 are the bits of
    ``a`` most significant first, roots 8-15 the bits of ``b``, and output j is
    bit ``7 - j`` of the sum. So for bit position ``k`` counting from the LSB,
    ``a_k`` is root ``7 - k``, ``b_k`` is root ``15 - k``, and the sum bit goes
    to output ``7 - k``.

    Each stage is the standard 9-NAND full adder; the least significant bit
    needs no carry-in and uses a 5-NAND half adder. The final carry-out is
    discarded, matching the truth table's ``(a + b) mod 256``. Trunk nodes
    beyond the 68 used are filled with dead gates to meet the fixed node count.
    """
    if num_root_nodes % 2 != 0:
        raise ValueError("an adder needs an even number of input bits")
    width = num_root_nodes // 2
    if num_output_nodes != width:
        raise ValueError(
            f"{num_root_nodes} inputs implies a {width}-bit adder, but "
            f"{num_output_nodes} outputs were requested"
        )

    root_types_start = num_trunk_node_types
    output_type = num_trunk_node_types + num_root_nodes

    node_types = [root_types_start + i for i in range(num_root_nodes)]
    node_inputs_indices: list[list[int]] = [[] for _ in range(num_root_nodes)]

    def nand(left: int, right: int) -> int:
        node_inputs_indices.append([left, right])
        node_types.append(NAND_TYPE)
        return len(node_inputs_indices) - 1

    sum_bits: dict[int, int] = {}
    carry: int | None = None
    for bit in range(width):
        a = (width - 1) - bit
        b = (num_root_nodes - 1) - bit

        # Half adder: xor_ab = a XOR b, and n1 = NOT (a AND b).
        n1 = nand(a, b)
        xor_ab = nand(nand(a, n1), nand(b, n1))

        if carry is None:
            sum_bits[bit] = xor_ab
            carry = nand(n1, n1)  # NOT n1 == a AND b
        else:
            n4 = nand(xor_ab, carry)
            sum_bits[bit] = nand(nand(xor_ab, n4), nand(carry, n4))
            # NAND(n1, n4) == (a AND b) OR (xor_ab AND carry), the carry-out.
            carry = nand(n1, n4)

    gates_used = len(node_inputs_indices) - num_root_nodes
    if gates_used > num_trunk_nodes:
        raise ValueError(
            f"the adder needs {gates_used} gates but only {num_trunk_nodes} "
            "trunk nodes are available"
        )
    # Dead filler: the description requires exactly num_trunk_nodes trunk nodes.
    while len(node_inputs_indices) - num_root_nodes < num_trunk_nodes:
        nand(0, 1)

    for output_slot in range(num_output_nodes):
        node_inputs_indices.append([sum_bits[(width - 1) - output_slot]])
        node_types.append(output_type)

    return FixedInDegreeDAGDescription(
        num_root_nodes=num_root_nodes,
        num_trunk_nodes=num_trunk_nodes,
        num_output_nodes=num_output_nodes,
        num_trunk_node_types=num_trunk_node_types,
        trunk_node_in_degrees=[2] * num_trunk_node_types,
        node_inputs_indices=node_inputs_indices,
        node_types=node_types,
    )
