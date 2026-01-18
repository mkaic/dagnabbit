import torch
from jaxtyping import UInt16, UInt8
from torch import Tensor

from dagnabbit.dag.operators import VALID_OPERATORS

NUM_GATE_TYPES = len(VALID_OPERATORS.keys())


def format_list_with_n_digits(lst: list, n: int) -> str:
    new_list = [f"{i:0{n}d}" for i in lst]
    return ",".join(new_list)


class BinaryLogicGateDAGDescription:
    gate_inputs_array_indices: UInt16[
        Tensor, "2 num_gates"
    ]  # must be contiguous, will be offset by number of root nodes
    gate_types: UInt8[Tensor, "num_gates"]  # must be contiguous
    lookback: int
    num_root_nodes: int
    num_gates: int

    def __init__(
        self,
        gate_inputs_array_indices: UInt16[Tensor, "2 num_gates"],
        gate_types: UInt8[Tensor, "num_gates"],
        lookback: int,
        num_root_nodes: int,
    ):
        self.gate_inputs_array_indices = gate_inputs_array_indices.to(torch.uint16)
        self.gate_types = gate_types.to(torch.uint8)
        self.lookback = lookback
        self.num_root_nodes = num_root_nodes
        self.num_gates = gate_types.shape[0]
        assert self.gate_inputs_array_indices.is_contiguous()
        assert self.gate_types.is_contiguous()
        assert self.lookback > 0

        self.leaf_node_indices = self.identify_leaf_nodes()

    def identify_leaf_nodes(self) -> list[int]:
        """
        Identify all leaf nodes in the DAG.

        A leaf node is a node whose output is not referenced as an input to any gate.
        Returns array indices of all leaf nodes as a list of integers.
        """
        total_nodes = self.num_root_nodes + self.num_gates
        all_node_indices = torch.arange(total_nodes, dtype=torch.int32)

        # Get all referenced indices (flatten the 2 x num_gates tensor)
        referenced_indices = self.gate_inputs_array_indices.flatten().to(torch.int32)

        # Mark which nodes are referenced
        is_referenced = torch.zeros(total_nodes, dtype=torch.bool)
        is_referenced[referenced_indices] = True

        # Leaf nodes are those NOT referenced by any gate
        leaf_node_indices = all_node_indices[~is_referenced].tolist()

        return leaf_node_indices

    def __str__(self) -> str:
        return f"""
BinaryLogicGateDAGDescription(
    num_root_nodes={self.num_root_nodes},
    num_gates={self.num_gates},
    lookback={self.lookback},
    gate_inputs_array_indices
    {format_list_with_n_digits(self.gate_inputs_array_indices[0].tolist(), 2)},
    {format_list_with_n_digits(self.gate_inputs_array_indices[1].tolist(), 2)},
    gate_types
    {format_list_with_n_digits(self.gate_types.tolist(), 2)},
)"""

    @classmethod
    # @torch.compile()
    def random(
        cls, num_root_nodes: int, num_gates: int, lookback: int
    ) -> "BinaryLogicGateDAGDescription":
        """
        Generate a random DAG where each gate can reference:
          - Any of the `num_root_nodes` root nodes (indices 0 to num_root_nodes-1)
          - Any previous gate within the `lookback` window

        The DAG layout (example with num_root_nodes=4, num_gates=16, lookback=8):

            Index:  0   1   2   3   4   5   6   ...  11  12  13  ...  19
                    [ root nodes ]   [           gates                   ]
                    R0  R1  R2  R3  G0  G1  G2  ...  G7  G8  G9  ... G15

        Gate G0 (index 4) can reference: root nodes R0-R3 (no previous gates exist)
        Gate G5 (index 9) can reference: root nodes R0-R3, gates G0-G4 (5 gates, all within lookback)
        Gate G8 (index 12) can reference: root nodes R0-R3, gates G0-G7 (8 gates = lookback limit)
        Gate G9 (index 13) can reference: root nodes R0-R3, gates G1-G8 (lookback window slides)
        Gate G15 (index 19) can reference: root nodes R0-R3, gates G7-G14 (lookback window)
        """

        gate_indices = torch.arange(num_gates, dtype=torch.int32)

        # How many valid references does each gate have?
        # - Always has num_root_nodes root node references
        # - Has min(gate_index, lookback) gate references
        # For early gates: gate_index + num_root_nodes total references
        # For late gates: lookback + num_root_nodes total references (capped)
        #
        # Example (num_root_nodes=4, lookback=8):
        #   G0: 4 refs, G1: 5 refs, ..., G8+: 12 refs (capped at 4+8)
        num_valid_references = (gate_indices + num_root_nodes).clamp(
            max=lookback + num_root_nodes
        )

        # Sample uniformly from [0, num_valid_references - 1] for each gate.
        # Since noise is in [0, 1), (noise * n).int() gives integers in [0, n-1].
        noise = torch.rand(2, num_gates, dtype=torch.float32)
        sampled_slots = (noise * num_valid_references.float()).floor().int()

        # The sample space is divided into two regions:
        #   slots [0, num_root_nodes - 1]    -> references a root node (position = slot)
        #   slots [num_root_nodes, max - 1]  -> references a previous gate
        #
        # Note: "position" = index in the full array (root nodes + gates).
        #       Gate i is at position (num_root_nodes + i).
        #
        # For GATE references, the valid window depends on gate_index:
        #   "Early" gates (gate_index < lookback): can reference gates 0 to gate_index-1
        #   "Late" gates (gate_index >= lookback): can reference gates (gate_index - lookback) to (gate_index - 1)
        #
        # For early gates: position = slot (maps directly to the full array)
        # For late gates: position = slot + (gate_index - lookback)
        #   The offset shifts the gate references forward because the first
        #   (gate_index - lookback) gates are no longer in the valid window.
        #
        # Example for G24 (gate_index=24, lookback=4, num_root_nodes=8):
        #   num_valid_refs = min(24 + 8, 4 + 8) = 12, so slots are [0, 11]
        #   Slots 0-7 -> root nodes at positions 0-7
        #   Slots 8-11 -> gates G20-G23 at positions 28-31
        #
        #   If slot=9, we want position 29 (gate G21):
        #   late_gate_offset = 24 - 4 = 20
        #   position = 9 + 20 = 29 ✓ (which is gate 29 - 8 = G21)

        # This is 0 for early gates, positive for late gates
        late_gate_offset = (gate_indices - lookback).clamp(min=0)

        is_gate_reference = sampled_slots >= num_root_nodes

        sampled_gate_input_indices = torch.where(
            condition=is_gate_reference,
            input=sampled_slots + late_gate_offset,
            other=sampled_slots,
        )

        sampled_gate_types = torch.randint(
            0, NUM_GATE_TYPES, (num_gates,), dtype=torch.uint8
        )

        return cls(
            gate_inputs_array_indices=sampled_gate_input_indices,
            gate_types=sampled_gate_types,
            lookback=lookback,
            num_root_nodes=num_root_nodes,
        )

    def to(self, device: torch.device) -> "BinaryLogicGateDAGDescription":
        self.gate_inputs_array_indices = self.gate_inputs_array_indices.to(device)
        self.gate_types = self.gate_types.to(device)
        return self


if __name__ == "__main__":
    import time

    num_root_nodes = 12
    num_gates = 4096
    lookback = 256
    num_iterations = 10_000

    # # Warmup
    # for _ in range(100):
    #     dag = BinaryLogicGateDAGDescription.random(num_root_nodes, num_gates, lookback)

    # start = time.perf_counter()
    # for _ in range(num_iterations):
    #     dag = BinaryLogicGateDAGDescription.random(num_root_nodes, num_gates, lookback)
    # elapsed = time.perf_counter() - start

    # print(f"Created {num_iterations} DAGs in {elapsed:.3f}s")
    # print(f"  {num_iterations / elapsed:.0f} DAGs/sec")
    # print(f"  {elapsed / num_iterations * 1e6:.2f} µs/DAG")

    # Validation test
    print("\nValidating 100 DAGs...")
    num_test_dags = 100
    for dag_idx in range(num_test_dags):
        dag = BinaryLogicGateDAGDescription.random(num_root_nodes, num_gates, lookback)

        # Check gate_types are valid
        assert dag.gate_types.max() < NUM_GATE_TYPES, (
            f"DAG {dag_idx}: gate_type out of range"
        )

        # Check each gate's inputs are valid references
        for gate_idx in range(num_gates):
            for input_slot in range(2):
                ref = dag.gate_inputs_array_indices[input_slot, gate_idx].item()

                # Valid references: root nodes [0, num_root_nodes) or previous gates within lookback
                is_root_node_ref = ref < num_root_nodes

                if is_root_node_ref:
                    continue  # Root node references are always valid

                # It's a gate reference - check it's within the valid window
                referenced_gate = ref - num_root_nodes  # Convert to gate index
                min_valid_gate = max(0, gate_idx - lookback)
                max_valid_gate = gate_idx - 1

                assert min_valid_gate <= referenced_gate <= max_valid_gate, (
                    f"DAG {dag_idx}, gate {gate_idx}, slot {input_slot}: "
                    f"references gate {referenced_gate}, valid range is [{min_valid_gate}, {max_valid_gate}]"
                )

    print(f"All {num_test_dags} DAGs passed validation ✓")
