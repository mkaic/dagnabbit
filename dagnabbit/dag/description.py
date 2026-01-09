import logging

import torch
from jaxtyping import UInt16, UInt8
from torch import Tensor

logger = logging.getLogger(__name__)

AVAILABLE_GATE_TYPES = [
    "NAND",
    "NOR",
]
NUM_GATE_TYPES = len(AVAILABLE_GATE_TYPES)


def format_list_with_n_digits(lst: list, n: int) -> str:
    new_list = [f"{i:0{n}d}" for i in lst]
    return ",".join(new_list)


class BinaryLogicGateDAGDescription:
    gate_inputs: UInt16[
        Tensor, "2 num_gates"
    ]  # must be contiguous, will be offset by number of inputs
    gate_types: UInt8[Tensor, "num_gates"]  # must be contiguous

    def __init__(
        self,
        gate_inputs: UInt16[Tensor, "2 num_gates"],
        gate_types: UInt8[Tensor, "num_gates"],
    ):
        self.gate_inputs = gate_inputs.to(torch.uint16)
        self.gate_types = gate_types.to(torch.uint8)
        assert self.gate_inputs.is_contiguous()
        assert self.gate_types.is_contiguous()

    @classmethod
    def random(
        cls, num_inputs: int, num_gates: int, lookback: int
    ) -> "BinaryLogicGateDAGDescription":
        # gates inputs may reference any previous gate within the lookback window, or the inputs.
        maximum_gate_input_lookback = (
            torch.arange(num_gates, dtype=torch.float32) + num_inputs
        )
        maximum_gate_input_lookback = maximum_gate_input_lookback.clamp(
            min=0, max=lookback
        )
        logger.debug(
            "maximum_gate_input_lookback: %s",
            format_list_with_n_digits(maximum_gate_input_lookback.int().tolist(), 2),
        )

        noise = torch.rand(2, num_gates, dtype=torch.float32)
        sampled_gate_lookbacks = noise * maximum_gate_input_lookback

        # min = 1, max = lookback
        sampled_gate_lookbacks = sampled_gate_lookbacks.ceil().int()

        logger.debug(
            "sampled_gate_lookbacks[0]: %s",
            format_list_with_n_digits(sampled_gate_lookbacks[0].tolist(), 2),
        )
        logger.debug(
            "sampled_gate_lookbacks[1]: %s",
            format_list_with_n_digits(sampled_gate_lookbacks[1].tolist(), 2),
        )

        gate_positions = (torch.arange(num_gates) + num_inputs).int()
        logger.debug(
            "gate_positions: %s", format_list_with_n_digits(gate_positions.tolist(), 2)
        )

        sampled_gate_input_indices = gate_positions - sampled_gate_lookbacks

        logger.debug(
            "sampled_gate_input_indices[0]: %s",
            format_list_with_n_digits(sampled_gate_input_indices[0].tolist(), 2),
        )
        logger.debug(
            "sampled_gate_input_indices[1]: %s",
            format_list_with_n_digits(sampled_gate_input_indices[1].tolist(), 2),
        )

        sampled_gate_types = torch.randint(
            0, NUM_GATE_TYPES, (num_gates,), dtype=torch.uint8
        )
        logger.debug(
            "sampled_gate_types: %s",
            format_list_with_n_digits(sampled_gate_types.tolist(), 2),
        )

        return cls(sampled_gate_input_indices, sampled_gate_types)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    for _ in range(10):
        dag = BinaryLogicGateDAGDescription.random(
            num_inputs=4, num_gates=16, lookback=8
        )
        print("\n")
