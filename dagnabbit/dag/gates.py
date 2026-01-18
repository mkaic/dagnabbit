import torch

AVAILABLE_GATE_TYPES = ["NAND", "NOR"]

BITWISE_GATE_FUNCTIONS = {
    "NAND": lambda x, y: torch.bitwise_not(torch.bitwise_and(x, y)),
    "NOR": lambda x, y: torch.bitwise_not(torch.bitwise_or(x, y)),
}
