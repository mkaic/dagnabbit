import torch

VALID_OPERATORS = {
    "NAND": lambda x, y: torch.bitwise_not(torch.bitwise_and(x, y)),
    "NOR": lambda x, y: torch.bitwise_not(torch.bitwise_or(x, y)),
}
