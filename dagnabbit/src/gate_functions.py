import numpy as np


def NP_AND(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.bitwise_and(a, b)


def NP_NAND(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.bitwise_not(np.bitwise_and(a, b))


def NP_OR(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.bitwise_or(a, b)


def NP_NOR(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.bitwise_not(np.bitwise_or(a, b))


def NP_XOR(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.bitwise_xor(a, b)


def NP_XNOR(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.bitwise_not(np.bitwise_xor(a, b))


def NP_NOT(a: np.ndarray) -> np.ndarray:
    return np.bitwise_not(a)


AVAILABLE_FUNCTIONS = [
    NP_AND,
    NP_NAND,
    NP_OR,
    NP_NOR,
    NP_XOR,
    NP_XNOR,
    # NP_NOT
]
