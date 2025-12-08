import numpy as np


def NP_AND(a: np.ndarray[np.uint8], b: np.ndarray[np.uint8]) -> np.ndarray[np.uint8]:
    return np.bitwise_and(a, b)


def NP_NAND(a: np.ndarray[np.uint8], b: np.ndarray[np.uint8]) -> np.ndarray[np.uint8]:
    return np.bitwise_not(np.bitwise_and(a, b))


def NP_OR(a: np.ndarray[np.uint8], b: np.ndarray[np.uint8]) -> np.ndarray[np.uint8]:
    return np.bitwise_or(a, b)


def NP_NOR(a: np.ndarray[np.uint8], b: np.ndarray[np.uint8]) -> np.ndarray[np.uint8]:
    return np.bitwise_not(np.bitwise_or(a, b))


def NP_XOR(a: np.ndarray[np.uint8], b: np.ndarray[np.uint8]) -> np.ndarray[np.uint8]:
    return np.bitwise_xor(a, b)


def NP_XNOR(a: np.ndarray[np.uint8], b: np.ndarray[np.uint8]) -> np.ndarray[np.uint8]:
    return np.bitwise_not(np.bitwise_xor(a, b))


def NP_NOT_A(a: np.ndarray[np.uint8], b: np.ndarray[np.uint8]) -> np.ndarray[np.uint8]:
    return np.bitwise_not(a)


def NP_NOT_B(a: np.ndarray[np.uint8], b: np.ndarray[np.uint8]) -> np.ndarray[np.uint8]:
    return np.bitwise_not(b)


def NP_PASSTHROUGH_A(
    a: np.ndarray[np.uint8], b: np.ndarray[np.uint8]
) -> np.ndarray[np.uint8]:
    return a


def NP_PASSTHROUGH_B(
    a: np.ndarray[np.uint8], b: np.ndarray[np.uint8]
) -> np.ndarray[np.uint8]:
    return b


def NP_NOT_A_AND_B(
    a: np.ndarray[np.uint8], b: np.ndarray[np.uint8]
) -> np.ndarray[np.uint8]:
    return np.bitwise_and(np.bitwise_not(a), b)


def NP_A_AND_NOT_B(
    a: np.ndarray[np.uint8], b: np.ndarray[np.uint8]
) -> np.ndarray[np.uint8]:
    return np.bitwise_and(a, np.bitwise_not(b))


def NP_NOT_A_OR_B(
    a: np.ndarray[np.uint8], b: np.ndarray[np.uint8]
) -> np.ndarray[np.uint8]:
    return np.bitwise_or(np.bitwise_not(a), b)


def NP_A_OR_NOT_B(
    a: np.ndarray[np.uint8], b: np.ndarray[np.uint8]
) -> np.ndarray[np.uint8]:
    return np.bitwise_or(a, np.bitwise_not(b))


AVAILABLE_FUNCTIONS = [
    # NP_FALSE,
    # NP_TRUE,
    NP_AND,
    NP_NAND,
    NP_OR,
    NP_NOR,
    NP_XOR,
    NP_XNOR,
    # NP_NOT_A,
    # NP_NOT_B,
    # NP_PASSTHROUGH_A,
    # NP_PASSTHROUGH_B,
    NP_NOT_A_AND_B,
    NP_A_AND_NOT_B,
    NP_NOT_A_OR_B,
    NP_A_OR_NOT_B,
]
