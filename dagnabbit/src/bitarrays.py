from typing import Tuple

import numpy as np


def calculate_address_bitdepth(shape: Tuple[int]):
    total = np.prod(shape)
    bitdepth = int(np.ceil(np.log2(total + 1)))
    return bitdepth


def printbin(x):
    for i in x:
        print("".join(str(j) for j in i))


def get_address_bitarrays(shape: Tuple[int]):
    total = np.prod(shape)
    # dtype=">u8" means a big-endian 8-byte (64-bit) unsigned integer
    addresses = np.arange(0, total, dtype=">u8")  # uint64 [total]
    addresses = np.expand_dims(addresses, axis=-1)  # uint64 [total, 1]

    # from https://numpy.org/doc/stable/reference/generated/numpy.ndarray.view.html:
    # For a.view(some_dtype), if some_dtype has a different number of bytes per entry than the previous dtype (for example, converting a regular array to a structured array), then the last axis of a must be contiguous. This axis will be resized in the result.
    addresses = addresses.view(np.uint8)  # uint8 [total, 8]

    bitdepth = calculate_address_bitdepth(shape)
    print("bitdepth", bitdepth)

    unpacked = np.unpackbits(addresses, axis=-1)  # {1, 0} uint8 [total, 64]
    unpacked = unpacked[..., -bitdepth:]  # {1, 0} uint8 [total, bitdepth]
    unpacked = unpacked.transpose()  # {1, 0} uint8 [bitdepth, total]
    repacked = np.packbits(unpacked, axis=-1)  # uint8 [bitdepth, ceil(total / 8)]

    return repacked


def output_to_image_array(
    output: np.ndarray[np.uint8], shape: Tuple[int]
) -> np.ndarray[np.uint8]:
    total = np.prod(shape)

    # [num_output_bits, ceil(total / 8)] -> [num_output_bits, ceil(total / 8) * 8]
    output = np.unpackbits(output, axis=-1)

    # trim off any padding from if the address count is not a multiple of 8
    output = output[:, :total]  # [num_output_bits, total]
    output = output.transpose()  # [total, num_output_bits]

    # for this project, num_output_bits will almost always be 8, so last dim will be 1 here.
    output = np.packbits(output, axis=-1)  # [total, ceil(num_output_bits / 8)]
    output = output.reshape(shape)  # original image shape

    return output
