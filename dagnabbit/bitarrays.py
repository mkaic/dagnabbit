from typing import Tuple

import numpy as np


def calculate_required_bitdepth(x: Tuple[int]):
    return int(np.ceil(np.log2(x + 1)))


def printbin(x):
    for i in x:
        print("".join(str(j) for j in i))


def get_address_bitarrays(shape: Tuple[int]):
    ndim = len(shape)

    # dtype=">u8" means a big-endian 8-byte (64-bit) unsigned integer
    unpacked_address_bitarrays = []
    total_bitdepth = 0
    for i, dimension in enumerate(shape):
        bitdepth = calculate_required_bitdepth(dimension)
        addresses = np.arange(0, dimension, dtype=">u8")
        addresses = np.expand_dims(addresses, axis=-1)  # uint64 [dimension, 1]

        # from https://numpy.org/doc/stable/reference/generated/numpy.ndarray.view.html:
        # For a.view(some_dtype), if some_dtype has a different number of bytes per entry than the previous dtype (for example, converting a regular array to a structured array), then the last axis of a must be contiguous. This axis will be resized in the result.
        addresses = addresses.view(np.uint8)  # uint8 [total, 8]

        unpacked = np.unpackbits(addresses, axis=-1)  # {1, 0} uint8 [total, 64]
        unpacked = unpacked[..., -bitdepth:]  # {1, 0} uint8 [total, bitdepth]
        unpacked = unpacked.transpose()  # {1, 0} uint8 [bitdepth, dimension]

        # This reshape brought to you by my friend ChatGPT
        unpacked = unpacked.reshape(
            (bitdepth,)
            + (1,) * i  # leading singleton dims
            + (dimension,)  # dimension of this axis
            + (1,) * (ndim - i - 1)  # trailing singleton dims
        )
        # (bitdepth, *shape)
        unpacked = np.broadcast_to(unpacked, (bitdepth,) + shape)
        total_bitdepth += bitdepth
        unpacked_address_bitarrays.append(unpacked)

    # (sum_bitdepths, *shape)
    concatenated = np.concatenate(unpacked_address_bitarrays, axis=0)
    concatenated = concatenated.reshape((total_bitdepth, -1))

    repacked = np.packbits(
        concatenated, axis=-1
    )  # uint8 [sum_bitdepths, ceil(total / 8)]

    return repacked


def output_to_image_array(
    output: np.ndarray[np.uint8], shape: Tuple[int]
) -> np.ndarray[np.uint8]:
    total = np.prod(shape)

    # output: [batch_size, num_output_bits, ceil(total / 8)]

    # [batch_size, num_output_bits, ceil(total / 8)] -> [batch_size, num_output_bits, ceil(total / 8) * 8]
    output = np.unpackbits(output, axis=-1)

    # trim off any padding from if the address count is not a multiple of 8
    output = output[:, :, :total]  # [batch_size, num_output_bits, total]
    output = output.transpose(0, 2, 1)  # [batch_size, total, num_output_bits]

    # for this project, num_output_bits will almost always be 8, so last dim will be 1 here.
    output = np.packbits(
        output, axis=2
    )  # [batch_size, total, ceil(num_output_bits / 8)]
    output = output.reshape(shape)  # original image shape

    return output


def get_8bit_adder_truth_table(return_intermediates: bool = False):
    input_a = np.arange(256, dtype=np.uint8)
    input_b = np.arange(256, dtype=np.uint8)

    grid = np.stack(np.meshgrid(input_a, input_b), axis=-1)

    sums = grid.sum(axis=-1, dtype=np.uint8, keepdims=True)

    unpacked_inputs = np.unpackbits(grid, axis=-1)
    unpacked_inputs = np.moveaxis(unpacked_inputs, -1, 0)
    # (16, 256, 256) - bit planes before flattening
    unpacked_inputs_2d = unpacked_inputs
    unpacked_inputs = unpacked_inputs.reshape(16, -1)
    packed_inputs = np.packbits(unpacked_inputs, axis=-1)

    unpacked_sums = np.unpackbits(sums, axis=-1)
    unpacked_sums = np.moveaxis(unpacked_sums, -1, 0)
    # (8, 256, 256) - bit planes before flattening
    unpacked_sums_2d = unpacked_sums
    unpacked_sums = unpacked_sums.reshape(8, -1)
    packed_sums = np.packbits(unpacked_sums, axis=-1)

    if return_intermediates:
        return {
            "grid": grid,  # (256, 256, 2)
            "sums": sums,  # (256, 256, 1)
            "unpacked_inputs_2d": unpacked_inputs_2d,  # (16, 256, 256)
            "unpacked_sums_2d": unpacked_sums_2d,  # (8, 256, 256)
            "unpacked_inputs": unpacked_inputs,  # (16, 65536)
            "unpacked_sums": unpacked_sums,  # (8, 65536)
            "packed_inputs": packed_inputs,  # (16, 8192)
            "packed_sums": packed_sums,  # (8, 8192)
        }
    else:
        return packed_inputs, packed_sums


if __name__ == "__main__":
    intermediates = get_8bit_adder_truth_table(return_intermediates=True)
    print(intermediates)
