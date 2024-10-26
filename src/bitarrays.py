from typing import Tuple

import numpy as np
from PIL import Image


def calculate_address_bitdepth(shape: Tuple[int]):
    total = np.prod(shape)
    bitdepth = int(np.ceil(np.log2(total + 1)))
    return bitdepth


def printbin(x):
    for i in x:
        print("".join(str(j) for j in i))


def get_address_bitarrays(shape: Tuple[int]):
    total = np.prod(shape)
    # dtype="<u8" means a little-endian 8-byte (64-bit) unsigned integer
    addresses = np.arange(0, total, dtype=">u8")
    addresses = np.expand_dims(addresses, axis=-1)
    addresses = addresses.view(np.uint8)

    bitdepth = calculate_address_bitdepth(shape)
    print("bitdepth", bitdepth)

    unpacked = np.unpackbits(addresses, axis=-1)[..., -bitdepth:]
    unpacked = unpacked.transpose()
    repacked = np.packbits(unpacked, axis=-1)

    return repacked


def output_to_image_array(
    output: np.ndarray[np.uint8], shape: Tuple[int]
) -> np.ndarray[np.uint8]:
    total = np.prod(shape)
    output = np.unpackbits(
        output, axis=-1
    )  # num_outputs x batch_size//8 -> num_outputs x batch_size
    output = output[
        :, :total
    ]  # trim off any padding from if the address count is not a multiple of 8
    output = output.transpose()  # batch_size x num_outputs

    # output = np.sum(output, axis=-1).astype(np.uint8)  # batch_size x 1
    output = np.packbits(output, axis=-1)  # batch_size x 1
    output = output.reshape(shape)  # original image shape
    # output = output.view(np.uint8)

    return output
