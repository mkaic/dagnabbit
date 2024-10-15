import numpy as np
from typing import Tuple

def calculate_address_bitdepth(shape:Tuple[int]):
    total = np.prod(shape)
    bitdepth = int(np.ceil(np.log2(total + 1)))
    return bitdepth

def printbin(x):
    for i in x:
        print("".join(str(j) for j in i))

def get_address_bitarrays(shape:Tuple[int]):
    total = np.prod(shape)
    # dtype="<u8" means a little-endian 8-byte (64-bit) unsigned integer
    addresses = np.arange(0, total, dtype=">u8")
    addresses = np.expand_dims(addresses, axis=-1)
    # view as byte array to allow us to use np.unpackbits
    addresses = addresses.view(np.uint8)

    bitdepth = calculate_address_bitdepth(shape)

    unpacked = np.unpackbits(addresses, axis=-1)[:, -bitdepth:]
    unpacked = unpacked.transpose()
    repacked = np.packbits(unpacked, axis=-1)
    
    return repacked
