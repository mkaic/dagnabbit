from typing import Tuple

import torch
import torch.nn as nn
from torch.nn import ModuleList
from torch import Tensor, LongTensor, BoolTensor

from dagnabbit.bitarrays import output_to_image_array

BitTensor = Tensor
