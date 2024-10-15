import numpy as np
from PIL import Image
from argparse import ArgumentParser

from ..src.dag import ComputationGraph
from ..src.bitarrays import (
    calculate_address_bitdepth,
    get_address_bitarrays,
    output_to_pil,
)

parser = ArgumentParser()
parser.add_argument("-g", "--num_gates", type=int, default=512)
parser.add_argument("-i", "--image_path", type=str, default="dagnabbit-torch/test_images/branos.png")
args = parser.parse_args()

image = Image.open(args.image_path).convert("RGB")
image = image.resize((64, 64))
image = np.array(image)
image = np.moveaxis(image, -1, 0)

print("image.shape", image.shape)

address_bitdepth = calculate_address_bitdepth(image.shape)

graph = ComputationGraph(num_gates=args.num_gates, num_inputs=address_bitdepth)

address_bitarrays = get_address_bitarrays(image.shape)

output = graph.evaluate(address_bitarrays)


output_image = output_to_pil(output, image.shape)

output_image.save("dagnabbit-torch/test_images/output.png")
