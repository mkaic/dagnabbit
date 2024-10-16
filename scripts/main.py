from argparse import ArgumentParser

import numpy as np
from PIL import Image
from tqdm import tqdm

from ..src import gates as GF
from ..src.bitarrays import (
    calculate_address_bitdepth,
    get_address_bitarrays,
    output_to_np,
)
from ..src.dag import ComputationGraph, Node

parser = ArgumentParser()
parser.add_argument("-g", "--num_gates", type=int, default=512)
parser.add_argument(
    "-i", "--image_path", type=str, default="dagnabbit-torch/test_images/branos.png"
)
parser.add_argument("-r", "--resize", type=int, default=64)
args = parser.parse_args()

image = Image.open(args.image_path).convert("RGB")
image = image.resize((args.resize, args.resize))
image = np.array(image)
original_shape = image.shape

print("image.shape", original_shape)

address_bitdepth = calculate_address_bitdepth(original_shape)

graph = ComputationGraph(num_gates=args.num_gates, num_inputs=address_bitdepth)

address_bitarrays = get_address_bitarrays(original_shape)

best_loss = np.inf
last_updated_at = best_loss

progress_bar = tqdm(range(100_000))
for epoch in range(1_000):
    permutation = np.random.permutation(args.num_gates)
    for i, gate_idx in enumerate(permutation):
        mutant: Node = graph.gate_nodes[gate_idx]
        old_function = mutant.logical_function
        mutant.logical_function = np.random.choice(GF.AVAILABLE_FUNCTIONS)

        output = graph.evaluate(address_bitarrays)
        output = output_to_np(output, original_shape)

        loss = np.sqrt(
            np.mean(np.square(image.astype(np.float32) - output.astype(np.float32)))
        )

        if loss < best_loss:
            best_loss = loss
            print(
                f"RMSE: {best_loss:.5f} | Epoch: {epoch:04} | Step: {i:04} | Last Saved At: {last_updated_at:.5f}"
            )

            if best_loss < last_updated_at * 0.995:
                output_pil = Image.fromarray(output)
                output_pil.save(
                    "dagnabbit-torch/test_images/output.jpg",
                    format="JPEG",
                    subsampling=0,
                    quality=100,
                )

                last_updated_at = best_loss

        elif loss == best_loss:
            pass
        else:
            mutant.logical_function = old_function
