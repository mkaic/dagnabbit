import shutil
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..src.bitarrays import (
    calculate_address_bitdepth,
    get_address_bitarrays,
    output_to_image_array,
)
from ..src.dag import ComputationGraph, random_dag
from ..src.model import (
    MLP,
    binary_to_integer,
    get_sinusoidal_position_encodings,
)

parser = ArgumentParser()
parser.add_argument("-g", "--num_compute_nodes", type=int, default=512)
parser.add_argument(
    "-i", "--image_path", type=str, default="dagnabbit/test_images/branos.png"
)
parser.add_argument("-r", "--resize", type=int, default=64)
args = parser.parse_args()

image = Image.open(args.image_path).convert("RGB")
image = image.resize((args.resize, args.resize))
image = np.array(image)
image = np.moveaxis(image, -1, 0)
original_shape = image.shape

print("image.shape", original_shape)

address_bitdepth = calculate_address_bitdepth(original_shape)
address_bitarrays = get_address_bitarrays(original_shape)

pos_enc = get_sinusoidal_position_encodings(
    length=args.num_compute_nodes * 2, dim=64, base=10000
)
mlp = MLP(layer_sizes=[64, 64, 64, args.num_compute_nodes + address_bitdepth])


best_loss = np.inf
last_updated_at = best_loss
update_counter = 0


shutil.rmtree("dagnabbit/outputs/timelapse", ignore_errors=True)
Path("dagnabbit/outputs/timelapse").mkdir(parents=True, exist_ok=True)

for step in range(1_000):

    logits = mlp(pos_enc)

    # each pair of two connection decisions is allowed to reference previous gates as well as input gates
    valid_indices = torch.arange(0, args.num_compute_nodes * 2) // 2
    for i in range(len(logits)):
        logits[i, address_bitdepth + i :] = -torch.inf

    probabilities = torch.softmax(logits, dim=-1)
    print(probabilities)

    samples = torch.multinomial(probabilities, 1)
    samples = samples.flatten()

    decisions = [tuple() for _ in range(address_bitdepth)]
    sample_pairs = zip(samples[::2], samples[1::2])
    decisions.extend([(a.item(), b.item()) for a, b in sample_pairs])

    print(decisions)

    graph = ComputationGraph.from_valid_decision_sequence(
        decisions=decisions, num_inputs=address_bitdepth, num_outputs=3
    )

    output = graph.evaluate(address_bitarrays)
    output = output_to_image_array(output, original_shape)

    loss = np.sqrt(
        np.mean(np.square(image.astype(np.float32) - output.astype(np.float32)))
    )

    if loss < best_loss:
        best_loss = loss
        print(f"RMSE: {best_loss:.5f} | Step: {step:04} | Saved: {last_updated_at:.5f}")

        if best_loss < last_updated_at * 0.995:

            output_pil = Image.fromarray(np.moveaxis(output, 0, -1))
            output_pil.save(
                "dagnabbit/outputs/output.jpg",
                format="JPEG",
                subsampling=0,
                quality=100,
            )
            output_pil.save(
                f"dagnabbit/outputs/timelapse/{update_counter:06}.jpg",
                format="JPEG",
                subsampling=0,
                quality=100,
            )

            last_updated_at = best_loss
            update_counter += 1
