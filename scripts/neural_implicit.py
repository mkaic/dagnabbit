import shutil
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.optim import Adam
from tqdm import tqdm

from ..src import gates as GF
from ..src.bitarrays import (
    calculate_address_bitdepth,
    get_address_bitarrays,
    output_to_image_array,
)
from ..src.dag import ComputationGraph
from ..src.model import MLP, get_sinusoidal_position_encodings, save_if_best_loss

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(f"Using device: {device}")

parser = ArgumentParser()
parser.add_argument("-g", "--num_compute_nodes", type=int, default=512)
parser.add_argument(
    "-i", "--image_path", type=str, default="dagnabbit/test_images/branos.png"
)
parser.add_argument("-r", "--resize", type=int, default=64)
parser.add_argument("-b", "--batch_size", type=int, default=32)
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
).to(device)
mlp = MLP(
    layer_sizes=[64, 512, 1024, 512, args.num_compute_nodes + address_bitdepth],
    activation=torch.nn.GELU,
).to(device)

num_params = sum(p.numel() for p in mlp.parameters())
print(f"Number of trainable parameters: {num_params:,}")

optimizer = Adam(mlp.parameters(), lr=1e-2 / args.batch_size)
loss_fn = torch.nn.CrossEntropyLoss()

best_loss = np.inf
last_updated_at = best_loss
update_counter = 0


shutil.rmtree("dagnabbit/outputs/timelapse", ignore_errors=True)
Path("dagnabbit/outputs/timelapse").mkdir(parents=True, exist_ok=True)

for step in tqdm(range(100_000)):

    discrete_sampled_actions = []
    losses = []

    with torch.no_grad():
        logits = mlp(pos_enc)

        # each pair of two connection decisions is allowed to reference previous gates as well as input gates
        valid_indices = torch.arange(0, args.num_compute_nodes * 2, device=device) // 2
        for i in range(len(logits)):
            logits[i, address_bitdepth + valid_indices[i] :] = -torch.inf

        probabilities = torch.softmax(logits, dim=-1)

    for attempt in range(args.batch_size):

        samples = torch.multinomial(probabilities, 1)
        samples = samples.flatten()

        # Convert samples into edges list
        edges = []
        functions = [GF.NP_NAND for _ in range(args.num_compute_nodes)]

        # Group samples into pairs to form edges
        sample_pairs = zip(samples[::2], samples[1::2])
        for a, b in sample_pairs:
            edges.append((a.item(), b.item()))

        description = {
            "num_inputs": address_bitdepth,
            "num_outputs": 8,
            "edges": edges,
            "functions": functions,
        }

        graph = ComputationGraph.from_description(description=description)

        output = graph.evaluate(address_bitarrays)
        output = output_to_image_array(output, original_shape)

        loss = np.sqrt(
            np.mean(np.square(image.astype(np.float32) - output.astype(np.float32)))
        )

        discrete_sampled_actions.append(samples)
        losses.append(loss)

        best_loss, last_updated_at, update_counter = save_if_best_loss(
            loss, best_loss, output, last_updated_at, update_counter, step
        )

    median_loss = np.median(losses)
    advantages = [l - median_loss for l in losses]

    logits_with_grad = (
        mlp(pos_enc).expand(args.batch_size, -1, -1).to(device)
    )  # batch_size, num_compute_nodes, num_possible_actions

    discrete_sampled_actions = (
        torch.stack(discrete_sampled_actions).unsqueeze(-1).to(device)
    )

    one_hots = torch.zeros_like(logits_with_grad)
    one_hots.scatter_(2, discrete_sampled_actions, 1)

    loss = loss_fn(logits_with_grad, one_hots)
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()
