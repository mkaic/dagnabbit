import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from ..src import gates as GF
from ..src.bitarrays import (
    calculate_address_bitdepth,
    get_address_bitarrays,
)
from dagnabbit.src.model import (
    NeuralImplicitComputationGraph,
    get_sinusoidal_position_encodings,
)
from dagnabbit.src.cd_rge import apply_perturbation

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")

IMAGE_PATH = "test_images/branos.png"
RESIZE = 64
NUM_COMPUTE_NODES = 512
NUM_OUTPUT_NODES = 8
POSITION_ENCODING_DIM = 128
HIDDEN_DIM = 256

NUM_PERTURBATIONS = 128
PERTURBATION_SIZE = 0.01
STEP_SIZE = 0.01


def compute_rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    return np.sqrt(
        np.mean(np.square(prediction.astype(np.float32) - target.astype(np.float32)))
    )


image = Image.open(IMAGE_PATH).convert("RGB")
image = image.resize((RESIZE, RESIZE))
image = np.array(image)
image = np.moveaxis(image, -1, 0)
original_shape = image.shape

print("image.shape", original_shape)

address_bitdepth = calculate_address_bitdepth(original_shape)
address_bitarrays = get_address_bitarrays(original_shape)

pos_enc_dim = 64
position_encodings = (
    get_sinusoidal_position_encodings(
        length=NUM_COMPUTE_NODES * 2,
        dim=pos_enc_dim,
        base=10_000,
    )
    .float()
    .to(DEVICE)
)

with torch.no_grad():

    model = NeuralImplicitComputationGraph(
        num_input_nodes=address_bitdepth,
        num_compute_nodes=NUM_COMPUTE_NODES,
        num_output_nodes=NUM_OUTPUT_NODES,
        num_layers=4,
        input_dim=POSITION_ENCODING_DIM,
        hidden_dim=HIDDEN_DIM,
    )
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of trainable parameters: {num_params:,}")

    best_rmse = np.inf
    last_updated_at = best_rmse
    update_counter = 0

    shutil.rmtree("dagnabbit/outputs/timelapse", ignore_errors=True)
    Path("dagnabbit/outputs/timelapse").mkdir(parents=True, exist_ok=True)

    for step in tqdm[int](range(100_000)):
        perturbation_progress_bar = tqdm[int](
            range(NUM_PERTURBATIONS),
            leave=False,
            total=NUM_PERTURBATIONS,
        )

        seed_gradient_pairs = []

        for perturbation_idx in perturbation_progress_bar:

            perturbation_seed = torch.seed() + perturbation_idx

            # Apply positive step size perturbation
            model: NeuralImplicitComputationGraph = apply_perturbation(
                module=model,
                seed=perturbation_seed,
                step_size=(1 * PERTURBATION_SIZE),
                device=DEVICE,
            )
            output = model.forward(position_encodings=position_encodings)
            positive_step_loss = compute_rmse(output, image)

            # Undo positive step size perturbation and apply negative step size perturbation
            model: NeuralImplicitComputationGraph = apply_perturbation(
                module=model,
                seed=perturbation_seed,
                step_size=(-2 * PERTURBATION_SIZE),
                device=DEVICE,
            )
            output = model.forward(position_encodings=position_encodings)
            negative_step_loss = compute_rmse(output, image)

            gradient = (positive_step_loss - negative_step_loss) / (
                2 * PERTURBATION_SIZE
            )

            # Record this noisy estimate of the gradient along the perturbation
            seed_gradient_pairs.append((perturbation_seed, gradient))

            # Undo negative step size perturbation and restore model to original state.
            model: NeuralImplicitComputationGraph = apply_perturbation(
                module=model,
                seed=perturbation_seed,
                step_size=(1 * PERTURBATION_SIZE),
                device=DEVICE,
            )

        # Now we reapply each perturbation, but this timne we weight them by their estimated gradients.
        # This whole process is a zero-order optimization method called Central Difference Random Gradient Estimation.
        # I am basing my implementation on this paper by Francois Chaubard: https://arxiv.org/abs/2505.17852
        for perturbation_seed, gradient in tqdm[tuple[int, float]](
            seed_gradient_pairs, leave=False, desc="Applying weighted perturbations"
        ):
            model: NeuralImplicitComputationGraph = apply_perturbation(
                module=model,
                seed=perturbation_seed,
                step_size=-1 * (gradient / (2 * NUM_PERTURBATIONS)),
                device=DEVICE,
            )

        model: NeuralImplicitComputationGraph
        output = model.forward(position_encodings=position_encodings)
        rmse = compute_rmse(output, image)

        if rmse < best_rmse:
            best_rmse = rmse
            print(
                f"RMSE: {best_rmse:.5f} | Step: {step:04} | Saved: {last_updated_at:.5f}"
            )

            if best_rmse < last_updated_at * 0.995:

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

                last_updated_at = best_rmse
                update_counter += 1
