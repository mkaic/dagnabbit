from random import sample
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from collections import Counter

from dagnabbit.src.bitarrays import (
    calculate_address_bitdepth,
    get_address_bitarrays,
)
from dagnabbit.src.model import (
    NeuralImplicitComputationGraph,
    get_sinusoidal_position_encodings,
)
from dagnabbit.src.cd_rge import apply_perturbation
from dagnabbit.scripts import config
from dagnabbit.src.gate_functions import AVAILABLE_FUNCTIONS

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")


def compute_mse(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = prediction.astype(np.float32)
    target = target.astype(np.float32)
    mean_square_error = np.mean(np.square(prediction - target))
    return np.sqrt(mean_square_error)


image = Image.open(config.IMAGE_PATH).convert("RGB")
image = image.resize((config.RESIZE, config.RESIZE))
image = np.array(image)
image = np.moveaxis(image, -1, 0)
original_shape = image.shape

print("image.shape", original_shape)

address_bitdepth = calculate_address_bitdepth(original_shape)
address_bitarrays = get_address_bitarrays(original_shape)

position_encodings = (
    get_sinusoidal_position_encodings(
        length=config.NUM_COMPUTE_NODES * 2,
        dim=config.POSITION_ENCODING_DIM,
        base=10_000,
    )
    .float()
    .to(DEVICE)
)

with torch.no_grad():

    model = NeuralImplicitComputationGraph(
        num_input_nodes=address_bitdepth,
        num_compute_nodes=config.NUM_COMPUTE_NODES,
        num_output_nodes=config.NUM_OUTPUT_NODES,
        num_layers=config.NUM_LAYERS,
        position_encoding_dim=config.POSITION_ENCODING_DIM,
        hidden_dim=config.HIDDEN_DIM,
        recurrent_dim=config.RECURRENT_DIM,
        num_prior_gates_connectable=config.NUM_PRIOR_GATES_CONNECTABLE,
    )
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of trainable parameters: {num_params:,}")

    best_rmse = np.inf
    last_updated_at = best_rmse
    update_counter = 0

    shutil.rmtree("dagnabbit/outputs/timelapse", ignore_errors=True)
    Path("dagnabbit/outputs/timelapse").mkdir(parents=True, exist_ok=True)

    shutil.rmtree("dagnabbit/outputs/all_outputs", ignore_errors=True)
    Path("dagnabbit/outputs/all_outputs").mkdir(parents=True, exist_ok=True)

    progress_bar = tqdm(range(100_000))
    for step in progress_bar:
        perturbation_progress_bar = tqdm(
            range(config.NUM_PERTURBATIONS),
            leave=False,
            total=config.NUM_PERTURBATIONS,
        )

        seed_gradient_pairs = []

        for perturbation_idx in perturbation_progress_bar:

            perturbation_seed = torch.seed() + perturbation_idx

            # Apply positive step size perturbation
            model: NeuralImplicitComputationGraph = apply_perturbation(
                module=model,
                seed=perturbation_seed,
                step_size=(1 * config.PERTURBATION_SIZE),
                device=DEVICE,
            )
            output = model.forward(
                position_encodings=position_encodings,
                address_bitarrays=address_bitarrays,
                output_shape=original_shape,
            )
            positive_step_loss = np.sqrt(compute_mse(output, image))
            # print("\nPos", positive_step_loss)

            # Undo positive step size perturbation and apply negative step size perturbation
            model: NeuralImplicitComputationGraph = apply_perturbation(
                module=model,
                seed=perturbation_seed,
                step_size=(-2 * config.PERTURBATION_SIZE),
                device=DEVICE,
            )
            output = model.forward(
                position_encodings=position_encodings,
                address_bitarrays=address_bitarrays,
                output_shape=original_shape,
            )
            negative_step_loss = np.sqrt(compute_mse(output, image))
            # print("Neg", negative_step_loss)

            gradient = positive_step_loss - negative_step_loss
            gradient = gradient / (2 * config.PERTURBATION_SIZE)
            # print("Grad", gradient)

            # Record this noisy estimate of the gradient along the perturbation
            seed_gradient_pairs.append((perturbation_seed, gradient))

            # Undo negative step size perturbation and restore model to original state.
            model: NeuralImplicitComputationGraph = apply_perturbation(
                module=model,
                seed=perturbation_seed,
                step_size=(1 * config.PERTURBATION_SIZE),
                device=DEVICE,
            )

        # Now we reapply each perturbation, but this timne we weight them by their estimated gradients.
        # This whole process is a zero-order optimization method called Central Difference Random Gradient Estimation.
        # I am basing my implementation on this paper by Francois Chaubard: https://arxiv.org/abs/2505.17852
        model_param_copy = next(model.parameters()).data.clone()
        for perturbation_seed, gradient in tqdm(
            seed_gradient_pairs, leave=False, desc="Applying weighted perturbations"
        ):
            step_size = (
                -1 * gradient * config.STEP_SIZE / (2 * config.NUM_PERTURBATIONS)
            )
            # print("Step", step_size)
            model: NeuralImplicitComputationGraph = apply_perturbation(
                module=model,
                seed=perturbation_seed,
                step_size=step_size,
                device=DEVICE,
            )

        print(
            "mean absolute parameter update",
            torch.mean(
                torch.abs(next(model.parameters()).data - model_param_copy)
            ).item(),
        )

        model: NeuralImplicitComputationGraph
        sampled_rmses = []
        num_to_sample = 10
        for _ in range(num_to_sample):
            output = model.forward(
                position_encodings=position_encodings,
                address_bitarrays=address_bitarrays,
                output_shape=original_shape,
            )
            sampled_rmses.append(np.sqrt(compute_mse(output, image)))

        average_rmse = np.mean(sampled_rmses)
        last_rmse = sampled_rmses[-1]
        rmse_variance = np.std(sampled_rmses)

        output_pil = Image.fromarray(np.moveaxis(output, 0, -1))
        output_pil.save(
            f"dagnabbit/outputs/all_outputs/{step:06}.jpg",
            format="JPEG",
            subsampling=0,
            quality=100,
        )
        output_pil.save(
            f"dagnabbit/outputs/latest.jpg",
            format="JPEG",
            subsampling=0,
            quality=100,
        )

        if last_rmse < best_rmse:
            best_rmse = last_rmse

            output_pil.save(
                "dagnabbit/outputs/best.jpg",
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
            update_counter += 1

        progress_bar.set_description(
            f"Step: {step:06} | RMSE@{num_to_sample}: {average_rmse:.3f} ±{rmse_variance:.3f} | DiscRMSE: {last_rmse:.3f} | Best: {best_rmse:.3f}"
        )
