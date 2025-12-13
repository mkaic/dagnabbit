import shutil
from pathlib import Path
import time

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from dagnabbit.bitarrays import (
    get_address_bitarrays,
)
from dagnabbit.dag import (
    LayeredNANDGraph,
)
from dagnabbit.scripts import config

EPSILON = 1e-6

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cuda")
print(f"Using device: {DEVICE}")


def compute_rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = prediction.astype(np.float32)
    target = target.astype(np.float32)
    mean_square_error = np.mean(np.square(prediction - target))
    return np.sqrt(mean_square_error)


def compute_reward(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Computes unreduced reward for each batch element."""

    prediction = prediction.astype(np.float32)  # [batch_size, *image_shape]
    target = target.astype(np.float32)  # [1, *image_shape]

    axes_to_reduce = tuple(range(1, len(prediction.shape)))

    # [batch_size]
    absolute_error = np.mean(np.abs(prediction - target), axis=axes_to_reduce) / 255.0

    accuracy = 1 - absolute_error

    # [batch_size]
    accuracy_reward = np.power(accuracy, 6) * 2

    target_std = np.std(target) / 127.5
    prediction_std = np.std(prediction) / 127.5
    

    if target_std > prediction_std:
        std_ratio = target_std / (prediction_std + EPSILON)
    else:
        std_ratio = prediction_std / (target_std + EPSILON)

    log_std_ratio = np.log(std_ratio)
    std_reward = 1 / (1 + log_std_ratio * 5)
    # print(f"std_reward: {std_reward}")
    return accuracy_reward + std_reward


target_image = Image.open(config.IMAGE_PATH).convert("RGB")
target_image = target_image.resize((config.RESIZE, config.RESIZE))
target_image = np.array(target_image)
target_image = np.moveaxis(target_image, -1, 0)
original_shape = target_image.shape

print("image.shape", original_shape)

address_bitarrays = torch.from_numpy(get_address_bitarrays(original_shape))
address_bitarrays = address_bitarrays.to(DEVICE)

address_bitdepth = address_bitarrays.shape[0]
print("address_bitdepth", address_bitdepth)

model = LayeredNANDGraph(
    num_inputs=address_bitdepth,
    num_outputs=8,
    num_layers=config.NUM_LAYERS,
    num_neurons_per_layer=config.LAYER_WIDTH,
).to(DEVICE)
num_params = sum(p.numel() for p in model.parameters())
print(f"Number of trainable parameters: {num_params:,}")

optimizer = config.OPTIMIZER(model.parameters(), lr=config.LEARNING_RATE)

best_rmse = np.inf
last_updated_at = best_rmse
update_counter = 0

last_latest_save_time = time.time()

shutil.rmtree("dagnabbit/outputs/timelapse", ignore_errors=True)
Path("dagnabbit/outputs/timelapse").mkdir(parents=True, exist_ok=True)

shutil.rmtree("dagnabbit/outputs/all_outputs", ignore_errors=True)
Path("dagnabbit/outputs/all_outputs").mkdir(parents=True, exist_ok=True)

progress_bar = tqdm(range(100_000))
for step in progress_bar:
    optimizer.zero_grad()
    for _ in range(config.GRADIENT_ACCUMULATION_STEPS):
        with torch.no_grad():
            # [batch_size, *image_shape]
            outputs, connection_indices, invert_mask = model.forward(
                input_bitarrays=address_bitarrays,
                output_shape=original_shape,
                stochastic=True,
                batch_size=config.BATCH_SIZE,
            )

            rewards = compute_reward(outputs, np.expand_dims(target_image, 0))

        model.calculate_gradients(
            connection_indices=connection_indices,
            invert_mask=invert_mask,
            rewards=rewards,
            batch_size=config.BATCH_SIZE,
        )

    optimizer.step()

    output, _, _ = model.forward(
        input_bitarrays=address_bitarrays,
        output_shape=original_shape,
        stochastic=False,
        batch_size=1
    )
    output = output.squeeze(0)

    deterministic_rmse = compute_rmse(output, target_image)

    output_pil = Image.fromarray(np.moveaxis(output, 0, -1))
    if time.time() - last_latest_save_time > 5:
        output_pil.save(
            f"dagnabbit/outputs/latest.jpg",
            format="JPEG",
            subsampling=0,
            quality=100,
        )
        output_pil.save(
            f"dagnabbit/outputs/all_outputs/{step:06}.jpg",
            format="JPEG",
            subsampling=0,
            quality=100,
        )
        last_latest_save_time = time.time()

    if deterministic_rmse < best_rmse:
        best_rmse = deterministic_rmse

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
        f"Step: {step:06} | RMSE: {deterministic_rmse:.3f} | Best: {best_rmse:.3f} | RRMSE: {np.sqrt(deterministic_rmse):.3f}"
    )
