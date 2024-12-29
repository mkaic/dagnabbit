from argparse import ArgumentParser
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from ..src.bitarrays import (
    calculate_address_bitdepth,
    get_address_bitarrays,
    output_to_image_array,
)
from ..src.dag import ComputationGraph, random_dag

parser = ArgumentParser()
parser.add_argument("-g", "--num_gates", type=int, default=512)
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

initialization = random_dag(
    num_gates=args.num_gates, num_inputs=address_bitdepth, num_outputs=3
)
graph = ComputationGraph.from_valid_edges(
    edges=initialization, num_inputs=address_bitdepth, num_outputs=3
)

address_bitarrays = get_address_bitarrays(original_shape)

best_loss = np.inf
last_updated_at = best_loss
update_counter = 0

shutil.rmtree("dagnabbit/outputs/timelapse", ignore_errors=True)
Path("dagnabbit/outputs/timelapse").mkdir(parents=True, exist_ok=True)

for epoch in range(1_000):
    permutation = np.random.permutation(args.num_gates)
    for i, gate_idx in enumerate(permutation):
        mutation_type = np.random.choice(["function", "input"])
        mutant_id = sorted(graph.evaluation_order)[gate_idx]
        match mutation_type:
            case "function":
                old_function, new_function = graph.stage_node_function_mutation(
                    mutant_id
                )
            case "input":
                old_input_id, new_input_id = graph.stage_node_input_mutation(mutant_id)

        output = graph.evaluate(address_bitarrays)
        output = output_to_image_array(output, original_shape)

        loss = np.sqrt(
            np.mean(np.square(image.astype(np.float32) - output.astype(np.float32)))
        )

        if loss < best_loss:
            best_loss = loss
            print(
                f"RMSE: {best_loss:.5f} | E: {epoch:04} | S: {i:04} | Saved: {last_updated_at:.5f} | {mutation_type}"
            )

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

        elif loss == best_loss:
            pass
        else:
            match mutation_type:
                case "function":
                    graph.undo_node_function_mutation(
                        node_id=mutant_id,
                        old_function=old_function,
                        new_function=new_function,
                    )
                case "input":
                    graph.undo_node_input_mutation(
                        node_id=mutant_id,
                        old_input_id=old_input_id,
                        new_input_id=new_input_id,
                    )
