"""Does the model survive a round trip of a circuit that actually computes?

A **diagnostic only** -- nothing here touches the loss or the optimizer. It
asks a question reconstruction loss cannot: after encoding a known-good circuit
and decoding it again, does the result still compute the same function?

The answer on a structurally-accurate checkpoint can be "no" by a wide margin.
Pointer accuracy averages over wires that are interchangeable in random
training graphs, while a circuit's behaviour depends on every wire; ~80% of
wires reproduced still scores at chance. That gap is what this measures.

Two reference circuits, both 16 inputs / 8 outputs / 128 gates:

* **adder** -- an 8-bit ripple-carry adder. Deep, with a carry chain that makes
  bit 7 depend on bit 0.
* **xor** -- bitwise ``a XOR b``. Same shape, but every output depends on just
  two inputs through four gates, with nothing travelling far.

Reading them together separates "cannot hold long-range structure" from
"cannot hold structured circuits at all".

Neither circuit is in the training distribution, and both should stay out of
it: once a probe circuit is trained on, it stops measuring generalization.
"""

from dataclasses import dataclass

import numpy as np
import torch

from dagnabbit.dag.description import FixedInDegreeDAGDescription, graphs_match
from dagnabbit.tasks.logic_gates.bitarrays import get_8bit_adder_truth_table
from dagnabbit.tasks.logic_gates.evaluate import (
    BitpackedTask,
    bit_accuracy,
    evaluate_graphs,
    make_valid_bit_mask,
)
from dagnabbit.tasks.logic_gates.reference_circuits import (
    AdderAnnotations,
    build_nand_bitwise_xor,
    build_nand_ripple_carry_adder,
)


@dataclass(frozen=True)
class ProbeCircuit:
    name: str
    graph: FixedInDegreeDAGDescription
    annotations: AdderAnnotations
    task: BitpackedTask


_CIRCUIT_CACHE: dict[str, list[ProbeCircuit]] = {}


def _xor_task() -> BitpackedTask:
    """Same inputs as the adder table, targets are ``a XOR b`` bitwise."""
    packed_inputs, _ = get_8bit_adder_truth_table()
    width = packed_inputs.shape[0] // 2
    targets = np.stack(
        [packed_inputs[bit] ^ packed_inputs[bit + width] for bit in range(width)]
    )
    num_words = packed_inputs.shape[1]
    return BitpackedTask(
        root_values=torch.from_numpy(packed_inputs.copy()),
        target_values=torch.from_numpy(targets),
        num_rows=256 * 256,
        valid_bit_mask=make_valid_bit_mask(256 * 256, num_words),
    )


def reference_circuits(device: torch.device | str = "cpu") -> list[ProbeCircuit]:
    """The probe circuits, built once per device and cached."""
    key = str(device)
    if key not in _CIRCUIT_CACHE:
        from dagnabbit.tasks.logic_gates.evaluate import adder_task

        adder_graph, adder_annotations = build_nand_ripple_carry_adder()
        xor_graph, xor_annotations = build_nand_bitwise_xor()
        _CIRCUIT_CACHE[key] = [
            ProbeCircuit("adder", adder_graph, adder_annotations, adder_task(device)),
            ProbeCircuit("xor", xor_graph, xor_annotations, _xor_task().to(device)),
        ]
    return _CIRCUIT_CACHE[key]


def _wire_accuracy(
    predicted: torch.Tensor,
    graph: FixedInDegreeDAGDescription,
    nodes,
) -> float:
    """Fraction of ``nodes``' input wires the decoder pointed at correctly."""
    truth = graph.canonical_parent_positions.to(predicted.device)
    mask = graph.canonical_parent_slot_mask.to(predicted.device)
    rows = sorted(graph.canonical_positions[node] for node in nodes)
    if not rows:
        return float("nan")
    correct = ((predicted == truth) & mask)[rows]
    total = int(mask[rows].sum())
    return float(correct.sum()) / total if total else float("nan")


def _carry_survival(
    predicted: torch.Tensor,
    graph: FixedInDegreeDAGDescription,
    annotations: AdderAnnotations,
) -> float:
    """Fraction of carry gates whose every consumer still points at them."""
    carries = annotations.carry_node_of_bit
    if not carries:
        return float("nan")
    positions = graph.canonical_positions
    survived = 0
    for node in carries.values():
        consumers = [
            (consumer, slot)
            for consumer in range(graph.num_nodes)
            for slot, parent in enumerate(graph.node_inputs_indices[consumer])
            if parent == node
        ]
        survived += all(
            int(predicted[positions[consumer]][slot]) == positions[node]
            for consumer, slot in consumers
        )
    return survived / len(carries)


@torch.no_grad()
def roundtrip_metrics(model, device: torch.device | str = "cpu") -> dict[str, float]:
    """Encode -> decode each reference circuit and score what comes back.

    Returns flat ``{"roundtrip/<circuit>/<metric>": value}`` scalars, ready to
    hand to a TensorBoard writer.

    Runs the model in eval mode so dropout does not corrupt the measurement,
    restores the previous mode, and saves/restores the RNG state so calling
    this cannot perturb a training run's random stream.
    """
    circuits = reference_circuits(device)
    was_training = model.training
    rng_state = torch.random.get_rng_state()
    model.eval()
    metrics: dict[str, float] = {}

    try:
        for circuit in circuits:
            graph, annotations = circuit.graph, circuit.annotations
            latent = model.encode_to_latent([graph]).float()
            reconstructed = model.decode_latent(latent)
            predicted = model.parent_pointer_logits(reconstructed).argmax(dim=-1)[0]

            rebuilt = model.generate(latent)
            fitness, _ = bit_accuracy(
                evaluate_graphs(rebuilt, circuit.task), circuit.task
            )

            output_start = graph.num_root_nodes + graph.num_trunk_nodes
            prefix = f"roundtrip/{circuit.name}"
            metrics[f"{prefix}/decoded_fitness"] = float(fitness[0])
            metrics[f"{prefix}/core_wires"] = _wire_accuracy(
                predicted, graph, annotations.core_nodes
            )
            metrics[f"{prefix}/buffer_wires"] = _wire_accuracy(
                predicted, graph, annotations.buffer_nodes
            )
            metrics[f"{prefix}/output_wires"] = _wire_accuracy(
                predicted, graph, range(output_start, graph.num_nodes)
            )
            metrics[f"{prefix}/exact_match"] = float(graphs_match(graph, rebuilt[0]))
            carry = _carry_survival(predicted, graph, annotations)
            if carry == carry:  # skip NaN for circuits with no carry chain
                metrics[f"{prefix}/carry_survival"] = carry
    finally:
        if was_training:
            model.train()
        torch.random.set_rng_state(rng_state)

    return metrics
