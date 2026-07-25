"""Tests for the round-trip diagnostic probe.

The probe runs inside the training loop, so the properties that matter most are
the ones about *not* disturbing it: no gradients, no RNG consumption, and the
model handed back in the mode it arrived in. The other load-bearing invariant
is that both reference circuits still score exactly 1.0 -- a probe circuit that
silently stops being correct turns the whole metric into noise.
"""

import pytest
import torch

from dagnabbit.dag.autoencoder import DagnabbitAutoEncoder
from dagnabbit.tasks.logic_gates.evaluate import bit_accuracy, evaluate_graphs
from dagnabbit.tasks.logic_gates.roundtrip_probe import (
    reference_circuits,
    roundtrip_metrics,
)

EXPECTED_KEYS = {
    "roundtrip/adder/decoded_fitness",
    "roundtrip/adder/core_wires",
    "roundtrip/adder/buffer_wires",
    "roundtrip/adder/output_wires",
    "roundtrip/adder/exact_match",
    "roundtrip/adder/carry_survival",
    "roundtrip/xor/decoded_fitness",
    "roundtrip/xor/core_wires",
    "roundtrip/xor/buffer_wires",
    "roundtrip/xor/output_wires",
    "roundtrip/xor/exact_match",
}


def probe_sized_model() -> DagnabbitAutoEncoder:
    """Full node geometry (the probe circuits need it) at a small width."""
    torch.manual_seed(0)
    return DagnabbitAutoEncoder(
        node_embedding_dim=64,
        trunk_node_type_in_degrees=2,
        num_trunk_node_types=2,
        num_root_nodes=16,
        num_trunk_nodes=128,
        num_output_nodes=8,
        mlp_expansion_factor=1.0,
        encoder_num_layers=1,
        compressor_num_layers=1,
        decoder_num_layers=1,
    )


def test_reference_circuits_are_exactly_correct():
    """Both probe circuits must compute their target function perfectly."""
    for circuit in reference_circuits():
        fitness, per_output = bit_accuracy(
            evaluate_graphs([circuit.graph], circuit.task), circuit.task
        )
        assert fitness.item() == 1.0, f"{circuit.name} is not a correct circuit"
        assert bool((per_output == 1.0).all())


def test_reference_circuits_have_no_dead_gates():
    """Dead gates never occur in training graphs; a probe must not have them."""
    for circuit in reference_circuits():
        graph = circuit.graph
        output_start = graph.num_root_nodes + graph.num_trunk_nodes
        referenced = set()
        for parents in graph.node_inputs_indices:
            referenced.update(parents)
        dead = [
            node
            for node in range(graph.num_root_nodes, output_start)
            if node not in referenced
        ]
        assert dead == [], f"{circuit.name} has {len(dead)} dead gates"


def test_xor_probe_is_shallower_than_the_adder():
    """The contrast only works if one circuit really lacks long-range structure."""
    depths = {c.name: max(c.graph.node_ranks) for c in reference_circuits()}
    assert depths["xor"] < depths["adder"]


def test_metrics_have_the_expected_keys_and_ranges():
    metrics = roundtrip_metrics(probe_sized_model())
    assert set(metrics) == EXPECTED_KEYS
    for name, value in metrics.items():
        assert 0.0 <= value <= 1.0, f"{name} out of range: {value}"


def test_probe_does_not_disturb_training_state():
    """Mode, RNG stream and gradients must all come back untouched."""
    model = probe_sized_model()
    model.train()
    rng_before = torch.random.get_rng_state()

    roundtrip_metrics(model)

    assert model.training, "probe left the model in eval mode"
    assert torch.equal(rng_before, torch.random.get_rng_state()), (
        "probe consumed random numbers the training run depends on"
    )
    assert all(parameter.grad is None for parameter in model.parameters())


def test_probe_restores_eval_mode_too():
    model = probe_sized_model()
    model.eval()
    roundtrip_metrics(model)
    assert not model.training


def test_probe_is_deterministic():
    model = probe_sized_model()
    first = roundtrip_metrics(model)
    second = roundtrip_metrics(model)
    assert first == second


def test_untrained_model_scores_near_chance():
    """A sanity floor: a random model cannot round-trip a real circuit."""
    metrics = roundtrip_metrics(probe_sized_model())
    assert metrics["roundtrip/adder/decoded_fitness"] == pytest.approx(0.5, abs=0.15)
    assert metrics["roundtrip/adder/exact_match"] == 0.0
