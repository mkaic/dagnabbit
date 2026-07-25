"""Shape, masking, and gradient-flow checks for the sequence compressor/decoder.

Uses a deliberately small model so everything runs quickly on CPU.

Run directly::

    python -m dagnabbit.dag.tests.test_sequence_model
"""

import torch

from dagnabbit.dag.autoencoder import DagnabbitAutoEncoder
from dagnabbit.dag.description import make_random_graph_description

NUM_ROOTS = 8
NUM_TRUNKS = 24
NUM_OUTPUTS = 4
NUM_TRUNK_TYPES = 2
IN_DEGREES = 2
EMBEDDING_DIM = 64
NUM_NODES = NUM_ROOTS + NUM_TRUNKS + NUM_OUTPUTS
OUTPUT_START = NUM_ROOTS + NUM_TRUNKS


def build_small_model() -> DagnabbitAutoEncoder:
    return DagnabbitAutoEncoder(
        node_embedding_dim=EMBEDDING_DIM,
        trunk_node_type_in_degrees=IN_DEGREES,
        num_trunk_node_types=NUM_TRUNK_TYPES,
        num_root_nodes=NUM_ROOTS,
        num_trunk_nodes=NUM_TRUNKS,
        num_output_nodes=NUM_OUTPUTS,
        mlp_expansion_factor=2.0,
        transformer_num_layers=1,
        transformer_mlp_depth=1,
        transformer_num_register_tokens=1,
        transformer_num_heads=4,
        compressor_num_layers=2,
        decoder_num_layers=2,
    )


def sample_graphs(count: int):
    return [
        make_random_graph_description(
            num_root_nodes=NUM_ROOTS,
            num_trunk_nodes=NUM_TRUNKS,
            num_output_nodes=NUM_OUTPUTS,
            trunk_node_in_degrees=IN_DEGREES,
            num_trunk_node_types=NUM_TRUNK_TYPES,
        )
        for _ in range(count)
    ]


def test_training_forward_shapes_and_masks() -> None:
    torch.manual_seed(0)
    model = build_small_model()
    losses = model.training_forward_batch(sample_graphs(3))

    assert losses.node_classification_losses.shape == (3, NUM_NODES)
    assert losses.node_predicted_type_logits.shape == (
        3,
        NUM_NODES,
        model.num_node_types,
    )
    assert losses.parent_pointer_losses.shape == (3, NUM_NODES, IN_DEGREES)
    assert losses.parent_pointer_logits.shape == (3, NUM_NODES, IN_DEGREES, NUM_NODES)

    assert torch.isfinite(losses.node_classification_losses).all()
    assert torch.isfinite(losses.parent_pointer_losses).all()
    # Invalid slots contribute exactly zero pointer loss.
    assert (losses.parent_pointer_losses[~losses.parent_pointer_slot_mask] == 0).all()
    # Roots have no input slots; outputs have exactly slot 0.
    slot_mask = losses.parent_pointer_slot_mask
    assert not slot_mask[:, :NUM_ROOTS].any()
    assert slot_mask[:, OUTPUT_START:, 0].all()
    assert not slot_mask[:, OUTPUT_START:, 1:].any()
    # Valid slots have strictly positive cross-entropy.
    assert (losses.parent_pointer_losses[slot_mask] > 0).all()


def test_pointer_candidate_mask_layout() -> None:
    model = build_small_model()
    mask = model.pointer_candidate_mask
    positions = torch.arange(NUM_NODES)
    expected = (positions[None, :] < positions[:, None]) & (
        positions[None, :] < OUTPUT_START
    )
    assert (mask == expected).all()


def test_pointer_logits_masked_exactly_outside_candidates() -> None:
    torch.manual_seed(0)
    model = build_small_model()
    losses = model.training_forward_batch(sample_graphs(1))
    logits = losses.parent_pointer_logits
    neg_inf = torch.isneginf(logits)
    expected = ~model.pointer_candidate_mask.view(1, NUM_NODES, 1, NUM_NODES).expand_as(
        logits
    )
    assert (neg_inf == expected).all()


def test_end_to_end_gradient_flow() -> None:
    torch.manual_seed(0)
    model = build_small_model()
    losses = model.training_forward_batch(sample_graphs(2))
    slot_mask = losses.parent_pointer_slot_mask
    total = (
        losses.node_classification_losses.mean()
        + losses.parent_pointer_losses.sum() / slot_mask.sum()
    )
    total.backward()

    named = dict(model.named_parameters())
    for name in (
        "root_node_embeddings.weight",
        "mask_token",
        "pointer_key_proj.weight",
        "pointer_slot_query_projs.0.weight",
        "pointer_slot_query_projs.1.weight",
        "node_type_predictor.weight",
    ):
        grad = named[name].grad
        assert grad is not None and float(grad.abs().sum()) > 0, name

    # End-to-end: gradient reaches the recursive encoder through the
    # compressor bottleneck.
    encoder_grad_mass = sum(
        float(parameter.grad.abs().sum())
        for name, parameter in named.items()
        if name.startswith("node_encoder.") and parameter.grad is not None
    )
    assert encoder_grad_mass > 0

    # Compressor and decoder both train.
    for prefix in ("compressor.", "decoder."):
        stack_grad_mass = sum(
            float(parameter.grad.abs().sum())
            for name, parameter in named.items()
            if name.startswith(prefix) and parameter.grad is not None
        )
        assert stack_grad_mass > 0, prefix


def assert_valid_generated(description) -> None:
    """Structural validity beyond the constructor's own asserts."""
    assert description.num_nodes == NUM_NODES
    for node_idx, parents in enumerate(description.node_inputs_indices):
        for parent in parents:
            # Parents strictly precede children and are never outputs, so the
            # graph is acyclic and outputs stay leaves.
            assert parent < node_idx
            assert parent < OUTPUT_START


def test_generate_is_always_a_valid_dag() -> None:
    torch.manual_seed(0)
    model = build_small_model()
    for _ in range(25):
        latent = torch.randn(2, NUM_OUTPUTS, EMBEDDING_DIM)
        descriptions = model.generate(latent)
        assert len(descriptions) == 2
        for description in descriptions:
            assert_valid_generated(description)

    # Single-graph convenience shape: [K, D] in, one description out.
    description = model.generate(torch.randn(NUM_OUTPUTS, EMBEDDING_DIM))
    assert_valid_generated(description)


def test_encode_to_latent_roundtrips_through_generate() -> None:
    torch.manual_seed(0)
    model = build_small_model()
    graphs = sample_graphs(3)
    latent = model.encode_to_latent(graphs)
    assert latent.shape == (3, NUM_OUTPUTS, EMBEDDING_DIM)
    rebuilt = model.generate(latent)
    assert len(rebuilt) == 3
    for description in rebuilt:
        assert_valid_generated(description)


def main() -> None:
    test_training_forward_shapes_and_masks()
    test_pointer_candidate_mask_layout()
    test_pointer_logits_masked_exactly_outside_candidates()
    test_end_to_end_gradient_flow()
    test_generate_is_always_a_valid_dag()
    test_encode_to_latent_roundtrips_through_generate()
    print("ALL SEQUENCE-MODEL CHECKS PASSED")


if __name__ == "__main__":
    main()
