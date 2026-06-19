import os

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import torch

from dagnabbit.dag.autoencoder import DagnabbitAutoEncoder
from dagnabbit.dag.description import FixedInDegreeDAGDescription


def _build_mixed_rank_graph() -> FixedInDegreeDAGDescription:
    # Type layout for this graph:
    #   trunk: 0, 1
    #   roots: 2, 3
    #   output: 4
    return FixedInDegreeDAGDescription(
        num_root_nodes=2,
        num_trunk_nodes=2,
        num_output_nodes=1,
        num_trunk_node_types=2,
        trunk_node_in_degrees=[1, 2],
        node_inputs_indices=[
            [],
            [],
            [0],
            [0, 1],
            [2],
        ],
        node_types=[
            2,
            3,
            0,
            1,
            4,
        ],
    )


def _build_model() -> DagnabbitAutoEncoder:
    return DagnabbitAutoEncoder(
        node_embedding_dim=16,
        trunk_node_type_in_degrees=[1, 2],
        num_trunk_node_types=2,
        num_root_nodes=2,
        num_output_nodes=1,
        mlp_depth=1,
        mlp_expansion_factor=2.0,
        transformer_num_layers=2,
        transformer_num_register_tokens=2,
        transformer_num_heads=4,
    )


def test_mixed_type_rank_uses_one_shared_transformer_call() -> None:
    torch.manual_seed(0)
    model = _build_model()
    graph = _build_mixed_rank_graph()

    encoder_calls: list[tuple[list[int], list[int]]] = []
    decoder_calls: list[tuple[list[int], list[int]]] = []

    original_encode = model.node_encoder.forward_batch
    original_decode = model.node_decoder.forward_batch

    def counted_encode(parent_embeddings, subtypes, valid_parent_mask):
        encoder_calls.append(
            (
                subtypes.detach().cpu().tolist(),
                valid_parent_mask.detach().sum(dim=1).cpu().tolist(),
            )
        )
        return original_encode(parent_embeddings, subtypes, valid_parent_mask)

    def counted_decode(node_embeddings, subtypes):
        decoder_calls.append(
            (
                subtypes.detach().cpu().tolist(),
                [model._in_degree_for_type(int(t)) for t in subtypes.detach().cpu()],
            )
        )
        return original_decode(node_embeddings, subtypes)

    model.node_encoder.forward_batch = counted_encode
    model.node_decoder.forward_batch = counted_decode

    losses, encode_buffer, decode_buffer = model.training_forward(
        graph,
        return_buffers=True,
    )

    assert encode_buffer.shape == (graph.num_nodes, model.node_embedding_dim)
    assert decode_buffer.shape == (graph.num_nodes, model.node_embedding_dim)
    assert torch.isfinite(encode_buffer).all()
    assert torch.isfinite(decode_buffer).all()
    assert torch.isfinite(losses.primary_node_classification_losses).all()

    # Rank 1 has two trunk nodes with different type ids and different
    # in-degrees. They should still enter a single shared encoder call.
    assert ([0, 1], [1, 2]) in encoder_calls

    # The decode pass should likewise process those rank-1 mixed-type nodes in a
    # single shared decoder call, producing max_indegree slots and scattering
    # only each node's valid parent slots.
    assert ([0, 1, 0, 1], [1, 2, 1, 2]) in decoder_calls
