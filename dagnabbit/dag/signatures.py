"""Comparing decoded graphs as tensors.

``decode_latent`` plus a pair of argmaxes turns any latent into a concrete DAG.
Asking whether two latents decoded to the *same* DAG then comes up in two
unrelated places -- counting how many distinct circuits a sampler actually
produced, and measuring how far a perturbation has to travel before the decode
moves -- so the comparison lives here rather than in either caller.

The exact alternative is
:func:`~dagnabbit.dag.description.graphs_match`, which is where this would
naturally belong except that it costs a Python pass per *pair*. Counting
distinct samples among a few hundred candidates is precisely the case where
that per-pair cost is the whole measurement, hence the tensor form. This module
cannot live in :mod:`dagnabbit.dag.description` for a duller reason too: it
needs the model's node geometry, and the autoencoder already imports
descriptions.

Nothing here knows what a circuit is, and nothing in this module may import
from :mod:`dagnabbit.tasks`.
"""

import torch
from torch import Tensor

from dagnabbit.dag.autoencoder import DagnabbitAutoEncoder


def choice_signatures(
    model: DagnabbitAutoEncoder,
    trunk_types: Tensor,
    parent_choices: Tensor,
) -> Tensor:
    """Decoded graphs as comparable integer rows: ``[B, T + N * S]``.

    Two graphs are the same graph iff their signatures match. Slots beyond a
    position's in-degree are set to -1 before comparison, because a slot that
    does not exist in the decoded circuit cannot be a difference between two
    circuits -- comparing them raw would report structural diversity that is
    pure padding noise.

    Kept in tensor form on purpose. The alternatives are
    :func:`~dagnabbit.dag.description.graphs_match`, which is exact but costs a
    Python pass per *pair*, and nothing -- and counting distinct samples among a
    few hundred candidates is precisely a place where a per-pair Python loop is
    the whole cost of the measurement.
    """
    in_degrees = torch.tensor(model.trunk_node_in_degrees, device=trunk_types.device)
    active = torch.zeros(
        trunk_types.shape[0],
        model.num_nodes,
        dtype=torch.long,
        device=trunk_types.device,
    )
    active[:, model.num_root_nodes : model.output_start] = in_degrees[trunk_types]
    active[:, model.output_start :] = 1

    slot_index = torch.arange(model.maximum_indegree, device=trunk_types.device)
    slot_mask = slot_index.view(1, 1, -1) < active.unsqueeze(-1)
    masked_parents = parent_choices.masked_fill(~slot_mask, -1)
    return torch.cat([trunk_types, masked_parents.flatten(start_dim=1)], dim=1)


def count_distinct_signatures(signatures: Tensor, group_size: int) -> Tensor:
    """Distinct graphs per group of ``group_size`` consecutive rows -> ``[P]``.

    ``signatures`` is ``[P * group_size, L]`` laid out group-major, matching what
    ``repeat_interleave`` on the conditioning produces. Reported as a fraction of
    ``group_size`` by callers: near 1 means the sampler is exploring, near
    1/group_size means every draw collapsed to the same circuit and best-of-N is
    buying nothing.
    """
    if signatures.shape[0] % group_size != 0:
        raise ValueError(
            f"{signatures.shape[0]} signatures do not divide into groups of "
            f"{group_size}"
        )
    grouped = signatures.reshape(-1, group_size, signatures.shape[-1]).cpu()
    return torch.tensor(
        [len({tuple(row.tolist()) for row in group}) for group in grouped],
        dtype=torch.long,
    )
