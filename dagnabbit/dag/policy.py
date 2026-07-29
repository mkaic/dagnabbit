"""The decoder read as a stochastic policy over graphs.

:meth:`DagnabbitAutoEncoder.generate` takes argmaxes, which is what you want
for inference and exactly what you cannot use for policy-gradient training: a
deterministic decode gives every sample in a group the same graph and the same
reward, and there is nothing left to learn from. This module samples those same
choices instead, and returns the log-probability of what it sampled.

The action space is the graph itself, factorized the way the decoder already
factorizes it:

* one categorical choice of trunk type at each of ``T`` trunk positions;
* one categorical choice of parent at each *active* input slot.

Which slots are active is decided by the *sampled* types, not by the argmax
ones -- a type's in-degree is what says how many slots it has -- so the second
group of choices is conditioned on the first. Inactive slots contribute nothing
to the log-probability, because they contribute nothing to the graph.

Gradients flow back through the sampled log-probabilities to the latent and
from there into whatever produced it. The autoencoder itself can stay frozen;
nothing here updates it.
"""

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.distributions import Categorical

from dagnabbit.dag.autoencoder import DagnabbitAutoEncoder
from dagnabbit.dag.description import FixedInDegreeDAGDescription


@dataclass
class SampledGraphs:
    """One batch of graphs sampled from the decoder, with their log-probs."""

    graphs: list[FixedInDegreeDAGDescription]
    # [B] summed log-probability of every choice that shaped each graph.
    log_probs: Tensor
    # [B] summed entropy of those same choice distributions.
    entropies: Tensor
    # Number of choices actually made per graph, for reporting entropy per
    # action rather than per graph.
    num_actions: Tensor


def sample_graphs_from_latent(
    model: DagnabbitAutoEncoder,
    latent: Tensor,
    temperature: float = 1.0,
) -> SampledGraphs:
    """Sample ``[B, K, D]`` latents into graphs, keeping the log-probs.

    ``temperature`` scales the logits before sampling: higher explores more.
    It deliberately affects the sampling distribution *and* the reported
    log-probabilities together, so the returned values are the log-probs of the
    policy actually being followed.

    Not decorated with ``no_grad``: the returned log-probabilities are the
    training signal.
    """
    if latent.ndim != 3:
        raise ValueError("latent must have shape [B, K, D]")
    if temperature <= 0:
        raise ValueError(f"temperature must be positive; got {temperature}")

    reconstructed = model.decode_latent(latent)
    batch_size = latent.shape[0]
    device = latent.device

    type_distribution = Categorical(
        logits=model.node_type_predictor(
            reconstructed[:, model.num_root_nodes : model.output_start]
        ).float()
        / temperature
    )
    trunk_types = type_distribution.sample()  # [B, T]
    type_log_probs = type_distribution.log_prob(trunk_types)  # [B, T]
    type_entropies = type_distribution.entropy()  # [B, T]

    # Roots have no input slots, so only the trunk and output rows are ever
    # sampled. That also sidesteps position 0, whose candidate row is entirely
    # -inf and would make a degenerate distribution.
    pointer_logits = model.parent_pointer_logits(reconstructed)
    sampled_rows = pointer_logits[:, model.num_root_nodes :].float() / temperature
    pointer_distribution = Categorical(logits=sampled_rows)
    parent_choices = pointer_distribution.sample()  # [B, N - R, S]

    # A trunk position's active slot count comes from the type just sampled;
    # every output position has exactly one.
    in_degrees = torch.tensor(model.trunk_node_in_degrees, device=device)
    active_slots = torch.cat(
        [
            in_degrees[trunk_types],
            torch.ones(
                batch_size,
                model.num_output_nodes,
                dtype=torch.long,
                device=device,
            ),
        ],
        dim=1,
    )
    slot_index = torch.arange(model.maximum_indegree, device=device)
    slot_mask = slot_index.view(1, 1, -1) < active_slots.unsqueeze(-1)
    weights = slot_mask.to(type_log_probs.dtype)

    pointer_log_probs = pointer_distribution.log_prob(parent_choices) * weights
    pointer_entropies = pointer_distribution.entropy() * weights

    log_probs = type_log_probs.sum(dim=1) + pointer_log_probs.sum(dim=(1, 2))
    entropies = type_entropies.sum(dim=1) + pointer_entropies.sum(dim=(1, 2))
    num_actions = model.num_trunk_nodes + slot_mask.sum(dim=(1, 2))

    # descriptions_from_choices indexes parents by canonical position over all
    # N nodes; pad the unsampled root rows back in so the layouts line up.
    padded_choices = torch.zeros(
        batch_size,
        model.num_nodes,
        model.maximum_indegree,
        dtype=parent_choices.dtype,
        device=device,
    )
    padded_choices[:, model.num_root_nodes :] = parent_choices

    return SampledGraphs(
        graphs=model.descriptions_from_choices(trunk_types, padded_choices),
        log_probs=log_probs,
        entropies=entropies,
        num_actions=num_actions,
    )
