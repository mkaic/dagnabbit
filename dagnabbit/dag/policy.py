"""Stochastic policies over graphs, for policy-gradient training.

:meth:`DagnabbitAutoEncoder.generate` is deterministic, which is what you want
for inference and exactly what you cannot train on: every sample in a group
comes out identical and there is nothing to learn from. Something has to be
random. There are two places to put it, and they are not equally good.

Latent noise (:func:`sample_from_latent_noise`) perturbs the latent and decodes
by argmax. Decoder sampling (:func:`sample_from_decoder`) keeps one latent and
samples the decoder's per-slot distributions.

**Latent noise is the default, and the measurement is lopsided.** Scored
against a target its latent encodes exactly, a group of 64:

===========================  ======  ======
exploration                    mean     std
===========================  ======  ======
argmax (none)                1.0000  0.0000
decoder sampling, T=1.0      0.9999  0.0008
latent noise, sigma=0.03     0.9942  0.0072
latent noise, sigma=0.1      0.8211  0.0545
latent noise, sigma=0.3      0.5633  0.1095
===========================  ======  ======

Decoder sampling produces almost no spread once the latent is *meaningful*.
Its apparent diversity early in training is an artefact of an untrained
proposer emitting garbage latents, on which the decoder's distributions are
flat -- so that mechanism dies precisely as the model starts working. Latent
noise gives a monotone dial instead, all the way out to the spread of
independent random graphs, and is indifferent to decoder confidence.

The two do not combine. Under latent noise the sampled latent is detached, so
the decoder's log-probabilities have no gradient path back to the proposer;
adding them would contribute reward variance and no signal.
"""

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.distributions import Categorical, Normal

from dagnabbit.dag.autoencoder import DagnabbitAutoEncoder


@dataclass
class PolicySample:
    """One batch of graphs drawn from a policy, with what it costs to score them.

    Graphs stay in choice-tensor form -- the ``(trunk_types, parent_choices)``
    pair that :meth:`DagnabbitAutoEncoder.descriptions_from_choices` consumes
    -- rather than as built description objects. Reward evaluation reads the
    tensors directly (:func:`~dagnabbit.tasks.logic_gates.evaluate.
    evaluate_choices`), and building thousands of Python descriptions per
    training step was the single largest cost of a GRPO step. Callers that
    need real descriptions (rendering, structural comparison) can call
    ``model.descriptions_from_choices(sample.trunk_types,
    sample.parent_choices)`` themselves.
    """

    # [B, T] trunk class ids of each sampled graph.
    trunk_types: Tensor
    # [B, N, S] parent canonical positions (root rows ignored).
    parent_choices: Tensor
    # [B] summed log-probability of the choices that produced each graph.
    log_probs: Tensor
    # [B] summed entropy of those choice distributions.
    entropies: Tensor
    # [B] how many choices were made, for reporting entropy per action.
    num_actions: Tensor
    # [B, K] the same log-probability split per latent token, where the policy
    # factorizes that way -- summing this over K reproduces ``log_probs``. The
    # latent has one token per output node, so this is what lets a per-output
    # reward be paid to the token positionally responsible for it (see
    # ``GRPOConfig.factored_credit``). ``None`` when the policy's factorization
    # has no latent-token axis, as decoder sampling's does not.
    token_log_probs: Tensor | None = None


def project_to_shell(latent: Tensor) -> Tensor:
    """Rescale every latent token onto a shell of radius ``sqrt(D)``.

    Encoded latents come off a LayerNorm and live on that shell, so anything
    handed to the decoder should too.
    """
    scale = latent.shape[-1] ** 0.5
    norms = latent.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    return latent * (scale / norms)


def sample_from_latent_noise(
    model: DagnabbitAutoEncoder,
    mean_latent: Tensor,
    sigma: float,
) -> PolicySample:
    """Perturb ``[B, K, D]`` latents by Gaussian noise, then decode by argmax.

    The *action* is the pre-normalization latent: noise is drawn in ambient
    space and scored by a plain Gaussian density, while projecting back onto
    the shell and decoding are treated as part of the environment. That keeps
    the log-probability exact and simple, and the discarded radial component is
    one direction out of D.

    ``sigma`` reads directly as a relative step: noise of scale ``sigma`` on a
    token of norm ``sqrt(D)`` displaces it by roughly ``sigma`` of its own
    length. Useful range measured on this decoder is 0.03 (barely moves the
    decoded graph) to 0.3 (as diverse as unrelated random graphs).

    Sampling is deliberately not reparameterized: the reward is non
    differentiable, so the gradient has to come through the log-probability.
    """
    if mean_latent.ndim != 3:
        raise ValueError("mean_latent must have shape [B, K, D]")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive; got {sigma}")

    distribution = Normal(mean_latent, sigma)
    latents = distribution.sample()
    # Kept split per latent token; the [B] total is its sum over that axis.
    token_log_probs = distribution.log_prob(latents).sum(dim=2)
    log_probs = token_log_probs.sum(dim=1)
    entropies = distribution.entropy().sum(dim=(1, 2))

    trunk_types, parent_choices = model.generate_choices(project_to_shell(latents))
    num_actions = torch.full(
        (mean_latent.shape[0],),
        mean_latent.shape[1] * mean_latent.shape[2],
        device=mean_latent.device,
        dtype=log_probs.dtype,
    )
    return PolicySample(
        trunk_types=trunk_types,
        parent_choices=parent_choices,
        log_probs=log_probs,
        entropies=entropies,
        num_actions=num_actions,
        token_log_probs=token_log_probs,
    )


def sample_from_decoder(
    model: DagnabbitAutoEncoder,
    latent: Tensor,
    temperature: float = 1.0,
) -> PolicySample:
    """Sample the decoder's own choices, keeping one latent per graph.

    The action is the graph, factorized the way the decoder factorizes it: one
    categorical trunk type per trunk position, and one parent per *active*
    input slot. Which slots are active follows from the sampled types -- a
    type's in-degree is what says how many slots it has -- so the pointer
    choices are conditioned on the type choices, and inactive slots contribute
    nothing because they contribute nothing to the graph.

    Retained as the alternative to :func:`sample_from_latent_noise`, and as the
    thing to reach for if the latent-noise scale ever proves hard to tune. See
    the module docstring for why it is not the default.
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
    trunk_types = type_distribution.sample()
    type_log_probs = type_distribution.log_prob(trunk_types)
    type_entropies = type_distribution.entropy()

    # Roots have no input slots, so only trunk and output rows are sampled.
    # That also sidesteps position 0, whose candidate row is entirely -inf.
    pointer_logits = model.parent_pointer_logits(reconstructed)
    pointer_distribution = Categorical(
        logits=pointer_logits[:, model.num_root_nodes :].float() / temperature
    )
    parent_choices = pointer_distribution.sample()

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

    log_probs = type_log_probs.sum(dim=1) + (
        pointer_distribution.log_prob(parent_choices) * weights
    ).sum(dim=(1, 2))
    entropies = type_entropies.sum(dim=1) + (
        pointer_distribution.entropy() * weights
    ).sum(dim=(1, 2))

    # Choice consumers index parents by canonical position over all N nodes;
    # pad the unsampled root rows back in so the layouts line up.
    padded_choices = torch.zeros(
        batch_size,
        model.num_nodes,
        model.maximum_indegree,
        dtype=parent_choices.dtype,
        device=device,
    )
    padded_choices[:, model.num_root_nodes :] = parent_choices

    return PolicySample(
        trunk_types=trunk_types,
        parent_choices=padded_choices,
        log_probs=log_probs,
        entropies=entropies,
        num_actions=model.num_trunk_nodes + slot_mask.sum(dim=(1, 2)),
    )
