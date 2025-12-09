import torch
import torch.nn as nn


# This code based on https://github.com/sisl/zero_order_rnn/blob/49cddcf9f68760cca792305e42266aa44a670ea8/distributed_rge.py#L255C27-L255C30
def apply_perturbation(
    module: nn.Module, seed: int, step_size: float, device: torch.device
) -> None:
    # Set the seed for random number generator, this lets us "store" a whole-model perturbation
    # without ever materializing the whole thing — we can just regenerate the same perturbation procedurally.
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    # The perturbation itself is from the Rademacher distribution — basically, each parameter is nudged by
    # step_size * 1 or -1 with equal probability. You're basically picking a random nudge vector on the n-dimensional hypercube, which is also a subset of the points on the n-dimensional root-n-radius hypersphere.
    for param in module.parameters():
        perturbation = (
            torch.zeros_like(param.data)
            .bernoulli_(0.5, generator=rng)
            .mul_(2)
            .sub_(1)
            .mul_(step_size)
        )
        param.data = param.data + perturbation
