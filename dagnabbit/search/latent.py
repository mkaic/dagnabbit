"""Cross-entropy method search over the autoencoder's graph latent.

The decoder maps any point in latent space to a structurally valid DAG, so
search never needs repair operators or constraint handling -- every sample is a
legal circuit. What it does need is to stay on the manifold the decoder was
trained on, and the measured geometry says exactly where that is:

* Each of the K latent tokens comes out of a final LayerNorm, so its norm is
  pinned to a very thin shell (measured at 22.90 +/- 0.01 for a 512-dim
  checkpoint, against sqrt(512) = 22.63). Sampling is therefore projected back
  onto that shell per token, which is a free and exactly-right trust region.
* Cross-token correlation is low (~0.07), so a per-token diagonal covariance
  loses little and keeps the update O(K*D) rather than O((K*D)^2). A full
  covariance over 4096 dims is not affordable and, given that measurement, not
  worth affording.

The latent's effective dimensionality is high (participation ratio ~900 of
4096, needing over a thousand principal components to decode at all), so this
is *not* a low-dimensional search dressed up in a big ambient space. Treat CEM
here as the honest baseline for "does the latent help at all", to be read
against :mod:`dagnabbit.search.discrete` on the same budget.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from dagnabbit.dag.autoencoder import DagnabbitAutoEncoder
from dagnabbit.dag.description import make_random_graph_description
from dagnabbit.search.common import DEFAULT_EVAL_CHUNK, SearchLoop, SearchResult
from dagnabbit.tasks.logic_gates.evaluate import BitpackedTask


@dataclass
class LatentGeometry:
    """Where encoded graphs actually live, measured rather than assumed."""

    mean: Tensor  # [K, D]
    std: Tensor  # [K, D]
    token_norm: float  # mean L2 norm of a single latent token

    def project(self, latents: Tensor) -> Tensor:
        """Rescale every token of every latent onto the measured shell."""
        norms = latents.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return latents * (self.token_norm / norms)


@torch.no_grad()
def measure_latent_geometry(
    model: DagnabbitAutoEncoder,
    num_graphs: int = 512,
    batch_size: int = 64,
    seed: int | None = None,
) -> LatentGeometry:
    """Encode random graphs to find the latent distribution's shell and spread."""
    if seed is not None:
        # The graph generator seeds itself from torch, not the stdlib RNG.
        torch.manual_seed(seed)

    latents: list[Tensor] = []
    remaining = num_graphs
    while remaining > 0:
        count = min(batch_size, remaining)
        graphs = [
            make_random_graph_description(
                num_root_nodes=model.num_root_nodes,
                num_trunk_nodes=model.num_trunk_nodes,
                num_output_nodes=model.num_output_nodes,
                trunk_node_in_degrees=model.trunk_node_in_degrees,
                num_trunk_node_types=model.num_trunk_node_types,
            )
            for _ in range(count)
        ]
        latents.append(model.encode_to_latent(graphs).float())
        remaining -= count

    encoded = torch.cat(latents)
    return LatentGeometry(
        mean=encoded.mean(dim=0),
        std=encoded.std(dim=0),
        token_norm=float(encoded.norm(dim=-1).mean()),
    )


def cem_update(
    population: Tensor,
    fitness: Tensor,
    num_elites: int,
    sigma: Tensor,
    geometry: LatentGeometry,
    sigma_smoothing: float,
    sigma_floor: Tensor,
) -> tuple[Tensor, Tensor]:
    """One cross-entropy update: refit the sampling distribution to the elites.

    Split out from the search loop so the distribution mathematics can be
    tested against a synthetic objective, independently of any decoder.

    Note the high-dimensional caveat: the elite mean of ``num_elites`` samples
    in K*D dimensions is pulled hard toward the previous mean when
    ``num_elites`` is small relative to K*D. With a 4096-dim latent this is the
    dominant failure mode of CEM, not a subtlety.
    """
    elite_indices = torch.topk(fitness, num_elites).indices.to(population.device)
    elites = population[elite_indices]
    mean = geometry.project(elites.mean(dim=0, keepdim=True)).squeeze(0)
    updated_sigma = torch.maximum(
        (1.0 - sigma_smoothing) * sigma + sigma_smoothing * elites.std(dim=0),
        sigma_floor,
    )
    return mean, updated_sigma


@torch.no_grad()
def evolve_latent(
    model: DagnabbitAutoEncoder,
    task: BitpackedTask,
    *,
    num_generations: int,
    population_size: int = 128,
    elite_fraction: float = 0.125,
    sigma_scale: float = 1.0,
    sigma_smoothing: float = 0.5,
    sigma_floor_fraction: float = 0.05,
    seed: int = 0,
    geometry: LatentGeometry | None = None,
    initial_latent: Tensor | None = None,
    target_fitness: float = 1.0,
    chunk_size: int = DEFAULT_EVAL_CHUNK,
    on_generation=None,
) -> SearchResult:
    """Run CEM in latent space, decoding each sample into a circuit to score it.

    ``sigma_scale`` multiplies the measured per-dimension spread to set the
    initial search width. ``sigma_smoothing`` blends the elite spread into the
    running sigma (1.0 replaces it outright, which collapses fast);
    ``sigma_floor_fraction`` keeps sigma from falling below that fraction of
    its initial value, so the search cannot fully converge onto one point.
    """
    if not 0.0 < elite_fraction <= 1.0:
        raise ValueError(f"elite_fraction must be in (0, 1], got {elite_fraction}")
    if population_size < 2:
        raise ValueError("population_size must be at least 2")

    device = model.root_node_embeddings.weight.device
    generator = torch.Generator(device="cpu").manual_seed(seed)

    if geometry is None:
        geometry = measure_latent_geometry(model, seed=seed)

    mean = (
        geometry.mean.clone() if initial_latent is None else initial_latent.clone()
    ).to(device)
    if mean.shape != geometry.mean.shape:
        raise ValueError(
            f"initial_latent has shape {tuple(mean.shape)}, expected "
            f"{tuple(geometry.mean.shape)}"
        )
    mean = geometry.project(mean.unsqueeze(0)).squeeze(0)

    sigma = (geometry.std * sigma_scale).to(device)
    sigma_floor = sigma * sigma_floor_fraction
    num_elites = max(2, int(round(population_size * elite_fraction)))

    loop = SearchLoop(
        task=task,
        num_generations=num_generations,
        target_fitness=target_fitness,
        chunk_size=chunk_size,
        on_generation=on_generation,
    )

    for generation in range(num_generations):
        if loop.should_stop:
            break

        # Sample on CPU so a run is reproducible from ``seed`` regardless of
        # which device the model happens to be on.
        noise = torch.randn(
            (population_size, *mean.shape), generator=generator, dtype=torch.float32
        ).to(device)
        population = geometry.project(mean + sigma * noise)

        graphs = model.generate(population)
        fitness = loop.score(graphs)

        mean, sigma = cem_update(
            population=population,
            fitness=fitness,
            num_elites=num_elites,
            sigma=sigma,
            geometry=geometry,
            sigma_smoothing=sigma_smoothing,
            sigma_floor=sigma_floor,
        )

        loop.record(generation=generation, fitness=fitness)

    return loop.result()


@torch.no_grad()
def encode_graphs_to_latent(
    model: DagnabbitAutoEncoder,
    graphs: Sequence,
) -> Tensor:
    """Encode graphs to their latents, for warm-starting a search from them."""
    return model.encode_to_latent(list(graphs)).float()
