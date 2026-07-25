"""Tests for latent-space CEM search.

The search's real-world result on the current checkpoint is poor, but that is a
property of the representation rather than of this code (the perfect adder
round-trips through the latent to chance fitness). These tests therefore pin
down the algorithm itself: that the CEM update provably climbs a synthetic
objective, that sampling respects the measured shell, and that the loop's
bookkeeping is sound -- so that when the representation improves, a failure
here is distinguishable from a failure there.
"""

import pytest
import torch

from dagnabbit.dag.autoencoder import DagnabbitAutoEncoder
from dagnabbit.search.latent import (
    LatentGeometry,
    cem_update,
    evolve_latent,
    measure_latent_geometry,
)
from dagnabbit.tasks.logic_gates.evaluate import BitpackedTask, make_valid_bit_mask

NUM_ROOT_NODES = 4
NUM_TRUNK_NODES = 8
NUM_OUTPUT_NODES = 2
EMBEDDING_DIM = 64  # the sequence transformer fixes a 64-wide head


def tiny_model() -> DagnabbitAutoEncoder:
    torch.manual_seed(0)
    return DagnabbitAutoEncoder(
        node_embedding_dim=EMBEDDING_DIM,
        trunk_node_type_in_degrees=2,
        num_trunk_node_types=2,
        num_root_nodes=NUM_ROOT_NODES,
        num_trunk_nodes=NUM_TRUNK_NODES,
        num_output_nodes=NUM_OUTPUT_NODES,
        mlp_expansion_factor=1.0,
        encoder_num_layers=1,
        compressor_num_layers=1,
        decoder_num_layers=1,
    ).eval()


def tiny_task() -> BitpackedTask:
    generator = torch.Generator().manual_seed(0)
    num_rows = 16
    words = (num_rows + 7) // 8
    roots = torch.randint(
        0, 256, (NUM_ROOT_NODES, words), generator=generator, dtype=torch.uint8
    )
    targets = torch.randint(
        0, 256, (NUM_OUTPUT_NODES, words), generator=generator, dtype=torch.uint8
    )
    return BitpackedTask(
        root_values=roots,
        target_values=targets,
        num_rows=num_rows,
        valid_bit_mask=make_valid_bit_mask(num_rows, words),
    )


def synthetic_geometry(num_tokens: int = 4, dim: int = 8) -> LatentGeometry:
    return LatentGeometry(
        mean=torch.zeros(num_tokens, dim),
        std=torch.ones(num_tokens, dim),
        token_norm=float(dim) ** 0.5,
    )


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------


def test_projection_puts_every_token_on_the_shell():
    geometry = synthetic_geometry()
    latents = torch.randn(16, 4, 8) * 5.0
    projected = geometry.project(latents)
    norms = projected.norm(dim=-1)
    assert torch.allclose(norms, torch.full_like(norms, geometry.token_norm), atol=1e-5)


def test_projection_preserves_direction():
    geometry = synthetic_geometry()
    latents = torch.randn(4, 4, 8)
    projected = geometry.project(latents)
    cosine = torch.nn.functional.cosine_similarity(latents, projected, dim=-1)
    assert torch.allclose(cosine, torch.ones_like(cosine), atol=1e-5)


def test_measured_geometry_matches_the_encoder():
    model = tiny_model()
    geometry = measure_latent_geometry(model, num_graphs=16, batch_size=8, seed=0)
    assert geometry.mean.shape == (NUM_OUTPUT_NODES, EMBEDDING_DIM)
    assert geometry.std.shape == (NUM_OUTPUT_NODES, EMBEDDING_DIM)
    # The compressor ends in a LayerNorm, so tokens sit near sqrt(D).
    assert geometry.token_norm == pytest.approx(EMBEDDING_DIM**0.5, rel=0.25)


# --------------------------------------------------------------------------
# The CEM update, against a synthetic objective
# --------------------------------------------------------------------------


def test_cem_update_climbs_a_synthetic_objective():
    """With fitness = -distance to a target, CEM must converge onto it.

    This is the control that separates "the algorithm is broken" from "the
    latent space is unhelpful": no decoder is involved.
    """
    torch.manual_seed(0)
    geometry = synthetic_geometry(num_tokens=2, dim=8)
    target = geometry.project(torch.randn(1, 2, 8)).squeeze(0)

    mean = geometry.project(torch.randn(1, 2, 8)).squeeze(0)
    sigma = torch.ones_like(mean)
    sigma_floor = sigma * 1e-3

    start_distance = (mean - target).norm().item()
    for _ in range(60):
        population = geometry.project(mean + sigma * torch.randn(128, 2, 8))
        fitness = -(population - target).flatten(1).norm(dim=1)
        mean, sigma = cem_update(
            population=population,
            fitness=fitness,
            num_elites=16,
            sigma=sigma,
            geometry=geometry,
            sigma_smoothing=0.7,
            sigma_floor=sigma_floor,
        )

    end_distance = (mean - target).norm().item()
    assert end_distance < start_distance * 0.1, (
        f"CEM did not converge: {start_distance:.3f} -> {end_distance:.3f}"
    )


def test_cem_update_keeps_the_mean_on_the_shell():
    geometry = synthetic_geometry(num_tokens=2, dim=8)
    torch.manual_seed(0)
    population = geometry.project(torch.randn(32, 2, 8))
    fitness = torch.randn(32)
    sigma = torch.ones(2, 8)
    mean, _ = cem_update(
        population=population,
        fitness=fitness,
        num_elites=8,
        sigma=sigma,
        geometry=geometry,
        sigma_smoothing=0.5,
        sigma_floor=sigma * 0.01,
    )
    norms = mean.norm(dim=-1)
    assert torch.allclose(norms, torch.full_like(norms, geometry.token_norm), atol=1e-5)


def test_cem_update_respects_the_sigma_floor():
    geometry = synthetic_geometry(num_tokens=2, dim=8)
    # Identical elites would drive the elite std to zero.
    population = geometry.project(torch.ones(8, 2, 8))
    sigma = torch.ones(2, 8)
    floor = sigma * 0.25
    _, updated = cem_update(
        population=population,
        fitness=torch.zeros(8),
        num_elites=4,
        sigma=sigma,
        geometry=geometry,
        sigma_smoothing=1.0,
        sigma_floor=floor,
    )
    assert bool((updated >= floor - 1e-6).all())


# --------------------------------------------------------------------------
# The search loop
# --------------------------------------------------------------------------


def test_evolve_latent_runs_and_never_regresses():
    model = tiny_model()
    history = []
    result = evolve_latent(
        model,
        tiny_task(),
        num_generations=5,
        population_size=8,
        elite_fraction=0.5,
        seed=0,
        on_generation=history.append,
    )
    best = [record.best_fitness for record in history]
    assert best == sorted(best)
    assert 0.0 <= result.best_fitness <= 1.0
    assert result.evaluations == 5 * 8
    assert result.best_graph.num_nodes == model.num_nodes


def test_evolve_latent_is_reproducible():
    model = tiny_model()
    task = tiny_task()
    geometry = measure_latent_geometry(model, num_graphs=16, batch_size=8, seed=0)

    def run():
        return evolve_latent(
            model,
            task,
            num_generations=4,
            population_size=8,
            seed=7,
            geometry=geometry,
        ).best_fitness

    assert run() == run()


def test_evolve_latent_accepts_a_warm_start():
    model = tiny_model()
    geometry = measure_latent_geometry(model, num_graphs=16, batch_size=8, seed=0)
    result = evolve_latent(
        model,
        tiny_task(),
        num_generations=2,
        population_size=8,
        seed=0,
        geometry=geometry,
        initial_latent=geometry.mean.clone(),
    )
    assert result.best_graph is not None


def test_evolve_latent_rejects_bad_configuration():
    model = tiny_model()
    task = tiny_task()
    with pytest.raises(ValueError):
        evolve_latent(model, task, num_generations=1, population_size=1)
    with pytest.raises(ValueError):
        evolve_latent(model, task, num_generations=1, elite_fraction=0.0)
    with pytest.raises(ValueError, match="initial_latent"):
        evolve_latent(
            model,
            task,
            num_generations=1,
            population_size=4,
            initial_latent=torch.zeros(3, 5),
        )
