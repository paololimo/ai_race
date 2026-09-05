"""Genetic algorithm: fitness-proportional selection, crossover, mutation."""

import logging
from typing import List, Sequence

import numpy as np

from src.config import GeneticConfig

logger = logging.getLogger(__name__)


def select_breeding_pool(
    genomes: Sequence[np.ndarray], fitnesses: Sequence[float], cfg: GeneticConfig
) -> List[np.ndarray]:
    """Return the top `breeding_fraction` of genomes, ranked by fitness."""
    pool_size = max(2, int(len(genomes) * cfg.breeding_fraction))
    ranked = np.argsort(fitnesses)[::-1][:pool_size]
    return [genomes[i] for i in ranked]


def crossover(parent_a: np.ndarray, parent_b: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Uniform crossover: each gene is drawn at random from one of the parents."""
    mask = rng.random(parent_a.size) < 0.5
    return np.where(mask, parent_a, parent_b)


def mutate(genome: np.ndarray, cfg: GeneticConfig, rng: np.random.Generator) -> np.ndarray:
    """Perturb a small fraction of genes with gaussian noise."""
    mutated = genome.copy()
    mask = rng.random(genome.size) < cfg.mutation_rate
    mutated[mask] += rng.normal(0.0, cfg.mutation_scale, size=int(mask.sum()))
    return mutated


def next_generation(
    genomes: Sequence[np.ndarray],
    fitnesses: Sequence[float],
    cfg: GeneticConfig,
    rng: np.random.Generator,
) -> List[np.ndarray]:
    """Build the next population from the current one."""
    pool = select_breeding_pool(genomes, fitnesses, cfg)
    elites = [g.copy() for g in pool[: cfg.elite_count]]

    children: List[np.ndarray] = list(elites)
    while len(children) < cfg.population_size:
        a, b = rng.choice(len(pool), size=2, replace=len(pool) < 2)
        child = crossover(pool[a], pool[b], rng)
        children.append(mutate(child, cfg, rng))

    logger.debug("New generation: %d children from a pool of %d", len(children), len(pool))
    return children
