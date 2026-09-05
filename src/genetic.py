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


def crossover(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    rng: np.random.Generator,
    mode: str = "uniform",
) -> np.ndarray:
    """Recombine two genomes.

    `uniform` draws every gene independently, which is the harshest of the three
    on a neural genome: a neuron's incoming weights only mean anything together,
    and drawing each from a different parent hands the child a unit that detects
    neither of the things its parents detected. Recombination then behaves more
    like heavy mutation than like inheritance.

    `one-point` and `two-point` cut the vector into runs instead, so whole
    stretches of a parent survive intact — and with them, whatever co-adapted
    groups happen to sit inside a run. `none` skips recombination entirely and
    leaves mutation to do the work, which on small genomes is often no worse.

    This is a property of the harness, identical for every entrant.
    """
    size = parent_a.size
    if mode == "none":
        return parent_a.copy()
    if mode == "uniform":
        return np.where(rng.random(size) < 0.5, parent_a, parent_b)
    if mode in ("one-point", "two-point"):
        count = 1 if mode == "one-point" else 2
        cuts = np.sort(rng.choice(np.arange(1, size), size=min(count, size - 1), replace=False))
        taking, child, previous = True, np.empty_like(parent_a), 0
        for cut in (*cuts, size):
            source = parent_a if taking else parent_b
            child[previous:cut] = source[previous:cut]
            taking, previous = not taking, cut
        return child
    raise ValueError(f"Unknown crossover mode {mode!r}")


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
        child = crossover(pool[a], pool[b], rng, cfg.crossover)
        children.append(mutate(child, cfg, rng))

    logger.debug("New generation: %d children from a pool of %d", len(children), len(pool))
    return children
