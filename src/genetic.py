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


def mutation_sigma(cfg: GeneticConfig, progress: float) -> float:
    """Mutation size at `progress` through the run, from 0.0 to 1.0.

    Geometric, not linear: the useful quantity is the *ratio* of the step to the
    weights, so equal fractions of the run should shrink it by equal factors.
    A fixed size cannot serve both ends of a run — early on the population must
    cross the space, late on it must settle into a solution it has already
    found, and a step as large as the weights themselves can only ever explore.
    """
    span = max(cfg.mutation_scale_final, 1e-9) / max(cfg.mutation_scale, 1e-9)
    return cfg.mutation_scale * span ** min(max(progress, 0.0), 1.0)


def mutate(
    genome: np.ndarray,
    cfg: GeneticConfig,
    rng: np.random.Generator,
    progress: float = 0.0,
) -> np.ndarray:
    """Perturb a small fraction of genes with gaussian noise."""
    mutated = genome.copy()
    mask = rng.random(genome.size) < cfg.mutation_rate
    mutated[mask] += rng.normal(0.0, mutation_sigma(cfg, progress), size=int(mask.sum()))
    return mutated


def _rank_weights(count: int) -> np.ndarray:
    """Parent-choice probabilities over a fitness-ranked pool, best first.

    Drawing parents uniformly from the pool treats the best genome in it and
    the one scraping the cut-off as equally worth copying, which throws away
    the ranking selection has just produced. Weighting by log-decreasing rank
    is what the evolution-strategy literature settles on: the population's
    shift from one generation to the next is a noisy estimate of the direction
    fitness improves in, and this is the weighting that estimates it best.
    """
    weights = np.log(count + 0.5) - np.log(np.arange(1, count + 1))
    return weights / weights.sum()


def next_generation(
    genomes: Sequence[np.ndarray],
    fitnesses: Sequence[float],
    cfg: GeneticConfig,
    rng: np.random.Generator,
    progress: float = 0.0,
) -> List[np.ndarray]:
    """Build the next population from the current one."""
    pool = select_breeding_pool(genomes, fitnesses, cfg)
    elites = [g.copy() for g in pool[: cfg.elite_count]]
    weights = _rank_weights(len(pool))

    children: List[np.ndarray] = list(elites)
    while len(children) < cfg.population_size:
        a, b = rng.choice(len(pool), size=2, replace=len(pool) < 2, p=weights)
        child = crossover(pool[a], pool[b], rng, cfg.crossover)
        children.append(mutate(child, cfg, rng, progress))

    logger.debug("New generation: %d children from a pool of %d", len(children), len(pool))
    return children
