"""Evaluate a population across worker processes.

Within a generation every genome drives alone: no shared state, no interaction,
no ordering. That makes the evaluation step the natural place to use the other
cores — and it is where nearly all the time goes.

The result is identical to evaluating in one process. Loading a genome
overwrites every weight, so the random draw used to build a network never
reaches the outcome; the only randomness that matters (initial population,
crossover, mutation, start point) stays in the parent process. Parallel runs
therefore reproduce serial ones exactly, which the tests check.
"""

import os
from dataclasses import dataclass, replace
from typing import List, Optional, Sequence, Tuple

import numpy as np

from src.brains import BrainRef
from src.car import build_car, network_input_size
from src.config import SimulationConfig, track_variants
from src.track import Track


@dataclass(frozen=True)
class _Worker:
    """What a worker process sets up once and reuses for the life of the pool."""

    cfg: SimulationConfig
    circuits: List[List[Track]]
    rng: np.random.Generator
    inputs: int


# Circuits are expensive to build, so each worker builds them once and keeps them.
_WORKER: Optional[_Worker] = None


@dataclass(frozen=True)
class Job:
    """One block of genomes, and the conditions to score it under.

    The conditions are the parent's `Stage`, budget included, rather than
    anything a worker re-derives: the drawn cars and the pooled cars have to be
    driven over the same horizon for their scores to mean the same thing.

    `squad` is handed back untouched so the caller can reassemble its results;
    the pool preserves job order, so the blocks of one population arrive in the
    order they were cut in.
    """

    squad: int
    track_number: int
    variant: int
    start_index: Optional[int]
    budget: int
    genomes: List[np.ndarray]
    brain: BrainRef


def build_circuits(cfg: SimulationConfig) -> List[List[Track]]:
    """Every circuit, in each of its island layouts.

    Derived entirely from the configuration and its seed, so the parent process
    and every worker independently build byte-identical geometry — which is why
    the circuits are rebuilt in each process rather than sent to it. (They could
    not be sent anyway: a Track holds a pygame Surface, which cannot be pickled.)
    """
    circuits: List[List[Track]] = []
    for track_cfg in track_variants():
        built = [Track(track_cfg)]
        jitter = np.random.default_rng(cfg.seed)
        for _ in range(max(1, cfg.island_variants) - 1):
            shifted = tuple(
                float((fraction + jitter.uniform(0.05, 0.95)) % 1.0)
                for fraction in track_cfg.islands
            )
            built.append(Track(replace(track_cfg, islands=shifted)))
        circuits.append(built)
    return circuits


def worker_init(cfg: SimulationConfig) -> None:
    global _WORKER
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame

    pygame.init()
    _WORKER = _Worker(cfg, build_circuits(cfg), np.random.default_rng(0), network_input_size(cfg.car))


def evaluate(job: Job) -> Tuple[int, List[float]]:
    """Drive one block of genomes on one circuit; return their lap scores."""
    assert _WORKER is not None, "evaluate() runs in a pool started by worker_init"
    track = _WORKER.circuits[job.track_number][job.variant]

    cars = [
        build_car(
            job.brain,
            genome,
            track,
            _WORKER.cfg.car,
            _WORKER.inputs,
            _WORKER.rng,
            job.start_index,
        )
        for genome in job.genomes
    ]

    for _ in range(job.budget):
        for car in cars:
            car.update()
        if not any(car.alive for car in cars):
            break
    return job.squad, [car.fitness / track.lap_length for car in cars]


BLOCKS_PER_WORKER = 3


def split(genomes: Sequence[np.ndarray], workers: int) -> List[List[np.ndarray]]:
    """Cut the population into blocks for the pool.

    More blocks than workers on purpose: cars die at wildly different times, so
    equal-sized blocks take unequal time, and with exactly one block each every
    worker waits for the slowest. Smaller blocks let a worker that finishes
    early pick up more work.
    """
    if workers <= 1:
        return [list(genomes)]
    parts = min(len(genomes), workers * BLOCKS_PER_WORKER)
    size = max(1, -(-len(genomes) // parts))  # ceiling division
    return [list(genomes[i : i + size]) for i in range(0, len(genomes), size)]
