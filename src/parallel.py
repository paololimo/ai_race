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
from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.brains import BrainRef, build_brain
from src.car import Car, network_input_size
from src.config import SimulationConfig, track_variants
from src.track import Track

# Per-process state: circuits are expensive to build, so each worker builds them
# once and keeps them for the life of the pool.
_STATE: Dict[str, object] = {}

Job = Tuple[int, int, Optional[int], int, List[np.ndarray], BrainRef]


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
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame

    pygame.init()
    _STATE["cfg"] = cfg
    _STATE["circuits"] = build_circuits(cfg)
    _STATE["rng"] = np.random.default_rng(0)
    _STATE["inputs"] = network_input_size(cfg.car)


def evaluate(job: Job) -> Tuple[int, List[float]]:
    """Drive one block of genomes on one circuit; return their lap scores."""
    chunk_id, track_number, variant, start_index, genomes, brain = job
    cfg: SimulationConfig = _STATE["cfg"]  # type: ignore[assignment]
    circuits: Sequence[Sequence[Track]] = _STATE["circuits"]  # type: ignore[assignment]
    rng: np.random.Generator = _STATE["rng"]  # type: ignore[assignment]
    inputs: int = _STATE["inputs"]  # type: ignore[assignment]

    track = circuits[track_number][variant]
    budget = int(cfg.laps_budget * track.lap_length / cfg.car.max_speed)

    cars = []
    for genome in genomes:
        driver = build_brain(brain, inputs, rng)
        driver.set_genome(genome)
        cars.append(Car(track, driver, cfg.car, start_index))

    for _ in range(budget):
        for car in cars:
            car.update()
        if not any(car.alive for car in cars):
            break
    return chunk_id, [car.fitness / track.lap_length for car in cars]


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
