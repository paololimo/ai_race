"""Shared plumbing for the `train.py` and `race.py` entry points."""

import argparse
import logging
from dataclasses import replace
from pathlib import Path

from src.config import SimulationConfig

DEFAULT_GENOME = Path("outputs/best_genome.npz")


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--headless", action="store_true", help="run without a window")


def build_config(args: argparse.Namespace) -> SimulationConfig:
    """Apply command-line overrides to the project defaults.

    `dataclasses.replace` returns a copy, so the configuration stays immutable:
    what gets logged is what actually ran.
    """
    base = SimulationConfig()
    return replace(
        base,
        seed=args.seed,
        generations=getattr(args, "generations", base.generations),
        genetic=replace(
            base.genetic,
            population_size=getattr(args, "population", base.genetic.population_size),
        ),
    )
