"""Shared plumbing for the `train.py` and `race.py` entry points."""

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path

from src.brains import BrainRef, available_brains
from src.config import SimulationConfig

DEFAULT_GENOME = Path("outputs/best_genome.npz")


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--headless", action="store_true", help="run without a window")


def add_brain_arguments(parser: argparse.ArgumentParser) -> None:
    """Which brain to evolve, and who to credit it to.

    The brain is the only thing a competing agent supplies; everything the
    comparison depends on — circuits, physics, fitness, genetic algorithm, seed
    — comes from the configuration and is identical for every entrant.
    """
    parser.add_argument(
        "--brain",
        default="baseline",
        help=f"registered brain to evolve (available: {', '.join(available_brains())})",
    )
    parser.add_argument(
        "--brain-spec",
        type=str,
        default=None,
        help='JSON hyperparameters for the brain, e.g. \'{"hidden_sizes": [20, 20]}\'',
    )


def build_config(args: argparse.Namespace) -> SimulationConfig:
    """Apply command-line overrides to the project defaults.

    `dataclasses.replace` returns a copy, so the configuration stays immutable:
    what gets logged is what actually ran.
    """
    base = SimulationConfig()
    spec_text = getattr(args, "brain_spec", None)
    try:
        spec = json.loads(spec_text) if spec_text else {}
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--brain-spec is not valid JSON: {exc}") from None
    brain_name = getattr(args, "brain", base.brain.name)
    return replace(
        base,
        seed=args.seed,
        generations=getattr(args, "generations", base.generations),
        brain=BrainRef(brain_name, spec),
        genetic=replace(
            base.genetic,
            population_size=getattr(args, "population", base.genetic.population_size),
        ),
    )
