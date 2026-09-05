"""Shared plumbing for the `train.py` and `race.py` entry points."""

import argparse
import logging
from dataclasses import replace
from pathlib import Path

from src.config import SimulationConfig


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--headless", action="store_true", help="run without a window")
    parser.add_argument(
        "--record",
        type=Path,
        default=None,
        help="write an mp4 of the window (needs ffmpeg). Combine with --headless "
        "to record faster than real time: the frame rate cap is then lifted.",
    )
    parser.add_argument(
        "--record-every",
        type=int,
        default=3,
        help="film one generation in N (default 3). Frames are never thinned — "
        "that would replace motion with a slideshow — so this and --record-clip "
        "are what decide the video's length.",
    )
    parser.add_argument(
        "--record-clip",
        type=float,
        default=3.0,
        help="seconds filmed of each circuit, in each filmed generation "
        "(default 3.0). Length = generations / every x circuits x clip, so the "
        "defaults turn a 120-generation run into six minutes.",
    )


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
