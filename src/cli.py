"""Shared plumbing for the `train.py` and `race.py` entry points."""

import argparse
import logging
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

from src.config import SimulationConfig
from src.simulation import Simulation


def default_workers() -> int:
    """Cores left over once the display and the operating system have theirs."""
    return max(1, (os.cpu_count() or 2) - 2)


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


def build_simulation(args: argparse.Namespace, **extra: Any) -> Simulation:
    """Build the `Simulation` an entry point asked for.

    Recording headless still needs everything drawn — just not shown. The dummy
    video driver gives pygame a surface with no window on it, and with no window
    there is no reason to hold sixty frames a second, so the recording takes as
    long as the drawing does rather than as long as watching would have. That
    one rule turns `--headless --record` into three constructor arguments, which
    is why it is derived here rather than at each entry point.
    """
    recording_headless = args.headless and args.record is not None
    if recording_headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    return Simulation(
        build_config(args),
        render=not args.headless or recording_headless,
        record=args.record,
        throttle=not recording_headless,
        **extra,
    )
