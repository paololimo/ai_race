"""Train a population on the three training circuits.

The best genome of the run is written to `outputs/best_genome.npz`, which is
what `race.py` then takes to the circuit it has never seen.
"""

import argparse
import os

from src.cli import add_common_arguments, build_config, setup_logging
from src.simulation import Simulation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evolve self-driving cars")
    parser.add_argument("--generations", type=int, default=120)
    parser.add_argument("--population", type=int, default=100)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 2),
        help="parallel processes for --headless runs (default: cores - 2)",
    )
    add_common_arguments(parser)
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    simulation = Simulation(
        build_config(args), render=not args.headless, workers=args.workers
    )
    simulation.train()


if __name__ == "__main__":
    main()
