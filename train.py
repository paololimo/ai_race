"""Train a population on the three training circuits.

The champion is written to `outputs/best_genome.npz` and to a second file named
after the entrant, which is what `race.py` takes to the circuit it has never
seen. Every entrant must be trained on the same seed, generations and
population for the race to mean anything:

    python train.py --brain baseline --entrant human  --seed 42 --generations 120 --headless
    python train.py --brain codex    --entrant codex  --seed 42 --generations 120 --headless
"""

import argparse
import os

from src.cli import add_brain_arguments, add_common_arguments, build_config, setup_logging
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
    add_brain_arguments(parser)
    add_common_arguments(parser)
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    cfg = build_config(args)
    simulation = Simulation(cfg, render=not args.headless, workers=args.workers)
    simulation.train()
    # One file per entrant, so several brains can be trained side by side
    # without overwriting each other's champion.
    simulation.save_best(args.out.name if args.out else f"{cfg.entrant}_genome.npz")


if __name__ == "__main__":
    main()
