"""Train every entrant at once, on the same circuits, in one window.

    python train.py                       # dashboard, everyone training together
    python train.py --headless            # no window, all cores
    python train.py --generations 200

The entrants are whatever files are in `src/brains/`: adding one there puts it
in the next run with no change here. Each champion is written to
`outputs/<name>.npz`, which is what `race.py` reads.
"""

import argparse
import os

from src.cli import add_common_arguments, build_config, setup_logging
from src.simulation import Simulation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evolve every entrant together")
    parser.add_argument("--generations", type=int, default=120)
    parser.add_argument("--population", type=int, default=100)
    parser.add_argument(
        "--shown",
        type=int,
        default=8,
        help="cars drawn per entrant; the whole population is scored regardless",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 2),
        help="processes used for scoring (default: cores - 2)",
    )
    add_common_arguments(parser)
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    simulation = Simulation(
        build_config(args),
        render=not args.headless,
        workers=args.workers,
        shown=args.shown,
    )
    simulation.train()


if __name__ == "__main__":
    main()
