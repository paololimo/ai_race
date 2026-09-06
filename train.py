"""Train every entrant at once, on the same circuits, in one window.

    python train.py                       # dashboard, everyone training together
    python train.py --headless            # no window, all cores
    python train.py --generations 200
    python train.py --headless --record outputs/training.mp4   # every frame

The entrants are whatever files are in `src/brains/`: adding one there puts it
in the next run with no change here. Each champion is written to
`outputs/<name>.npz`, which is what `race.py` reads.
"""

import argparse

from src.cli import add_common_arguments, build_simulation, default_workers, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evolve every entrant together")
    parser.add_argument("--generations", type=int, default=120)
    parser.add_argument("--population", type=int, default=100)
    parser.add_argument(
        "--shown",
        type=int,
        default=10,
        help="cars drawn per entrant; the whole population is scored regardless",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=default_workers(),
        help="processes used for scoring (default: cores - 2)",
    )
    parser.add_argument(
        "--record-clip",
        type=float,
        default=None,
        help="film only the first N seconds of each circuit, in one generation "
        "in --record-every, instead of the whole run. Turns a 120-generation "
        "run into minutes rather than hours, at the cost of cutting between "
        "clips. Without it every frame is kept.",
    )
    parser.add_argument(
        "--record-every",
        type=int,
        default=3,
        help="with --record-clip: film one generation in N (default 3).",
    )
    add_common_arguments(parser)
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    simulation = build_simulation(
        args,
        workers=args.workers,
        shown=args.shown,
        record_every=args.record_every,
        record_clip=args.record_clip,
    )
    simulation.train()


if __name__ == "__main__":
    main()
