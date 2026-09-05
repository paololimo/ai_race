"""Take a trained genome to the race circuit — the one it never trained on.

This is the test the whole project exists for: driving a layout that was not in
the training set is the difference between having learned to drive and having
memorised three tracks. Pass `--track N` instead to watch the same genome on one
of the training circuits.
"""

import argparse
from pathlib import Path

from src.cli import DEFAULT_GENOME, add_common_arguments, build_config, setup_logging
from src.simulation import Simulation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Race a trained genome")
    parser.add_argument(
        "genome",
        nargs="?",
        type=Path,
        default=DEFAULT_GENOME,
        help=f"genome to drive (default: {DEFAULT_GENOME})",
    )
    parser.add_argument(
        "--track",
        type=int,
        default=None,
        help="replay on this training circuit (0-2) instead of the race circuit",
    )
    add_common_arguments(parser)
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    if not args.genome.exists():
        raise SystemExit(f"No genome at {args.genome} — run train.py first.")

    simulation = Simulation(build_config(args), render=not args.headless)
    if args.track is None:
        simulation.race(args.genome)
    else:
        simulation.replay_best(args.genome, track_number=args.track)


if __name__ == "__main__":
    main()
