"""Put every champion on the grid together, on the circuit none of them saw.

    python race.py                        # the race, on `gauntlet`
    python race.py --laps 3               # a shorter race
    python race.py --track 1              # the same grid on a training circuit
    python race.py --headless --record outputs/race.mp4

Driving a layout that was not in the training set is the difference between
having learned to drive and having memorised three tracks. The entrants are
whichever files are in `src/brains/` and have a champion in `outputs/`.
"""

import argparse
import logging

from src.cli import add_common_arguments, build_simulation, setup_logging
from src.simulation import format_results

logger = logging.getLogger("race")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Race every champion together")
    parser.add_argument(
        "--laps",
        type=float,
        default=5.0,
        help="laps to race (default 5). First past this distance wins; anyone "
        "who never gets there is placed behind the finishers, by distance.",
    )
    parser.add_argument(
        "--track",
        type=int,
        default=None,
        help="race on this training circuit (0-2) instead of the unseen one",
    )
    add_common_arguments(parser)
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    # A race is one lap of one circuit, so it is filmed whole — no clips, no
    # cuts, and none of train.py's clip flags.
    simulation = build_simulation(args)
    track = simulation.tracks[args.track] if args.track is not None else None
    results = simulation.race(track, laps=args.laps)
    logger.info("Classification:\n%s", format_results(results))


if __name__ == "__main__":
    main()
