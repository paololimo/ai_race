"""Put every champion on the grid together, on the circuit none of them saw.

    python race.py                        # the race, on `gauntlet`
    python race.py --track 1              # the same grid on a training circuit

Driving a layout that was not in the training set is the difference between
having learned to drive and having memorised three tracks. The entrants are
whichever files are in `src/brains/` and have a champion in `outputs/`.
"""

import argparse
import logging

from src.cli import add_common_arguments, build_config, setup_logging
from src.simulation import Simulation, format_results

logger = logging.getLogger("race")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Race every champion together")
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
    simulation = Simulation(build_config(args), render=not args.headless)
    track = simulation.tracks[args.track] if args.track is not None else None
    results = simulation.race(track)
    logger.info("Classification:\n%s", format_results(results))


if __name__ == "__main__":
    main()
