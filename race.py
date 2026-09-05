"""Take trained genomes to the race circuit — the one they never trained on.

This is the test the whole project exists for: driving a layout that was not in
the training set is the difference between having learned to drive and having
memorised three tracks.

    python race.py                                        # one champion, solo
    python race.py --track 1                               # replay on a training circuit
    python race.py --grid outputs/human_genome.npz \\
                          outputs/codex_genome.npz         # everyone at once
"""

import argparse
import logging
from pathlib import Path

from src.car import network_input_size
from src.cli import DEFAULT_GENOME, add_common_arguments, build_config, setup_logging
from src.grand_prix import GrandPrix, build_grid, format_results, save_results
from src.simulation import Simulation

logger = logging.getLogger("race")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Race trained genomes")
    parser.add_argument(
        "genome",
        nargs="?",
        type=Path,
        default=DEFAULT_GENOME,
        help=f"genome to drive alone (default: {DEFAULT_GENOME})",
    )
    parser.add_argument(
        "--grid",
        nargs="+",
        type=Path,
        default=None,
        metavar="GENOME",
        help="race these champions together on the race circuit",
    )
    parser.add_argument(
        "--track",
        type=int,
        default=None,
        help="replay on this training circuit (0-2) instead of the race circuit",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("outputs/grand_prix.json"),
        help="where to write the classification",
    )
    add_common_arguments(parser)
    return parser.parse_args()


def run_grand_prix(args: argparse.Namespace) -> None:
    cfg = build_config(args)
    missing = [p for p in args.grid if not p.exists()]
    if missing:
        raise SystemExit("No genome at: " + ", ".join(str(p) for p in missing))

    entrants = build_grid(args.grid, network_input_size(cfg.car))
    grand_prix = GrandPrix(cfg, render=not args.headless)
    results = grand_prix.run(entrants)
    grand_prix.close()
    logger.info("Final classification:\n%s", format_results(results))
    save_results(results, args.results)


def main() -> None:
    setup_logging()
    args = parse_args()
    if args.grid:
        run_grand_prix(args)
        return

    if not args.genome.exists():
        raise SystemExit(f"No genome at {args.genome} — run train.py first.")
    simulation = Simulation(build_config(args), render=not args.headless)
    if args.track is None:
        simulation.race(args.genome)
    else:
        simulation.replay_best(args.genome, track_number=args.track)


if __name__ == "__main__":
    main()
