"""Turn a finished run into report-ready figures and tables.

    python experiments/make_report.py                    # everything on disk
    python experiments/make_report.py --png              # raster copies too
    python experiments/make_report.py --out paper/figures

Reads `outputs/`: the champions, the training log, the race log, and any
`ablation_*.json`. Whatever is missing is skipped with a line saying so, so this
is worth running mid-project and not only at the end.

Figures are written as PDF — vector, so text stays text and nothing blurs at
print size. `--png` adds a 300 dpi copy of each for slides and for anything that
will not take a PDF.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

from matplotlib.figure import Figure

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.report import collect, figures, tables
from experiments.report.style import use_report_style
from src.config import race_track, track_variants

logger = logging.getLogger("report")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Figures and tables from a finished run")
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--out", type=Path, default=Path("outputs/report"))
    parser.add_argument(
        "--png",
        action="store_true",
        help="also write a 300 dpi raster copy of every figure",
    )
    parser.add_argument(
        "--at",
        type=int,
        nargs="+",
        default=(20, 60, 120),
        help="generations to tabulate the per-circuit breakdown at (default 20 60 120)",
    )
    parser.add_argument(
        "--no-tracks",
        action="store_true",
        help="skip the circuit maps, which have to build the tracks to draw them",
    )
    return parser.parse_args()


class Writer:
    """Saves figures and text, and remembers what it saved."""

    def __init__(self, out: Path, png: bool) -> None:
        out.mkdir(parents=True, exist_ok=True)
        self.out = out
        self.png = png
        self.written: List[Path] = []

    def figure(self, figure: Optional[Figure], name: str) -> None:
        if figure is None:
            return
        path = self.out / f"{name}.pdf"
        figure.savefig(path)
        self.written.append(path)
        if self.png:
            raster = self.out / f"{name}.png"
            figure.savefig(raster, dpi=300)
            self.written.append(raster)

    def text(self, body: str, name: str) -> None:
        if not body.strip():
            return
        path = self.out / name
        path.write_text(body.rstrip() + "\n")
        self.written.append(path)


def _tracks():
    """The four circuits, built the way the simulation builds them.

    Building a track rasterises it and derives a clearance field, which is why
    this is behind a flag: it is the one slow thing in the script, and it is
    only needed the first time the circuits go into a document.
    """
    import os

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame

    from src.track import Track

    pygame.init()
    configs = [*track_variants(), race_track()]
    return [Track(cfg) for cfg in configs], [cfg.name for cfg in configs]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # matplotlib subsets a font per figure and says so, at length, on the
    # same INFO channel as this script's own output.
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    args = parse_args()
    use_report_style()
    writer = Writer(args.out, args.png)
    sections: Dict[str, str] = {}
    latex: List[str] = []

    entrants = collect.load_entrants(args.outputs)
    seconds = None
    training_log = args.outputs / "training.log"
    if training_log.exists():
        _, seconds = collect.parse_training_log(training_log)
        writer.figure(figures.generation_cost(seconds), "generation-cost")
    else:
        logger.info("no training.log — skipping the cost of the run")

    if entrants:
        circuits = [cfg.name for cfg in track_variants()]
        writer.figure(figures.training_curves(entrants), "training-curves")
        writer.figure(figures.genetic_spread(entrants), "genetic-spread")
        writer.figure(figures.per_circuit(entrants, circuits), "per-circuit")
        sections["The run"] = tables.run_summary(entrants, seconds)
        # The two series the dashboard shows and no report could quote, now
        # that they are in the checkpoints: as CSV to be re-plotted elsewhere,
        # and at three generations as a table to be read.
        writer.text(tables.spread_csv(entrants), "spread.csv")
        writer.text(tables.circuits_csv(entrants, circuits), "circuits.csv")
        divisors = tables.spread_divisors(entrants)
        if divisors:
            sections["Genetic spread"] = (
                "`spread.csv` holds every generation, each entrant divided by its own "
                "generation-1 spread:\n\n" + divisors
            )
        breakdown = tables.circuits_at(entrants, circuits, args.at)
        if breakdown:
            sections["Best per circuit"] = breakdown["markdown"]
            latex.append(breakdown["latex"])
    else:
        logger.info("no champions in %s — run train.py first", args.outputs)

    race_log = args.outputs / "race.log"
    standings = collect.parse_race_log(race_log) if race_log.exists() else []
    if standings:
        target = collect.race_target(race_log)
        writer.figure(figures.race_standings(standings, target), "race-standings")
        if entrants:
            table = tables.standings_table(entrants, standings)
            sections["Results"] = table["markdown"]
            latex.append(table["latex"])
    else:
        logger.info("no classification in %s — skipping the race", race_log)

    groups = collect.load_ablations(args.outputs)
    for group in groups:
        writer.figure(figures.ablation_dots(group), f"ablation-{group.key}")
        table = tables.ablation_table(group)
        sections[group.title] = table["markdown"]
        latex.append(table["latex"])
    if groups:
        writer.figure(figures.parameters_against_race(groups), "parameters-against-race")
        seeds = collect.seed_count(groups)
        if seeds < 5:
            sections["Caveat"] = (
                f"The ablations ran on {seeds} seed(s). On this problem the seed-to-seed "
                f"spread swamps the difference between variants: treat the tables above as "
                f"provisional until they are re-run at seven seeds or more."
            )
    else:
        logger.info("no ablation_*.json in %s — skipping the ablations", args.outputs)

    if not args.no_tracks:
        tracks, names = _tracks()
        writer.figure(figures.track_maps(tracks, names), "circuits")

    writer.text(
        "\n\n".join(f"## {title}\n\n{body}" for title, body in sections.items()),
        "results.md",
    )
    writer.text("\n\n".join(latex), "results.tex")

    logger.info("\n%d files in %s:", len(writer.written), args.out)
    for path in writer.written:
        logger.info("  %s", path)


if __name__ == "__main__":
    main()
