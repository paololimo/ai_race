"""Everything the report draws, read back off disk.

A run leaves four kinds of trace: a checkpoint per entrant, the training log,
the race log, and one JSON per ablation group. Each is parsed here into a frozen
record, and nothing downstream touches a file — so a figure can be re-drawn
without a run, and a missing trace is a missing figure rather than a crash.

Two things a run *does not* leave are genetic spread and the per-circuit best.
Both are computed each generation and held only in memory, so no amount of
parsing recovers them; persisting them is two lines in `Simulation.save_all`
and a retrain, which is why they are absent here rather than approximated.
"""

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_GENERATION = re.compile(
    r"^(?P<stamp>\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+\s+INFO gen\s+(?P<generation>\d+)\s*\|\s*"
    r"(?P<scores>.+)$"
)
_SCORE = re.compile(r"(?P<name>[A-Za-z_][\w-]*)\s+(?P<value>-?\d+\.\d+)")
_CLASSIFICATION = re.compile(
    r"^\s*(?P<rank>\d+)\.\s+(?P<name>\S+)\s+(?P<laps>-?\d+\.\d+)\s+(?P<note>.+?)\s*$"
)
# The table times the winner absolutely and everyone else as a gap to it, so
# the two have to be told apart: reading "+6 frames" as a finishing time put
# the runner-up ahead of the winner by three hundred laps' worth of clock.
_WINNER = re.compile(r"WINNER\D+(?P<frames>\d+) frames")
_GAP = re.compile(r"\+(?P<frames>\d+) frames")


@dataclass(frozen=True)
class Entrant:
    """One competitor's champion and the history that produced it."""

    name: str
    history: np.ndarray  # best fitness in each generation
    best_fitness: float
    parameters: int
    generations: int
    population: int
    seed: int
    brain: str

    @property
    def running_best(self) -> np.ndarray:
        """The monotone envelope — the only honest progress curve.

        Each generation draws its own start point and island layout, so the raw
        per-generation best rises and falls with the draw even for an elite
        carried over untouched. The running maximum answers "is anyone still
        improving?"; the raw scores are kept because the gap between them is
        how much of a given entrant's score is luck.
        """
        return np.maximum.accumulate(self.history)

    @property
    def evaluations(self) -> int:
        """Genomes scored across the run, at three circuits a generation."""
        return self.generations * self.population * 3


@dataclass(frozen=True)
class Standing:
    """One line of the race classification."""

    rank: int
    name: str
    laps: float
    note: str
    finished_frames: Optional[int]

    @property
    def finished(self) -> bool:
        return self.finished_frames is not None


@dataclass(frozen=True)
class AblationRow:
    """One variant, over every seed it was run on."""

    variant: str
    parameters: int
    training: Tuple[float, ...]
    race: Tuple[float, ...]

    @property
    def race_median(self) -> float:
        return float(np.median(self.race))

    @property
    def training_median(self) -> float:
        return float(np.median(self.training))


@dataclass(frozen=True)
class AblationGroup:
    """One question, and the variants that answer it."""

    key: str
    title: str
    rows: Tuple[AblationRow, ...]


# -- checkpoints -------------------------------------------------------------


def load_entrants(outputs: Path) -> List[Entrant]:
    """Every champion in `outputs/`, in the order they finished training best.

    A `.npz` here is a champion by construction — `save_all` writes nothing
    else — so the ablation's JSON and the videos are simply not `.npz` files
    and need no exclusion list.
    """
    entrants = []
    for path in sorted(outputs.glob("*.npz")):
        data = np.load(path, allow_pickle=False)
        if "genome" not in data:
            logger.warning("%s is not a champion checkpoint — skipped", path.name)
            continue
        entrants.append(
            Entrant(
                name=path.stem,
                history=np.asarray(data["history"], dtype=float),
                best_fitness=float(data["fitness"]),
                parameters=int(np.asarray(data["genome"]).size),
                generations=int(data["generations"]),
                population=int(data["population"]),
                seed=int(data["seed"]),
                brain=str(data["brain"]),
            )
        )
    return sorted(entrants, key=lambda e: -e.best_fitness)


# -- logs --------------------------------------------------------------------


def parse_training_log(path: Path) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Per-entrant scores and per-generation wall-clock, from the training log.

    The checkpoints carry the same scores, so this exists for the one thing they
    do not: the timestamps, which are what a claim about the cost of a run has
    to rest on. It also recovers the history of an entrant whose checkpoint was
    overwritten by a later, shorter run.
    """
    scores: Dict[str, List[float]] = {}
    stamps: List[datetime] = []
    for line in path.read_text().splitlines():
        match = _GENERATION.match(line)
        if match is None:
            continue
        stamps.append(datetime.strptime(match["stamp"], "%Y-%m-%d %H:%M:%S"))
        for name, value in _SCORE.findall(match["scores"]):
            scores.setdefault(name, []).append(float(value))
    # The stamp is written when a generation *ends*, so the gaps between them
    # are the generation durations and the first generation has none.
    seconds = np.diff(np.array([s.timestamp() for s in stamps])) if len(stamps) > 1 else np.empty(0)
    return {name: np.array(values) for name, values in scores.items()}, seconds


def parse_race_log(path: Path) -> List[Standing]:
    """The classification table, read back out of the race log.

    Only the block under "Classification:" is a table; everything above it is
    ordinary logging that happens to contain numbers, so parsing starts there.
    """
    lines = path.read_text().splitlines()
    start = next((i for i, line in enumerate(lines) if "Classification:" in line), None)
    if start is None:
        return []
    standings = []
    won: Optional[int] = None
    for line in lines[start + 1 :]:
        match = _CLASSIFICATION.match(line)
        if match is None:
            if standings:  # the table has ended
                break
            continue
        note = match["note"]
        winner, gap = _WINNER.search(note), _GAP.search(note)
        if winner is not None:
            won = int(winner["frames"])
            frames: Optional[int] = won
        elif gap is not None and won is not None:
            frames = won + int(gap["frames"])
        else:  # crashed out, or still running when the limit came
            frames = None
        standings.append(
            Standing(
                rank=int(match["rank"]),
                name=match["name"],
                laps=float(match["laps"]),
                note=note,
                finished_frames=frames,
            )
        )
    return standings


# -- ablations ---------------------------------------------------------------


def _group_titles() -> Dict[str, Tuple[str, Tuple[str, ...]]]:
    """`ablation.py`'s own questions, so this file does not restate them."""
    try:
        from experiments.ablation import GROUPS
    except ImportError:  # the ablation script is not importable from here
        return {}
    return {
        key: (title, tuple(v.name for v in variants))
        for key, (variants, title) in GROUPS.items()
    }


def load_ablations(outputs: Path) -> List[AblationGroup]:
    """Every `ablation_*.json`, split back into the questions it answers.

    The JSON is a flat list of variants with no group field, so membership is
    recovered from `ablation.py`'s own definitions. A file holding a group that
    definition no longer knows about is still reported — under its filename,
    rather than dropped for disagreeing with the current source.
    """
    known = _group_titles()
    groups: List[AblationGroup] = []
    for path in sorted(outputs.glob("ablation_*.json")):
        try:
            rows = [
                AblationRow(
                    variant=row["variant"],
                    parameters=int(row["parameters"]),
                    training=tuple(float(v) for v in row["training"]),
                    race=tuple(float(v) for v in row["race"]),
                )
                for row in json.loads(path.read_text())
            ]
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            logger.warning("%s could not be read as ablation results: %s", path.name, error)
            continue

        remaining = {row.variant: row for row in rows}
        for key, (title, names) in known.items():
            claimed = tuple(remaining.pop(name) for name in names if name in remaining)
            if claimed:
                groups.append(AblationGroup(key, title, claimed))
        if remaining:
            stem = path.stem.replace("ablation_", "")
            title = stem.replace("_", " ").upper()
            groups.append(AblationGroup(stem, title, tuple(remaining.values())))
    return groups


_TARGET = re.compile(r"never trained on,\s*(?P<laps>\d+(?:\.\d+)?) laps")


def race_target(path: Path, fallback: float = 5.0) -> float:
    """How many laps the race was run to, taken from the log that ran it.

    Hard-coding five would silently mislabel the flag on the chart the first
    time someone races `--laps 3`.
    """
    match = _TARGET.search(path.read_text())
    return float(match["laps"]) if match else fallback


def seed_count(groups: Sequence[AblationGroup]) -> int:
    """The smallest number of seeds behind any variant shown."""
    return min((len(row.race) for group in groups for row in group.rows), default=0)


__all__ = [
    "AblationGroup",
    "AblationRow",
    "Entrant",
    "Standing",
    "load_ablations",
    "load_entrants",
    "parse_race_log",
    "parse_training_log",
    "race_target",
    "seed_count",
]
