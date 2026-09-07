"""The report's parsers, which are the only place it can go quietly wrong.

A figure that is drawn wrong is usually obvious. A log line read wrong is not:
the classification times the winner absolutely and everyone else as a gap to
it, and reading the gap as a time put the runner-up ahead of the winner on a
chart that otherwise looked entirely reasonable. That is what these cover.
"""

import json

import numpy as np

from experiments.report.collect import (
    load_ablations,
    load_entrants,
    parse_race_log,
    parse_training_log,
    race_target,
)
from experiments.report.tables import circuits_at, circuits_csv, spread_csv

RACE_LOG = """2026-09-06 03:14:24,563 INFO Recording to outputs/race.mp4 at 60 fps
2026-09-06 03:14:24,673 INFO Race on 'gauntlet', never trained on, 5 laps: a, b, c
2026-09-06 03:14:46,577 INFO Classification:
    entrant          laps  result
----------------------------------------------------
 1. paololimo        5.00  WINNER — 1806 frames
 2. claude           5.00  +6 frames
 3. gemini           0.82  out on frame 322
 4. codex            0.35  out on frame 154
"""

TRAINING_LOG = """2026-09-05 18:32:45,802 INFO gen   1 | claude  0.16   gemini  0.30
2026-09-05 18:33:06,268 INFO gen   2 | claude  0.21   gemini  0.38
2026-09-05 18:33:26,268 INFO gen   3 | claude  0.52   gemini  0.62
2026-09-05 18:33:26,300 INFO claude        0.52  ->  outputs/claude.npz
"""


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_race_log_times_the_winner_absolutely_and_the_rest_as_gaps(tmp_path):
    standings = parse_race_log(_write(tmp_path, "race.log", RACE_LOG))
    assert [s.name for s in standings] == ["paololimo", "claude", "gemini", "codex"]
    assert standings[0].finished_frames == 1806
    # 1812, not 6: the log gives the runner-up's gap, not its time.
    assert standings[1].finished_frames == 1812
    assert [s.finished for s in standings] == [True, True, False, False]
    assert standings[2].laps == 0.82


def test_race_target_comes_from_the_log_that_ran_it(tmp_path):
    assert race_target(_write(tmp_path, "race.log", RACE_LOG)) == 5.0
    assert race_target(_write(tmp_path, "empty.log", "nothing here"), fallback=3.0) == 3.0


def test_training_log_gives_every_entrant_and_the_time_between_generations(tmp_path):
    scores, seconds = parse_training_log(_write(tmp_path, "training.log", TRAINING_LOG))
    assert set(scores) == {"claude", "gemini"}
    assert list(scores["claude"]) == [0.16, 0.21, 0.52]
    # The final "-> outputs/claude.npz" line also holds a name and a number and
    # must not be read as a fourth generation.
    assert len(scores["gemini"]) == 3
    assert list(seconds) == [21.0, 20.0]


def test_entrants_are_read_back_out_of_the_checkpoints(tmp_path):
    np.savez(
        tmp_path / "codex.npz",
        genome=np.zeros(344),
        fitness=3.86,
        input_size=9,
        brain="codex",
        spec=json.dumps({}),
        seed=42,
        generations=120,
        population=100,
        history=np.array([0.1, 0.4, 0.3]),
    )
    (entrant,) = load_entrants(tmp_path)
    assert (entrant.name, entrant.parameters, entrant.best_fitness) == ("codex", 344, 3.86)
    # The running best only ever rises; the raw history does not.
    assert list(entrant.running_best) == [0.1, 0.4, 0.4]
    assert entrant.evaluations == 120 * 100 * 3


def test_ablation_rows_are_split_back_into_the_questions_they_answer(tmp_path):
    rows = [
        {"variant": name, "parameters": 366, "training": [1.0, 2.0], "race": [0.5, 1.5]}
        for name in ("uniform", "one-point", "two-point", "mutation only")
    ]
    (tmp_path / "ablation_crossover.json").write_text(json.dumps(rows))
    (group,) = load_ablations(tmp_path)
    assert group.key == "crossover"
    assert [row.variant for row in group.rows] == [r["variant"] for r in rows]
    assert group.rows[0].race_median == 1.0


def test_an_unreadable_ablation_file_is_skipped_rather_than_fatal(tmp_path):
    (tmp_path / "ablation_broken.json").write_text("{not json")
    assert load_ablations(tmp_path) == []


def _champion(path, **extra):
    np.savez(
        path,
        genome=np.zeros(150),
        fitness=1.0,
        input_size=9,
        brain="claude",
        spec=json.dumps({}),
        seed=42,
        generations=3,
        population=100,
        history=np.array([0.1, 0.4, 0.3]),
        **extra,
    )


def test_spread_and_per_circuit_survive_the_round_trip(tmp_path):
    _champion(
        tmp_path / "claude.npz",
        spread=np.array([0.8, 0.4, 0.2]),
        circuit_history=np.array([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5], [2.0, 3.0, 4.0]]),
    )
    (entrant,) = load_entrants(tmp_path)
    assert list(entrant.relative_spread) == [1.0, 0.5, 0.25]
    assert entrant.circuit_history.shape == (3, 3)

    lines = spread_csv([entrant]).splitlines()
    assert lines[0] == "generation,claude"
    assert lines[-1] == "3,0.250000"

    columns = circuits_csv([entrant], ["serpentine", "grid-city", "speedway"]).splitlines()[0]
    assert columns == "generation,claude_serpentine,claude_grid-city,claude_speedway"

    # The weakest circuit is starred, because it is the one leading the fitness.
    table = circuits_at([entrant], ["a", "b", "c"], [3])["markdown"]
    assert "2.00*" in table and "4.00 " in table


def test_a_checkpoint_written_before_the_fix_still_reports(tmp_path):
    """An old champion has neither array and must not take the report down."""
    _champion(tmp_path / "codex.npz")
    (entrant,) = load_entrants(tmp_path)
    assert entrant.spread is None and entrant.circuit_history is None
    assert entrant.relative_spread is None
    assert spread_csv([entrant]) == ""
    assert circuits_csv([entrant], ["a"]) == ""
    assert circuits_at([entrant], ["a"], [1]) == {}
