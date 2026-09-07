"""The figures, one function each, returning a `Figure` nobody has saved yet.

Separating "draw" from "write" is what lets the same function serve a vector PDF
for the report and a raster copy for slides without either format leaking into
the drawing. Nothing here reads a file; nothing here writes one.

Every figure is direct-labelled where it can be. A legend makes the reader hold
four colours in their head and look back and forth; a name at the end of its own
line does not, and on a chart of four entrants it costs no space.
"""

from typing import List, Optional, Sequence, Tuple

import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from matplotlib.ticker import NullFormatter, ScalarFormatter

from experiments.report.collect import AblationGroup, Entrant, Standing
from experiments.report.style import FAINT, GRID, INK, color_for

# One column of a two-column paper, and a shape that survives being dropped into
# a slide: everything is sized from these rather than tuned per figure.
WIDTH = 6.4
HEIGHT = 3.6


def _figure(width: float = WIDTH, height: float = HEIGHT) -> Tuple[Figure, "np.ndarray"]:
    figure = Figure(figsize=(width, height))
    return figure, figure.subplots()


def _label_line(axes, x: float, y: float, text: str, color: str) -> None:
    axes.annotate(
        text,
        (x, y),
        xytext=(5, 0),
        textcoords="offset points",
        color=color,
        fontsize=8,
        fontweight="bold",
        va="center",
        annotation_clip=False,
    )


def _spread_out(values: Sequence[float], gap: float) -> List[float]:
    """`values` nudged apart until no two are within `gap` of each other.

    Three entrants finished this run within 0.05 of each other, and three
    labels at the same height are one illegible smear. Each is moved the least
    distance that separates it, so the order still reads off the chart and no
    label sits nearer another curve than its own.
    """
    order = sorted(range(len(values)), key=lambda i: values[i])
    placed = list(values)
    for rank, index in enumerate(order[1:], start=1):
        below = placed[order[rank - 1]]
        placed[index] = max(placed[index], below + gap)
    return placed


# -- training ----------------------------------------------------------------


def training_curves(entrants: Sequence[Entrant]) -> Figure:
    """Best-so-far per entrant, with the raw per-generation best behind it.

    Two curves per entrant because they answer different questions. The solid
    envelope is progress: it can only rise, so where it flattens the search has
    stopped finding anything. The faint line is the draw: every generation gets
    a fresh start point and island layout, and how far it falls below the
    envelope is how much of any single generation's number is the circuit
    rather than the driver.
    """
    figure, axes = _figure()
    span = max(len(e.history) for e in entrants)
    for ordinal, entrant in enumerate(entrants):
        color = color_for(entrant.name, ordinal)
        generations = np.arange(1, len(entrant.history) + 1)
        axes.plot(generations, entrant.history, color=color, linewidth=0.6, alpha=0.30)
        axes.plot(generations, entrant.running_best, color=color, linewidth=1.6)

    peak = max(float(e.running_best[-1]) for e in entrants)
    heights = _spread_out([float(e.running_best[-1]) for e in entrants], peak * 0.055)
    for ordinal, (entrant, height) in enumerate(zip(entrants, heights, strict=True)):
        color = color_for(entrant.name, ordinal)
        _label_line(axes, len(entrant.history), height, entrant.name, color)

    axes.set_title("Training: best fitness so far")
    axes.set_xlabel("generation")
    axes.set_ylabel("fitness  (worst circuit + 0.35 x mean)")
    axes.set_xlim(1, span)
    axes.set_ylim(bottom=0)
    axes.margins(x=0.12)  # room for the labels at the right-hand end
    figure.tight_layout()
    return figure


def generation_cost(seconds: np.ndarray, window: int = 5) -> Figure:
    """Wall-clock per generation, which is what a run actually costs.

    It is not flat, and the shape is the search: while the population still
    crashes within a few frames a generation is cheap, and it triples in cost
    over the twenty generations in which the cars learn to survive, because an
    evaluation that no longer ends in a wall runs to the full budget. After
    that it is level — a plateau in cost, not in fitness. Reported as a rolling
    median because a single generation is hostage to whatever else the machine
    was doing at the time.
    """
    figure, axes = _figure(WIDTH, 2.6)
    generations = np.arange(2, len(seconds) + 2)
    axes.plot(generations, seconds, color=GRID, linewidth=0.8)
    if len(seconds) >= window:
        smoothed = np.array(
            [np.median(seconds[max(0, i - window + 1) : i + 1]) for i in range(len(seconds))]
        )
        axes.plot(generations, smoothed, color=INK, linewidth=1.4)
    total = seconds.sum() / 3600
    axes.set_title(f"Cost of the run: {total:.1f} hours over {len(seconds) + 1} generations")
    axes.set_xlabel("generation")
    axes.set_ylabel("seconds")
    axes.set_xlim(2, len(seconds) + 1)
    axes.set_ylim(bottom=0)
    figure.tight_layout()
    return figure


def genetic_spread(entrants: Sequence[Entrant]) -> Optional[Figure]:
    """Is the search still alive?

    A genetic algorithm can converge to one genome in a hundred copies and go
    on running for another ninety generations, improving nothing. This is the
    measurement that tells that apart from a hard problem: a flat fitness curve
    over a spread still well off the floor is a search that is still looking
    and not finding, while a flat curve over a spread at zero is a population
    with nothing left to recombine, where the remaining generations were spent
    before they were run.

    Each entrant is normalised to its own generation 1, because the genome
    sizes differ by more than a factor of two and the absolute spreads with
    them; what is comparable is the fraction of its own diversity each one has
    left.
    """
    tracked = [e for e in entrants if e.relative_spread is not None]
    if not tracked:
        return None
    figure, axes = _figure()
    for ordinal, entrant in enumerate(tracked):
        relative = entrant.relative_spread
        assert relative is not None
        color = color_for(entrant.name, ordinal)
        axes.plot(np.arange(1, len(relative) + 1), relative, color=color, linewidth=1.6)

    heights = _spread_out([float(e.relative_spread[-1]) for e in tracked], 0.045)
    for ordinal, (entrant, height) in enumerate(zip(tracked, heights, strict=True)):
        relative = entrant.relative_spread
        assert relative is not None
        _label_line(axes, len(relative), height, entrant.name, color_for(entrant.name, ordinal))

    axes.axhline(0.0, color=INK, linewidth=0.6)
    axes.set_title("Genetic spread, against each entrant's own generation 1")
    axes.set_xlabel("generation")
    axes.set_ylabel("population diversity remaining")
    axes.set_xlim(1, max(len(e.spread) for e in tracked if e.spread is not None))
    axes.set_ylim(bottom=0)
    axes.margins(x=0.12)
    figure.tight_layout()
    return figure


def per_circuit(entrants: Sequence[Entrant], circuits: Sequence[str]) -> Optional[Figure]:
    """What is holding each entrant back, circuit by circuit.

    Fitness is led by the worst of the three, so an entrant fast on two
    circuits and stuck on the third is a completely different problem from one
    uniformly slow — and the aggregate cannot tell them apart. One panel per
    circuit, shared axes, so a line that sits below its own level in the other
    panels names the circuit doing the holding.
    """
    tracked = [e for e in entrants if e.circuit_history is not None]
    if not tracked:
        return None
    count = min(min(e.circuit_history.shape[1] for e in tracked), len(circuits))
    figure = Figure(figsize=(WIDTH, 2.6))
    axes_row = figure.subplots(1, count, sharey=True)
    peak = max(float(np.max(e.circuit_history)) for e in tracked)
    for circuit in range(count):
        axes = axes_row[circuit] if count > 1 else axes_row
        for ordinal, entrant in enumerate(tracked):
            values = entrant.circuit_history[:, circuit]
            axes.plot(
                np.arange(1, len(values) + 1),
                np.maximum.accumulate(values),
                color=color_for(entrant.name, ordinal),
                linewidth=1.3,
            )
        axes.set_title(circuits[circuit], fontsize=8.5)
        axes.set_xlabel("generation")
        axes.set_ylim(0, peak * 1.05)
    first = axes_row[0] if count > 1 else axes_row
    first.set_ylabel("best score on this circuit")
    for ordinal, entrant in enumerate(tracked):
        values = entrant.circuit_history[:, count - 1]
        last = axes_row[count - 1] if count > 1 else axes_row
        _label_line(
            last, len(values), float(np.max(values)), entrant.name,
            color_for(entrant.name, ordinal),
        )
    figure.tight_layout()
    return figure


# -- the race ----------------------------------------------------------------


def race_standings(standings: Sequence[Standing], target: float) -> Figure:
    """Distance covered on the unseen circuit, in finishing order.

    Laps are the only quantity every entrant has: a finisher is timed and
    everyone else stops somewhere, so a chart of times would have to leave the
    ones that crashed off it entirely. The flag is drawn where the race ends,
    and the time gaps are written on the bars that reached it.
    """
    figure, axes = _figure(WIDTH, 0.55 * len(standings) + 1.4)
    winner = next((s.finished_frames for s in standings if s.finished), None)
    for ordinal, standing in enumerate(standings):
        color = color_for(standing.name, ordinal)
        y = len(standings) - ordinal
        faded = 1.0 if standing.finished else 0.45
        axes.barh(y, standing.laps, height=0.55, color=color, alpha=faded)
        note = standing.note
        if standing.finished and winner is not None:
            gap = standing.finished_frames - winner
            note = "winner" if gap == 0 else f"+{gap} frames"
        axes.annotate(
            note,
            (standing.laps, y),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=INK if standing.finished else FAINT,
        )

    axes.axvline(target, color=INK, linewidth=0.8, linestyle=(0, (4, 2)))
    axes.annotate(
        f"flag — {target:.0f} laps",
        (target, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(4, -9),
        textcoords="offset points",
        fontsize=8,
        color=INK,
    )
    axes.set_yticks(range(1, len(standings) + 1))
    axes.set_yticklabels([f"{s.rank}. {s.name}" for s in reversed(standings)])
    axes.set_xlabel("laps completed")
    axes.set_title("The race, on a circuit none of them trained on")
    axes.set_xlim(0, target * 1.35)
    axes.grid(axis="y", visible=False)
    figure.tight_layout()
    return figure


# -- ablations ---------------------------------------------------------------


def ablation_dots(group: AblationGroup) -> Figure:
    """Every seed as a dot, the median as a bar. Never the median alone.

    The same architecture has scored 0.04 and 3.05 laps with nothing but the
    seed changed. A bar chart of medians over that spread invites a conclusion
    the data cannot support, so the seeds are plotted and the median is drawn
    on top of them: where two variants' clouds overlap, the difference between
    them has not been measured, and the figure says so without a caption.
    """
    rows = group.rows
    figure, axes = _figure(WIDTH, 0.52 * len(rows) + 1.5)
    for ordinal, row in enumerate(rows):
        y = len(rows) - ordinal
        axes.plot(
            [min(row.race), max(row.race)], [y, y],
            color=GRID, linewidth=6, solid_capstyle="round",
        )
        # One colour for every variant. The variant is already named on the
        # axis, so colouring the rows as well would be a second encoding of
        # something the reader can read, and would imply the four crossover
        # operators belong to the same scale as the four entrants elsewhere.
        axes.plot(row.race, [y] * len(row.race), "o", color=FAINT, markersize=4.5, alpha=0.8)
        axes.plot(row.race_median, y, "|", color=INK, markersize=16, markeredgewidth=1.6)
        axes.annotate(
            f"{row.race_median:.2f}",
            (max(row.race), y),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            fontweight="bold",
        )

    axes.set_yticks(range(1, len(rows) + 1))
    axes.set_yticklabels(
        [f"{row.variant}  ({row.parameters}p)" for row in reversed(rows)]
    )
    axes.set_xlabel("laps on the unseen circuit  (one dot per seed, bar = median)")
    axes.set_title(group.title)
    axes.set_xlim(left=0)
    axes.margins(x=0.14)
    axes.grid(axis="y", visible=False)
    figure.tight_layout()
    return figure


def parameters_against_race(groups: Sequence[AblationGroup]) -> Optional[Figure]:
    """Does paying for parameters buy laps?

    The whole size ladder on one log axis, seeds and all. Plotted only when a
    size or depth group is present, because the question is meaningless for the
    ones that hold the parameter count fixed and vary something else.
    """
    rows = [row for group in groups if group.key in ("sizes", "depths") for row in group.rows]
    if not rows:
        return None
    rows = sorted(rows, key=lambda r: r.parameters)
    figure, axes = _figure(WIDTH, 3.2)
    axes.plot(
        [row.parameters for row in rows],
        [row.race_median for row in rows],
        color=GRID,
        linewidth=1.2,
        zorder=1,
    )
    for row in rows:
        axes.plot(
            [row.parameters] * len(row.race), row.race,
            "o", color=FAINT, markersize=4, alpha=0.55, zorder=2,
        )
        axes.plot(row.parameters, row.race_median, "D", color=INK, markersize=5, zorder=3)
        _label_line(axes, row.parameters, row.race_median, row.variant, INK)

    axes.set_xscale("log")
    # Log ticks at the parameter counts themselves: `2 x 10^1` is a spelling of
    # 18 that no reader of this chart wants.
    axes.set_xticks([row.parameters for row in rows])
    axes.xaxis.set_major_formatter(ScalarFormatter())
    axes.xaxis.set_minor_formatter(NullFormatter())
    axes.set_title("Parameters against performance")
    axes.set_xlabel("parameters (log scale, one dot per seed)")
    axes.set_ylabel("laps on the unseen circuit")
    axes.set_ylim(bottom=0)
    axes.margins(x=0.18)
    figure.tight_layout()
    return figure


# -- the circuits ------------------------------------------------------------


def track_maps(tracks: Sequence["object"], titles: Sequence[str]) -> Figure:
    """The circuits themselves, as geometry rather than as a screenshot.

    Drawn the way the simulation draws them — a disc at every centreline
    sample, sized by the local road width — so this is the track the cars
    actually drive, at any size a page or a projector asks for. Offsetting the
    centreline into two edges and filling between them is the obvious
    alternative and it is wrong: at a hairpin the inner edge crosses itself and
    the fill grows a spike through the middle of the corner.
    """
    columns = len(tracks)
    figure = Figure(figsize=(3.0 * columns, 2.4))
    for column, (track, title) in enumerate(zip(tracks, titles, strict=True)):
        axes = figure.add_subplot(1, columns, column + 1)
        axes.add_collection(_carriageway(track))
        for island in _islands(track):
            axes.fill(island[:, 0], island[:, 1], color="#ffffff", edgecolor=FAINT, linewidth=0.4)
        axes.plot(
            track.samples[:, 0], track.samples[:, 1],
            color=FAINT, linewidth=0.4, linestyle=(0, (3, 3)),
        )
        axes.set_title(f"{title}\n{track.lap_length:.0f} px lap", fontsize=8.5)
        axes.set_xlim(0, track.size[0])
        axes.set_ylim(track.size[1], 0)  # screen coordinates: y grows downwards
        axes.set_aspect("equal")
        axes.axis("off")
    figure.tight_layout()
    return figure


def _carriageway(track) -> PatchCollection:
    """The road surface, as overlapping discs along the centreline.

    Sampled every half-radius rather than at every sample: the samples are 4 px
    apart and the discs are some 40 px across, so all but one in ten are drawn
    entirely inside their neighbours and cost the PDF a path each for nothing.
    """
    step = max(1, int(track.cfg.road_width / 4 / track.cfg.sample_spacing))
    discs = [
        Circle(track.samples[index], track.road_width_at_index(index) / 2)
        for index in range(0, len(track.samples), step)
    ]
    return PatchCollection(discs, facecolor="#e3e6ee", edgecolor="none")


def _islands(track) -> List[np.ndarray]:
    """The islands sitting in the carriageway, if the circuit has any.

    Built by the track itself rather than re-derived here: the shape depends on
    the local road width and the heading, and a second implementation of that
    would eventually disagree with the one the cars actually crash into.
    """
    shapes = []
    for ordinal, fraction in enumerate(track.cfg.islands):
        shape = track._island_shape(track.index_at_fraction(fraction), ordinal)
        if shape is not None:
            shapes.append(np.array(shape))
    return shapes


__all__ = [
    "ablation_dots",
    "generation_cost",
    "genetic_spread",
    "parameters_against_race",
    "per_circuit",
    "race_standings",
    "track_maps",
    "training_curves",
]
