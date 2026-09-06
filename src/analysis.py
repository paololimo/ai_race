"""The strip under the circuit: what the search is actually doing.

Three questions a neuroevolution run cannot be read without, side by side.

*Is anyone still improving?* The best score of a generation is not a progress
curve — every generation draws a fresh start point and a different island
layout, so even an elite carried over untouched scores differently from one to
the next, and all the entrants rise and fall together with the draw. The
running maximum is monotonic and answers the question; the raw per-generation
score is kept faintly behind it, because the differences between those faint
lines are where one entrant handles a hard draw better than another.

*Is the search still alive?* A genetic algorithm can converge to one genome in
a hundred copies and go on running for another ninety generations, improving
nothing. Genetic spread — the mean per-gene standard deviation of the
population — is what shows that, and normalising each entrant to its own first
generation makes different genome sizes comparable on one axis.

*What is holding each entrant back?* Fitness across circuits is
`worst + 0.35 x mean`, so the lowest of the three is what leads the number, and
it was being aggregated away before anyone could see it. An entrant fast on two
circuits and stuck on the third is a different problem from one uniformly slow,
and a single fitness number cannot tell them apart.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pygame

from src.brains import Color
from src.theme import BG, CARD, MUTED, TEXT, tint


@dataclass(frozen=True)
class Series:
    """One entrant's history, for the analyses."""

    name: str
    color: Color
    # Best fitness in each generation. Not a progress curve on its own: every
    # generation draws a fresh start point and a different island layout, so
    # even an untouched elite scores differently from one to the next.
    best: Sequence[float]
    # Mean per-gene standard deviation of the population, per generation.
    spread: Sequence[float]
    # Best score on each circuit, this generation.
    circuits: Sequence[float]


class Analysis:
    """Renders the analysis strip onto its own surface."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.surface = pygame.Surface((width, height))
        self.font = pygame.font.SysFont("menlo,monospace", 12)
        self.font_small = pygame.font.SysFont("menlo,monospace", 11)

    # -- primitives ----------------------------------------------------------

    def _text(self, text: str, x: int, y: int, color=TEXT, small: bool = True) -> None:
        font = self.font_small if small else self.font
        self.surface.blit(font.render(text, True, color), (x, y))

    def _panel(self, x: int, width: int, title: str, note: str) -> pygame.Rect:
        """Draw a titled panel and return the rectangle its plot may use."""
        rect = pygame.Rect(x, 10, width, self.height - 20)
        pygame.draw.rect(self.surface, CARD, rect, border_radius=8)
        self._text(title, x + 14, 20, MUTED)
        self._text(note, x + 14, rect.bottom - 20, MUTED)
        return pygame.Rect(x + 16, 40, width - 32, rect.height - 66)

    def _line(
        self,
        plot: pygame.Rect,
        values: Sequence[float],
        span: int,
        peak: float,
        color: Color,
        thickness: int,
    ) -> List[Tuple[float, float]]:
        points = [
            (
                plot.x + plot.width * i / max(1, span - 1),
                plot.bottom - plot.height * min(1.0, float(v) / peak),
            )
            for i, v in enumerate(values)
        ]
        if len(points) > 1:
            pygame.draw.lines(self.surface, color, False, points, thickness)
        return points

    def _waiting(self, plot: pygame.Rect) -> None:
        self._text("collecting...", plot.x, plot.y + 8, MUTED)

    # -- the three questions -------------------------------------------------

    def _progress(self, x: int, width: int, series: Sequence[Series]) -> None:
        plot = self._panel(x, width, "BEST SO FAR", "faint: that generation's best")
        tracked = [s for s in series if len(s.best) >= 2]
        span = max((len(s.best) for s in tracked), default=0)
        if span < 2:
            return self._waiting(plot)

        running = [np.maximum.accumulate(np.asarray(s.best, dtype=float)) for s in tracked]
        peak = max(float(r[-1]) for r in running) or 1.0
        for s in tracked:  # the raw scores first, so the envelopes sit on top
            self._line(plot, s.best, span, peak, tint(s.color, 0.38), 1)
        for s, values in zip(tracked, running, strict=True):
            end = self._line(plot, values, span, peak, s.color, 2)[-1]
            pygame.draw.circle(self.surface, s.color, end, 3)
        self._text(f"{peak:.2f}", plot.right - 30, 20, MUTED)
        self._text(f"gen {span}", plot.right - 52, plot.bottom + 4, MUTED)

    def _spread(self, x: int, width: int, series: Sequence[Series]) -> None:
        plot = self._panel(x, width, "GENETIC SPREAD", "1.0 = generation 1; 0 = converged")
        tracked = [s for s in series if len(s.spread) >= 2 and s.spread[0] > 0]
        span = max((len(s.spread) for s in tracked), default=0)
        if span < 2:
            return self._waiting(plot)

        pygame.draw.line(self.surface, BG, (plot.x, plot.bottom), (plot.right, plot.bottom), 1)
        floor = 0.0
        for s in tracked:
            relative = np.asarray(s.spread, dtype=float) / float(s.spread[0])
            pygame.draw.circle(
                self.surface, s.color, self._line(plot, relative, span, 1.0, s.color, 2)[-1], 3
            )
            floor = max(floor, float(relative[-1]))
        # One number for where the lines have got to, not one per line: they
        # converge, and four labels on top of each other is worse than none.
        self._text("1.00", plot.right - 30, 20, MUTED)
        self._text(f"now {floor:.2f}", plot.right - 56, plot.bottom + 4, MUTED)

    def _circuits(
        self, x: int, width: int, series: Sequence[Series], names: Sequence[str]
    ) -> None:
        plot = self._panel(x, width, "BEST PER CIRCUIT", "solid: that entrant's weakest")
        tracked = [s for s in series if len(s.circuits) >= 1]
        if not tracked:
            return self._waiting(plot)

        peak = max(max(s.circuits) for s in tracked) or 1.0
        count = max(len(s.circuits) for s in tracked)
        # Grouped by circuit, not by entrant: the useful comparison is who is
        # struggling *here*, and the colours name the entrants without a label.
        group = plot.height / max(1, count)
        left, span = plot.x + 6, plot.width - 6 - 44

        for i in range(count):
            top = plot.y + group * i
            self._text(names[i] if i < len(names) else f"circuit {i + 1}", plot.x, int(top), TEXT)
            for slot, s in enumerate(tracked):
                if i >= len(s.circuits):
                    continue
                value, bar_y = s.circuits[i], int(top + 15 + slot * 10)
                if bar_y + 7 > plot.bottom:
                    break
                pygame.draw.rect(self.surface, BG, (left, bar_y, span, 7), border_radius=3)
                # An entrant's worst circuit is the one that leads its fitness,
                # so that bar is solid and the rest of its bars are faded.
                weakest = value <= min(s.circuits) + 1e-9
                pygame.draw.rect(
                    self.surface,
                    s.color if weakest else tint(s.color, 0.4),
                    (left, bar_y, max(2, int(span * float(value) / peak)), 7),
                    border_radius=3,
                )
                self._text(f"{value:.2f}", left + span + 6, bar_y - 2, MUTED)

    # -- assembly ------------------------------------------------------------

    def _run(self, x: int, width: int, stats: Sequence[Tuple[str, str]]) -> None:
        """Labelled numbers about the run itself.

        Not everything worth watching is a curve. The mutation size is one
        number, and since it anneals it is a different one every generation;
        the evaluation count is the quantity the whole argument about how many
        parameters an entrant can afford turns on. Neither appears in any chart
        here, and both are one line of text.
        """
        plot = self._panel(x, width, "RUN", "")
        for i, (label, value) in enumerate(stats):
            row = plot.y + i * 17
            if row + 12 > plot.bottom:
                return
            self._text(label, plot.x, row, MUTED)
            self._text(value, plot.x + 96, row, TEXT)

    def _order(
        self, x: int, width: int, order: Sequence[Tuple[str, Color, float, bool]], target: float
    ) -> None:
        """The running order of a race, against the distance that ends it.

        A race has no history to plot — the three training charts would all say
        "collecting..." for its whole twenty seconds — and it has something
        better to show: who is where, and how far from the flag.
        """
        plot = self._panel(x, width, "RACE ORDER", f"first to {target:.0f} laps")
        row = plot.height / max(1, len(order))
        left, span = plot.x + 132, plot.width - 132 - 54
        for i, (name, color, laps, alive) in enumerate(order):
            top = int(plot.y + row * i + row / 2 - 9)
            self._text(f"{i + 1}.", plot.x, top + 2, MUTED)
            self._text(name[:12], plot.x + 22, top + 2, color)
            pygame.draw.rect(self.surface, BG, (left, top + 4, span, 9), border_radius=4)
            pygame.draw.rect(
                self.surface,
                color if alive else tint(color, 0.35),
                (left, top + 4, max(2, int(span * min(1.0, laps / target))), 9),
                border_radius=4,
            )
            self._text(f"{laps:.2f}" if alive else f"{laps:.2f} out",
                       left + span + 8, top + 1, MUTED if alive else tint(color, 0.7))

    def render(
        self,
        series: Sequence[Series],
        circuits: Sequence[str],
        stats: Sequence[Tuple[str, str]] = (),
        order: Optional[Sequence[Tuple[str, Color, float, bool]]] = None,
        target: float = 0.0,
    ) -> pygame.Surface:
        self.surface.fill(BG)
        gap = 10
        if order is not None:
            run = 210 if stats else 0
            self._order(gap, self.width - gap * (2 if not stats else 3) - run, order, target)
            if stats:
                self._run(self.width - gap - run, run, stats)
            return self.surface
        # The run stats are a narrow column of text; the three charts share the
        # rest of the width equally.
        # Folded into one branch: with no stats column the width its gap would
        # have taken goes back to the charts, which is what `-gap` says.
        run = 210 if stats else -gap
        width = (self.width - gap * 5 - run) // 3
        self._progress(gap, width, series)
        self._spread(gap * 2 + width, width, series)
        self._circuits(gap * 3 + width * 2, width, series, circuits)
        if stats:
            self._run(gap * 4 + width * 3, run, stats)
        return self.surface


__all__ = ["Analysis", "Series"]
