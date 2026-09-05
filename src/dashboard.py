"""Side panel: who is ahead, what each brain is doing, and how it has improved.

The layout is derived from how many entrants there are, so adding a file to
`src/brains/` adds a card and a curve with no change here.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pygame

from src.brains import Brain, Color

_BG = (18, 20, 32)
_CARD = (28, 31, 48)
_TEXT = (232, 234, 245)
_MUTED = (140, 146, 170)
_ACCENT = (255, 176, 60)
_POSITIVE = (110, 220, 150)
_NEGATIVE = (235, 100, 110)

# Sensors are probed around mid-range: half-open road, half speed. Reading the
# response at the extremes would mostly show tanh saturation rather than what
# the brain does while driving.
_PROBE_LEVEL = 0.5
_PROBE_STEP = 0.05


@dataclass(frozen=True)
class Entry:
    """One entrant's line in the panel."""

    name: str
    color: Color
    laps: float
    alive: int
    shown: int
    best: float
    brain: Optional[Brain] = None


Curve = Tuple[str, Color, Sequence[float]]


def response(brain: Brain, input_size: int) -> np.ndarray:
    """How much each input moves each output: an `input_size` x 2 array.

    Measured rather than read off the weights, because the weights are the one
    thing an entrant is not obliged to expose — the contract stops at `forward`.
    That makes this the only view of a brain that works for every architecture,
    and the only one that compares across them: two brains with nothing in
    common structurally can still be asked the same question.
    """
    base = np.full(input_size, _PROBE_LEVEL)
    at_rest = np.asarray(brain.forward(base.copy()), dtype=float)
    jacobian = np.zeros((input_size, 2))
    for i in range(input_size):
        probe = base.copy()
        probe[i] += _PROBE_STEP
        jacobian[i] = (np.asarray(brain.forward(probe), dtype=float) - at_rest) / _PROBE_STEP
    return jacobian


class Dashboard:
    """Renders the right-hand panel onto its own surface."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.surface = pygame.Surface((width, height))
        self.font_big = pygame.font.SysFont("menlo,monospace", 26, bold=True)
        self.font = pygame.font.SysFont("menlo,monospace", 14)
        self.font_small = pygame.font.SysFont("menlo,monospace", 11)
        # Diagrams are redrawn only when the brain behind them changes, which is
        # when a squad's leader changes — a few times per evaluation, not sixty
        # times a second.
        self._diagrams: Dict[int, pygame.Surface] = {}

    def _text(self, text: str, x: int, y: int, font: pygame.font.Font, color=_TEXT) -> None:
        self.surface.blit(font.render(text, True, color), (x, y))

    def _card(self, y: int, height: int, title: str) -> int:
        """Draw a titled card and return the y where its content starts."""
        rect = pygame.Rect(12, y, self.width - 24, height)
        pygame.draw.rect(self.surface, _CARD, rect, border_radius=8)
        self._text(title, 24, y + 8, self.font_small, _MUTED)
        return y + 26

    def _header(
        self, generation: int, track: str, number: int, count: int, frame: int, total: int
    ) -> int:
        self._text(f"GEN {generation}" if generation else "RACE", 16, 14, self.font_big)
        self._text(f"TRACK {number}/{count}", self.width - 150, 16, self.font, _MUTED)
        self._text(f"STEP {frame}/{total}", self.width - 150, 34, self.font, _MUTED)
        self._text(track.upper(), 16, 46, self.font, _ACCENT)

        bar = pygame.Rect(16, 68, self.width - 32, 4)
        pygame.draw.rect(self.surface, _CARD, bar, border_radius=2)
        filled = pygame.Rect(16, 68, int((self.width - 32) * frame / max(1, total)), 4)
        pygame.draw.rect(self.surface, _ACCENT, filled, border_radius=2)
        return 82

    def _topology(self, topology, size: Tuple[int, int]) -> pygame.Surface:
        """An entrant's real structure: its layers, its nodes, its weights.

        Green edges are positive weights, red negative, intensity proportional
        to magnitude — and only the strongest are drawn, or the card turns into
        a blob. Layers sharing a column are parallel streams; an edge spanning
        more than one column is a skip.
        """
        surface = pygame.Surface(size)
        surface.fill(_CARD)
        width, height = size
        margin_x, margin_y = 12, 9
        columns = max(column for _, _, column in topology.layers) or 1
        span = width - 2 * margin_x - 8

        # Where every node sits. Layers sharing a column are stacked in bands so
        # two parallel streams stay visually separate.
        per_column: Dict[int, List[int]] = {}
        for index, (_, _, column) in enumerate(topology.layers):
            per_column.setdefault(column, []).append(index)

        positions: List[List[Tuple[float, float]]] = [[] for _ in topology.layers]
        for column, members in per_column.items():
            x = margin_x + span * column / columns
            band = (height - 2 * margin_y) / len(members)
            for slot, index in enumerate(members):
                count = topology.layers[index][1]
                top = margin_y + slot * band
                usable = band - (6 if len(members) > 1 else 0)
                positions[index] = [
                    (x, top + (usable / 2 if count == 1 else usable * n / (count - 1)))
                    for n in range(count)
                ]

        scale = max(
            (float(np.abs(np.asarray(w)).max()) for _, _, w in topology.edges if np.size(w)),
            default=1.0,
        ) or 1.0
        for source, target, weights in topology.edges:
            weights = np.asarray(weights, dtype=float)
            if not weights.size:
                continue
            # Only the strongest tenth: every edge at once is an unreadable mat.
            threshold = np.percentile(np.abs(weights), 90)
            for i, j in zip(*np.where(np.abs(weights) >= threshold)):
                strength = abs(weights[i, j]) / scale
                base = _POSITIVE if weights[i, j] > 0 else _NEGATIVE
                tint = tuple(int(_CARD[k] + (base[k] - _CARD[k]) * strength) for k in range(3))
                pygame.draw.line(surface, tint, positions[source][i], positions[target][j], 1)

        for index, (_, _, column) in enumerate(topology.layers):
            color = _ACCENT if column == 0 else _TEXT if column == columns else _MUTED
            for x, y in positions[index]:
                pygame.draw.circle(surface, color, (int(x), int(y)), 2)
        return surface

    def _response_diagram(self, brain: Brain, size: Tuple[int, int]) -> pygame.Surface:
        """The fallback for a brain that does not describe itself.

        Eight sensors on the left, steering and throttle on the right, an edge
        green where raising that input raises that output and red where it
        lowers it. Measured, not read off the weights, so it works for any
        architecture — including one whose weights mean nothing on their own.
        """
        surface = pygame.Surface(size)
        surface.fill(_CARD)
        width, height = size
        jacobian = response(brain, 8)
        scale = max(1e-6, float(np.abs(jacobian).max()))

        left, right = 14, width - 22
        rays, gap = 7, 6
        ray_ys = [gap + (height - 2 * gap - 14) * i / (rays - 1) for i in range(rays)]
        speed_y = height - gap - 4
        input_ys = ray_ys + [speed_y]
        output_ys = [height * 0.34, height * 0.66]

        for i, y_in in enumerate(input_ys):
            for j, y_out in enumerate(output_ys):
                strength = abs(jacobian[i, j]) / scale
                if strength < 0.12:  # everything faint at once is just noise
                    continue
                base = _POSITIVE if jacobian[i, j] > 0 else _NEGATIVE
                tint = tuple(int(_CARD[k] + (base[k] - _CARD[k]) * strength) for k in range(3))
                pygame.draw.line(surface, tint, (left, y_in), (right, y_out), 1)

        for i, y in enumerate(input_ys):
            color = _ACCENT if i < rays else _MUTED
            pygame.draw.circle(surface, color, (left, int(y)), 3)
        for y, label in zip(output_ys, ("S", "T")):
            pygame.draw.circle(surface, _TEXT, (right, int(y)), 3)
            surface.blit(self.font_small.render(label, True, _MUTED), (right + 6, int(y) - 6))
        return surface

    def _cached_diagram(self, brain: Brain, size: Tuple[int, int]) -> pygame.Surface:
        """Draw the brain's own structure if it offers one, its response if not."""
        key = id(brain)
        cached = self._diagrams.get(key)
        if cached is None or cached.get_size() != size:
            if len(self._diagrams) > 64:  # leaders come and go; do not hoard them
                self._diagrams.clear()
            describe = getattr(brain, "describe", None)
            topology = describe() if callable(describe) else None
            cached = (
                self._topology(topology, size)
                if topology is not None and topology.layers
                else self._response_diagram(brain, size)
            )
            self._diagrams[key] = cached
        return cached

    def _entries(self, y: int, entries: Sequence[Entry], card_height: int) -> int:
        """One card per entrant: standing on the left, its brain on the right."""
        for entry in sorted(entries, key=lambda e: -e.laps):
            top = self._card(y, card_height, "")
            pygame.draw.rect(
                self.surface, entry.color, pygame.Rect(24, y + 9, 10, 10), border_radius=2
            )
            self._text(entry.name[:12], 42, y + 6, self.font)
            self._text(f"{entry.laps:5.2f} laps", 42, top + 2, self.font, _ACCENT)
            detail = f"best {entry.best:.2f}"
            if entry.shown:
                detail += f"   {entry.alive}/{entry.shown} alive"
            self._text(detail, 42, top + 20, self.font_small, _MUTED)

            if entry.brain is not None:
                inner = (150, card_height - 16)
                self.surface.blit(
                    self._cached_diagram(entry.brain, inner),
                    (self.width - inner[0] - 24, y + 8),
                )
            y += card_height + 8
        return y

    def _chart(self, y: int, curves: Sequence[Curve]) -> None:
        """Best fitness per generation, one line per entrant in its colour."""
        height = self.height - y - 14
        if height < 60:
            return
        top = self._card(y, height, "BEST FITNESS / GENERATION")
        longest = max((len(history) for _, _, history in curves), default=0)
        if longest < 2:
            self._text("collecting...", 28, top + 14, self.font_small, _MUTED)
            return

        plot = pygame.Rect(28, top + 4, self.width - 56, height - 46)
        peak = max((max(h) for _, _, h in curves if h), default=1.0) or 1.0
        for _, color, history in curves:
            if len(history) < 2:
                continue
            points: List[Tuple[float, float]] = [
                (
                    plot.x + plot.width * i / (longest - 1),
                    plot.bottom - plot.height * (value / peak),
                )
                for i, value in enumerate(history)
            ]
            pygame.draw.lines(self.surface, color, False, points, 2)
            pygame.draw.circle(self.surface, color, points[-1], 3)
        self._text(f"{peak:.2f}", self.width - 96, top - 18, self.font_small, _MUTED)

    def _card_height(self, count: int) -> int:
        """Share the space between the cards and the chart.

        With a few entrants each card is tall enough for a legible diagram; with
        many, the cards give way first and the chart keeps a floor, because a
        cramped diagram says less than a cramped curve.
        """
        available = self.height - 82 - 150  # header, then the chart's floor
        return int(np.clip(available / max(1, count) - 8, 46, 92))

    def render(
        self,
        generation: int,
        track_name: str,
        track_number: int,
        track_count: int,
        frame: int,
        max_frames: int,
        entries: Sequence[Entry],
        curves: Sequence[Curve],
    ) -> pygame.Surface:
        self.surface.fill(_BG)
        y = self._header(generation, track_name, track_number, track_count, frame, max_frames)
        y = self._entries(y + 2, entries, self._card_height(len(entries)))
        self._chart(y + 2, curves)
        return self.surface


__all__ = ["Curve", "Dashboard", "Entry", "response"]
