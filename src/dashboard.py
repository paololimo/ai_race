"""Side panel: who is ahead, what each brain is, and where the run has got to.

The layout is derived from how many entrants there are, so adding a file to
`src/brains/` adds a card with no change here. The analyses live on the strip
under the circuit — see `analysis.py` — which is what leaves these cards tall
enough to draw a brain in.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pygame

from src.brains import Brain, Color
from src.theme import ACCENT, BG, CARD, MUTED, NEGATIVE, POSITIVE, TEXT, tint

logger = logging.getLogger(__name__)

# Where the brain diagram starts, leaving the three rows of text their width.
_TEXT_COLUMN = 194

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
    params: int = 0
    brain: Optional[Brain] = None


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

    def __init__(self, width: int, height: int, input_size: int) -> None:
        self.width = width
        self.height = height
        # What the cars actually feed their brains, so the fallback diagram
        # keeps probing the right number of inputs if the sensor layout changes.
        self.input_size = input_size
        self.surface = pygame.Surface((width, height))
        self.font_big = pygame.font.SysFont("menlo,monospace", 26, bold=True)
        self.font = pygame.font.SysFont("menlo,monospace", 14)
        self.font_small = pygame.font.SysFont("menlo,monospace", 11)
        # Diagrams are redrawn only when the brain behind them changes, which is
        # when a squad's leader changes — a few times per evaluation, not sixty
        # times a second.
        self._diagrams: Dict[int, pygame.Surface] = {}
        self._broken: Set[str] = set()  # named once, not once a frame

    # -- primitives ----------------------------------------------------------

    def _text(self, text: str, x: int, y: int, font: pygame.font.Font, color=TEXT) -> None:
        self.surface.blit(font.render(text, True, color), (x, y))

    def _card(self, x: int, y: int, width: int, height: int, title: str = "") -> int:
        """Draw a titled card and return the y where its content starts."""
        pygame.draw.rect(
            self.surface, CARD, pygame.Rect(x, y, width, height), border_radius=8
        )
        if title:
            self._text(title, x + 12, y + 8, self.font_small, MUTED)
        return y + 26

    def _header(
        self, generation: int, track: str, number: int, count: int, frame: int, total: int
    ) -> int:
        self._text(f"GEN {generation}" if generation else "RACE", 16, 14, self.font_big)
        self._text(f"TRACK {number}/{count}", self.width - 150, 16, self.font, MUTED)
        self._text(f"STEP {frame}/{total}", self.width - 150, 34, self.font, MUTED)
        self._text(track.upper(), 16, 46, self.font, ACCENT)

        bar = pygame.Rect(16, 68, self.width - 32, 4)
        pygame.draw.rect(self.surface, CARD, bar, border_radius=2)
        filled = pygame.Rect(16, 68, int((self.width - 32) * frame / max(1, total)), 4)
        pygame.draw.rect(self.surface, ACCENT, filled, border_radius=2)
        return 82

    # -- one entrant's brain -------------------------------------------------

    def _topology(self, topology, size: Tuple[int, int]) -> pygame.Surface:
        """An entrant's real structure: its layers, its nodes, its weights.

        Green edges are positive weights, red negative, intensity proportional
        to magnitude — and only the strongest are drawn, or the card turns into
        a blob. Layers sharing a column are parallel streams; an edge spanning
        more than one column is a skip.
        """
        surface = pygame.Surface(size)
        surface.fill(CARD)
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
                base = POSITIVE if weights[i, j] > 0 else NEGATIVE
                pygame.draw.line(
                    surface, tint(base, strength), positions[source][i], positions[target][j], 1
                )

        for index, (_, _, column) in enumerate(topology.layers):
            color = ACCENT if column == 0 else TEXT if column == columns else MUTED
            for x, y in positions[index]:
                pygame.draw.circle(surface, color, (int(x), int(y)), 2)
        return surface

    def _response_diagram(self, brain: Brain, size: Tuple[int, int]) -> pygame.Surface:
        """The fallback for a brain that does not describe itself.

        The sensor rays on the left, steering and throttle on the right, an edge
        green where raising that input raises that output and red where it
        lowers it. Measured, not read off the weights, so it works for any
        architecture — including one whose weights mean nothing on their own.
        """
        surface = pygame.Surface(size)
        surface.fill(CARD)
        width, height = size
        jacobian = response(brain, self.input_size)
        scale = max(1e-6, float(np.abs(jacobian).max()))

        left, right = 14, width - 22
        rays, gap = self.input_size - 1, 6  # every input but the last is a ray
        ray_ys = [gap + (height - 2 * gap - 14) * i / (rays - 1) for i in range(rays)]
        input_ys = ray_ys + [height - gap - 4]
        output_ys = [height * 0.34, height * 0.66]

        for i, y_in in enumerate(input_ys):
            for j, y_out in enumerate(output_ys):
                strength = abs(jacobian[i, j]) / scale
                if strength < 0.12:  # everything faint at once is just noise
                    continue
                base = POSITIVE if jacobian[i, j] > 0 else NEGATIVE
                pygame.draw.line(surface, tint(base, strength), (left, y_in), (right, y_out), 1)

        for i, y in enumerate(input_ys):
            pygame.draw.circle(surface, ACCENT if i < rays else MUTED, (left, int(y)), 3)
        for y, label in zip(output_ys, ("S", "T")):
            pygame.draw.circle(surface, TEXT, (right, int(y)), 3)
            surface.blit(self.font_small.render(label, True, MUTED), (right + 6, int(y) - 6))
        return surface

    def _draw_brain(self, brain: Brain, size: Tuple[int, int]) -> pygame.Surface:
        """This brain's structure if it offers one, its measured response if not.

        Both calls run competitor code — `describe()` is written by the entrant,
        and `forward` is too — so both are guarded. The registry already refuses
        to let a file that raises on import take the race down with it; a panel
        that takes the window down on the frame that entrant happens to lead
        would be the same failure arriving later. A brain that throws here is
        drawn as a blank card and named in the log, once.
        """
        try:
            describe = getattr(brain, "describe", None)
            topology = describe() if callable(describe) else None
            if topology is not None and topology.layers:
                return self._topology(topology, size)
            return self._response_diagram(brain, size)
        except Exception as exc:  # noqa: BLE001 - untrusted entrant code
            name = type(brain).__name__
            if name not in self._broken:
                self._broken.add(name)
                logger.error("%s cannot be drawn: %s", name, exc)
            surface = pygame.Surface(size)
            surface.fill(CARD)
            surface.blit(self.font_small.render("cannot be drawn", True, MUTED), (12, 10))
            return surface

    def _cached_diagram(self, brain: Brain, size: Tuple[int, int]) -> pygame.Surface:
        key = id(brain)
        cached = self._diagrams.get(key)
        if cached is None or cached.get_size() != size:
            if len(self._diagrams) > 64:  # leaders come and go; do not hoard them
                self._diagrams.clear()
            cached = self._draw_brain(brain, size)
            self._diagrams[key] = cached
        return cached

    # -- the standings -------------------------------------------------------

    def _card_height(self, room: int, count: int) -> int:
        """Share the space between the cards, whatever is left after the stats."""
        return int(np.clip(room / max(1, count) - 8, 46, 190))

    def _entries(self, y: int, entries: Sequence[Entry], card_height: int) -> None:
        """One card per entrant: the standing on the left, its brain beside it.

        Beside, not underneath. The nodes of a network are laid out down the
        card, so what a diagram needs is height, and beside the text it gets the
        whole card instead of whatever three rows of text leave over. Under a
        short card — and beside the circuit the panel is only as tall as the
        circuit is drawn — the diagram came out 40 px tall and flat.

        That costs width, so the text is three tight rows and the parameter
        count shares a line with the laps rather than taking one of its own.
        """
        for entry in sorted(entries, key=lambda e: -e.laps):
            top = self._card(12, y, self.width - 24, card_height)
            pygame.draw.rect(
                self.surface, entry.color, pygame.Rect(24, y + 9, 10, 10), border_radius=2
            )
            self._text(entry.name[:11], 42, y + 6, self.font)
            self._text(f"{entry.laps:5.2f} laps", 42, top + 2, self.font, ACCENT)
            # The parameter count is the one number that says what an entrant
            # bet on, and the whole argument about the search budget turns on it.
            self._text(f"{entry.params} par", 132, top + 5, self.font_small, MUTED)
            detail = f"best {entry.best:.2f}"
            if entry.shown:
                detail += f"  {entry.alive}/{entry.shown}"
            self._text(detail, 42, top + 20, self.font_small, MUTED)

            width = self.width - 24 - _TEXT_COLUMN
            if entry.brain is not None and width >= 90 and card_height >= 56:
                inner = (width, card_height - 16)
                self.surface.blit(self._cached_diagram(entry.brain, inner), (_TEXT_COLUMN, y + 8))
            y += card_height + 8

    # -- assembly ------------------------------------------------------------

    def render(
        self,
        generation: int,
        track_name: str,
        track_number: int,
        circuit_count: int,
        frame: int,
        max_frames: int,
        entries: Sequence[Entry],
    ) -> pygame.Surface:
        """The panel: who is ahead, what they are, and where the run is.

        Nothing about the search is here — it is all on the strip below, which
        runs the full width of the window. That is what lets these cards be
        tall enough for a brain diagram anyone can actually read.
        """
        self.surface.fill(BG)
        top = self._header(
            generation, track_name, track_number, circuit_count, frame, max_frames
        )
        self._entries(
            top + 2, entries, self._card_height(self.height - top - 16, len(entries))
        )
        return self.surface


__all__ = [
    "ACCENT",
    "BG",
    "CARD",
    "Dashboard",
    "Entry",
    "MUTED",
    "NEGATIVE",
    "POSITIVE",
    "Series",
    "TEXT",
    "response",
    "tint",
]
