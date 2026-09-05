"""Side panel: run status, fitness history and a view of the leader's network."""

from typing import List, Optional, Sequence, Tuple

import numpy as np
import pygame

from src.car import Car

_BG = (18, 20, 32)
_CARD = (28, 31, 48)
_TEXT = (232, 234, 245)
_MUTED = (140, 146, 170)
_ACCENT = (255, 176, 60)
_LINE = (90, 200, 250)
_POSITIVE = (110, 220, 150)
_NEGATIVE = (235, 100, 110)


def _is_drawable(brain: object) -> bool:
    """Whether a brain exposes enough structure to diagram.

    The `Brain` protocol deliberately stops at `forward` and the genome: a
    competitor's architecture owes the harness no account of its internals, and
    plenty of them have no layer stack to draw. So the panel asks, rather than
    assumes, and falls back to naming the brain when the answer is no.
    """
    return hasattr(brain, "layer_sizes") and hasattr(brain, "weights")


class Dashboard:
    """Renders the right-hand panel onto its own surface."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.surface = pygame.Surface((width, height))
        self.font_big = pygame.font.SysFont("menlo,monospace", 26, bold=True)
        self.font = pygame.font.SysFont("menlo,monospace", 14)
        self.font_small = pygame.font.SysFont("menlo,monospace", 11)
        self._network_cache: Optional[pygame.Surface] = None
        self._cached_genome_id: Optional[int] = None

    def _text(self, text: str, x: int, y: int, font: pygame.font.Font, color=_TEXT) -> None:
        self.surface.blit(font.render(text, True, color), (x, y))

    def _card(self, y: int, height: int, title: str) -> int:
        """Draw a titled card and return the y where its content starts."""
        rect = pygame.Rect(12, y, self.width - 24, height)
        pygame.draw.rect(self.surface, _CARD, rect, border_radius=8)
        self._text(title, 24, y + 8, self.font_small, _MUTED)
        return y + 26

    def _header(
        self, generation: int, track_name: str, number: int, count: int, frame: int, total: int
    ) -> int:
        self._text(f"GEN {generation}", 16, 14, self.font_big)
        self._text(f"TRACK {number}/{count}", self.width - 150, 16, self.font, _MUTED)
        self._text(f"STEP {frame}/{total}", self.width - 150, 34, self.font, _MUTED)
        self._text(track_name.upper(), 16, 46, self.font, _ACCENT)

        bar = pygame.Rect(16, 68, self.width - 32, 4)
        pygame.draw.rect(self.surface, _CARD, bar, border_radius=2)
        filled = pygame.Rect(16, 68, int((self.width - 32) * frame / max(1, total)), 4)
        pygame.draw.rect(self.surface, _ACCENT, filled, border_radius=2)
        return 84

    def _stats(self, y: int, cars: Sequence[Car], leader: Optional[Car], best_ever: float) -> int:
        top = self._card(y, 86, "THIS RUN")
        alive = sum(1 for c in cars if c.alive)
        rows = [
            ("alive", f"{alive}/{len(cars)}"),
            ("leader", f"{leader.laps:.2f} laps" if leader else "--"),
            ("best ever", f"{best_ever:.2f} laps"),
        ]
        for i, (label, value) in enumerate(rows):
            self._text(label, 24, top + i * 18, self.font, _MUTED)
            self._text(value, 150, top + i * 18, self.font)
        return y + 96

    def _history_chart(self, y: int, history: Sequence[float]) -> int:
        height = 96
        top = self._card(y, height, "BEST FITNESS / GENERATION")
        if len(history) >= 2:
            plot = pygame.Rect(28, top, self.width - 56, height - 40)
            peak = max(history) or 1.0
            points = [
                (
                    plot.x + plot.width * i / (len(history) - 1),
                    plot.bottom - plot.height * (value / peak),
                )
                for i, value in enumerate(history)
            ]
            pygame.draw.lines(self.surface, _LINE, False, points, 2)
            pygame.draw.circle(self.surface, _ACCENT, points[-1], 3)
            self._text(f"{peak:.2f}", self.width - 96, top - 18, self.font_small, _MUTED)
        else:
            self._text("collecting...", 28, top + 20, self.font_small, _MUTED)
        return y + height + 10

    def _network_surface(self, car: Car, size: Tuple[int, int]) -> pygame.Surface:
        """Draw the leader's weights: nodes per layer, edges tinted by sign."""
        surface = pygame.Surface(size)
        surface.fill(_CARD)
        layers = car.brain.layer_sizes
        margin_x, margin_y = 18, 12
        span_x = size[0] - 2 * margin_x
        columns = [
            margin_x + span_x * i / max(1, len(layers) - 1) for i in range(len(layers))
        ]

        def node_positions(count: int) -> List[float]:
            usable = size[1] - 2 * margin_y
            if count == 1:
                return [size[1] / 2]
            return [margin_y + usable * i / (count - 1) for i in range(count)]

        positions = [node_positions(n) for n in layers]
        scale = max(1e-6, max(float(np.abs(w).max()) for w in car.brain.weights))
        for layer, weights in enumerate(car.brain.weights):
            # Only the strongest connections, or the panel turns into a blob.
            threshold = np.percentile(np.abs(weights), 88)
            for i, j in zip(*np.where(np.abs(weights) >= threshold)):
                strength = abs(weights[i, j]) / scale
                color = _POSITIVE if weights[i, j] > 0 else _NEGATIVE
                tint = tuple(int(30 + (c - 30) * strength) for c in color)
                pygame.draw.line(
                    surface,
                    tint,
                    (columns[layer], positions[layer][i]),
                    (columns[layer + 1], positions[layer + 1][j]),
                    1,
                )
        for layer, ys in enumerate(positions):
            for y in ys:
                pygame.draw.circle(surface, _ACCENT if layer == 0 else _LINE, (int(columns[layer]), int(y)), 3)
        return surface

    def _network_card(self, y: int, leader: Optional[Car]) -> int:
        height = self.height - y - 16
        if leader is None:
            self._card(y, height, "LEADER NETWORK")
            return y + height
        if not _is_drawable(leader.brain):
            top = self._card(y, height, "LEADER BRAIN")
            self._text(type(leader.brain).__name__, 24, top, self.font, _ACCENT)
            self._text(f"{leader.brain.genome_size} parameters", 24, top + 20, self.font, _MUTED)
            self._text("no layer stack to draw", 24, top + 38, self.font_small, _MUTED)
            return y + height

        title = "-".join(str(n) for n in leader.brain.layer_sizes)
        top = self._card(y, height, f"LEADER NETWORK  {title}")
        genome_id = id(leader.brain)
        inner = (self.width - 48, height - 40)
        if self._cached_genome_id != genome_id or self._network_cache is None:
            self._network_cache = self._network_surface(leader, inner)
            self._cached_genome_id = genome_id
        self.surface.blit(self._network_cache, (24, top))
        return y + height

    def render(
        self,
        generation: int,
        track_name: str,
        track_number: int,
        track_count: int,
        frame: int,
        max_frames: int,
        cars: Sequence[Car],
        leader: Optional[Car],
        best_ever: float,
        history: Sequence[float],
    ) -> pygame.Surface:
        self.surface.fill(_BG)
        y = self._header(generation, track_name, track_number, track_count, frame, max_frames)
        y = self._stats(y, cars, leader, best_ever)
        y = self._history_chart(y, history)
        self._network_card(y + 4, leader)
        return self.surface
