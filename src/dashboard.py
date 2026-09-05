"""Side panel: who is ahead, and how each entrant has improved.

The layout is derived from how many entrants there are, so adding a file to
`src/brains/` widens the standings and adds a curve with no change here.
"""

from typing import List, Sequence, Tuple

import pygame

from src.brains import Color

_BG = (18, 20, 32)
_CARD = (28, 31, 48)
_TEXT = (232, 234, 245)
_MUTED = (140, 146, 170)
_ACCENT = (255, 176, 60)

# name, colour, laps now, alive, shown, best ever
Standing = Tuple[str, Color, float, int, int, float]
Curve = Tuple[str, Color, Sequence[float]]


class Dashboard:
    """Renders the right-hand panel onto its own surface."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.surface = pygame.Surface((width, height))
        self.font_big = pygame.font.SysFont("menlo,monospace", 26, bold=True)
        self.font = pygame.font.SysFont("menlo,monospace", 14)
        self.font_small = pygame.font.SysFont("menlo,monospace", 11)

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
        label = f"GEN {generation}" if generation else "RACE"
        self._text(label, 16, 14, self.font_big)
        self._text(f"TRACK {number}/{count}", self.width - 150, 16, self.font, _MUTED)
        self._text(f"STEP {frame}/{total}", self.width - 150, 34, self.font, _MUTED)
        self._text(track.upper(), 16, 46, self.font, _ACCENT)

        bar = pygame.Rect(16, 68, self.width - 32, 4)
        pygame.draw.rect(self.surface, _CARD, bar, border_radius=2)
        filled = pygame.Rect(16, 68, int((self.width - 32) * frame / max(1, total)), 4)
        pygame.draw.rect(self.surface, _ACCENT, filled, border_radius=2)
        return 84

    def _standings(self, y: int, standings: Sequence[Standing]) -> int:
        """One row per entrant, ordered by who is furthest round right now."""
        height = 34 + 40 * len(standings)
        top = self._card(y, height, "STANDINGS")
        for name, color, laps, alive, shown, best in sorted(standings, key=lambda s: -s[2]):
            pygame.draw.rect(self.surface, color, pygame.Rect(24, top + 3, 10, 10), border_radius=2)
            self._text(f"{name[:12]:<12} {laps:5.2f}", 42, top, self.font)
            detail = f"best {best:.2f}   {alive}/{shown} alive" if shown else f"best {best:.2f}"
            self._text(detail, 42, top + 17, self.font_small, _MUTED)
            top += 40
        return y + height + 10

    def _chart(self, y: int, curves: Sequence[Curve]) -> int:
        """Best fitness per generation, one line per entrant in its colour."""
        height = self.height - y - 16
        top = self._card(y, height, "BEST FITNESS / GENERATION")
        longest = max((len(history) for _, _, history in curves), default=0)
        if longest < 2:
            self._text("collecting...", 28, top + 20, self.font_small, _MUTED)
            return y + height

        plot = pygame.Rect(28, top + 6, self.width - 56, height - 52)
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
        return y + height

    def render(
        self,
        generation: int,
        track_name: str,
        track_number: int,
        track_count: int,
        frame: int,
        max_frames: int,
        standings: Sequence[Standing],
        curves: Sequence[Curve],
    ) -> pygame.Surface:
        self.surface.fill(_BG)
        y = self._header(generation, track_name, track_number, track_count, frame, max_frames)
        y = self._standings(y, standings)
        self._chart(y, curves)
        return self.surface


__all__ = ["Curve", "Dashboard", "Standing"]
