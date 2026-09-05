"""Pygame window: the circuit and the standings on top, the analyses beneath.

    +--------------------------+---------+
    |                          | panel   |   the circuit, drawn at whatever
    |         circuit          | one     |   scale the display allows, and one
    |                          | card    |   card per entrant beside it
    |                          | each    |
    +--------------------------+---------+
    |        analyses, the full width     |   everything about the search
    +-------------------------------------+

The circuit is drawn at its native 1000x700 onto an off-screen surface and
scaled on the way to the window. Without that the window is 1100 px tall before
anything is on screen, which is more than a 900 px display has — and the strip
was then dropped, which is how the analyses came to be invisible on the machine
they were written for.
"""

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import pygame

from src.car import Car
from src.track import Track

_SENSOR_COLOR = (90, 200, 250)

# Room for the window frame, the title bar, the menu bar and the dock.
_CHROME_W, _CHROME_H = 60, 90
_MIN_PANEL = 340
_MIN_STRIP = 150  # below this the analyses are unreadable, so there are none
# The circuit shrinks to make room for the analyses, but only so far: past this
# the cars are too small to follow, and the strip is not worth what it costs.
_MIN_SCALE = 0.62


@dataclass(frozen=True)
class Layout:
    """Where everything goes, once the display has had its say."""

    view: Tuple[int, int]  # the circuit as drawn, which may be scaled down
    panel: int  # width of the standings column, beside the circuit
    strip: int  # height of the analyses, across the full width; 0 if none
    scale: float  # what the circuit was shrunk by; 1.0 if not at all

    @property
    def window(self) -> Tuple[int, int]:
        return (self.view[0] + self.panel, self.view[1] + self.strip)


def fit_window(
    track: Tuple[int, int],
    panel: int,
    strip: int,
    screen: Optional[Tuple[int, int]] = None,
) -> Layout:
    """Fit the circuit, the panel and the analyses onto this display.

    The request — a 1000x700 circuit, a panel beside it and a 280 px strip
    underneath — is 1380x980, which a 1440x900 laptop cannot show. Something has
    to give, and the order matters: the analyses are the whole point of the
    strip, so the circuit is scaled down to make room for them rather than the
    strip being dropped to keep the circuit at full size. Only when even a
    shrunken circuit leaves no readable strip is the strip given up.

    `screen` overrides what the display reports, which is the only way to test
    the arithmetic without owning the monitor it runs on.
    """
    if screen is None:
        try:
            info = pygame.display.Info()
            screen = (info.current_w, info.current_h)
        except pygame.error:
            return Layout(track, panel, strip, 1.0)

    room_w, room_h = screen[0] - _CHROME_W, screen[1] - _CHROME_H
    if room_w <= track[0] // 2 or room_h <= track[1] // 2:  # no display, or a dummy one
        return Layout(track, panel, strip, 1.0)

    panel = max(_MIN_PANEL, min(panel, room_w - int(track[0] * _MIN_SCALE)))
    scale = min(1.0, (room_w - panel) / track[0], (room_h - strip) / track[1])

    if scale < _MIN_SCALE:
        # The strip cannot have all it asked for. Give the circuit its floor and
        # let the strip take whatever is left of the height, or nothing.
        scale = min(1.0, (room_w - panel) / track[0], _MIN_SCALE)
        strip = int(room_h - track[1] * scale)
        if strip < _MIN_STRIP:
            strip = 0
            scale = min(1.0, (room_w - panel) / track[0], room_h / track[1])

    view = (int(track[0] * scale), int(track[1] * scale))
    return Layout(view, panel, max(0, strip), scale)


class Renderer:
    """Owns the window. Track drawing here, panel and strip content elsewhere."""

    def __init__(self, track_size: Tuple[int, int], layout: Layout, fps: int) -> None:
        self.track_size = track_size
        self.layout = layout
        self.fps = fps
        self.screen = pygame.display.set_mode(layout.window)
        # The circuit is always drawn at its own coordinates and scaled once, on
        # the way out. Scaling the coordinates instead would mean scaling every
        # car, every sensor ray and the start line, in three separate places.
        self.view = pygame.Surface(track_size)
        pygame.display.set_caption("cars_ai - evolutionary self-driving cars")
        self.clock = pygame.time.Clock()

    def poll_quit(self) -> bool:
        """Return True if the user asked to close the window."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return True
        return False

    def _draw_car(self, car: Car, color: Tuple[int, int, int]) -> None:
        cfg = car.cfg
        cos_a, sin_a = math.cos(car.angle), math.sin(car.angle)
        half_l, half_w = cfg.length / 2, cfg.width / 2
        corners = [(-half_l, -half_w), (half_l, -half_w), (half_l, half_w), (-half_l, half_w)]
        points = [
            (car.x + cx * cos_a - cy * sin_a, car.y + cx * sin_a + cy * cos_a)
            for cx, cy in corners
        ]
        pygame.draw.polygon(self.view, color, points)

    def draw_track(
        self,
        track: Track,
        cars: Sequence[Car],
        colors: Sequence[Tuple[int, int, int]],
        start_index: int = 0,
    ) -> None:
        """Draw the circuit and the cars, one colour per car.

        Every car on screen belongs to an entrant that must stay recognisable,
        so the colour is always the entrant's; the leader is picked out by its
        sensor rays rather than by a colour of its own.

        The start line is drawn here rather than baked into the circuit, so it
        marks where this generation actually began. A fixed line was worse than
        no line: it said the lap started somewhere the cars had never been.
        """
        self.view.blit(track.surface, (0, 0))
        head, tail = track.start_line(start_index)
        pygame.draw.line(self.view, track.cfg.line_color, head, tail, 3)
        alive = [(car, color) for car, color in zip(cars, colors) if car.alive]
        for car, color in alive:
            self._draw_car(car, color)
        leader = max((car for car, _ in alive), key=lambda c: c.fitness, default=None)
        if leader is not None:
            for endpoint in leader.sensor_endpoints:
                pygame.draw.line(self.view, _SENSOR_COLOR, (leader.x, leader.y), endpoint, 1)

    def present(self, panel: pygame.Surface, strip: Optional[pygame.Surface] = None) -> None:
        """Scale the circuit into place, blit the panel and the strip, and flip."""
        if self.layout.scale >= 1.0:
            self.screen.blit(self.view, (0, 0))
        else:
            self.screen.blit(pygame.transform.smoothscale(self.view, self.layout.view), (0, 0))
        self.screen.blit(panel, (self.layout.view[0], 0))
        if strip is not None:
            self.screen.blit(strip, (0, self.layout.view[1]))
        pygame.display.flip()
        self.clock.tick(self.fps)
