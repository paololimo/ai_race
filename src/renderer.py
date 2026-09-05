"""Pygame window: the circuit on the left, the dashboard panel on the right."""

import math
from typing import Optional, Sequence, Tuple

import pygame

from src.car import Car
from src.track import Track

_CAR_COLOR = (196, 74, 74)
_BEST_COLOR = (255, 176, 60)
_SENSOR_COLOR = (90, 200, 250)


class Renderer:
    """Owns the window. Track drawing here, panel content in `Dashboard`."""

    def __init__(self, track_size: Tuple[int, int], panel_width: int, fps: int) -> None:
        self.track_size = track_size
        self.panel_width = panel_width
        self.fps = fps
        self.screen = pygame.display.set_mode((track_size[0] + panel_width, track_size[1]))
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
        pygame.draw.polygon(self.screen, color, points)

    def draw_track(self, track: Track, cars: Sequence[Car]) -> Optional[Car]:
        """Draw the circuit and the cars; return the current leader."""
        self.screen.blit(track.surface, (0, 0))
        alive = [c for c in cars if c.alive]
        leader = max(alive, key=lambda c: c.fitness, default=None)
        for car in alive:
            if car is not leader:
                self._draw_car(car, _CAR_COLOR)
        if leader is not None:
            for endpoint in leader.sensor_endpoints:
                pygame.draw.line(self.screen, _SENSOR_COLOR, (leader.x, leader.y), endpoint, 1)
            self._draw_car(leader, _BEST_COLOR)
        return leader

    def present(self, panel: pygame.Surface) -> None:
        """Blit the dashboard panel and flip the frame."""
        self.screen.blit(panel, (self.track_size[0], 0))
        pygame.display.flip()
        self.clock.tick(self.fps)
