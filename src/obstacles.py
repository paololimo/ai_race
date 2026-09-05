"""Timed hazards: traffic lights that turn red and pedestrians crossing the road.

Every hazard is represented as a set of circles, so collision checks and sensor
rays reduce to closed-form circle maths instead of pixel marching.
"""

import logging
import math
from typing import List, Sequence, Tuple

import numpy as np

from src.config import ObstacleConfig
from src.track import Track

logger = logging.getLogger(__name__)

Circle = Tuple[float, float, float]  # x, y, radius
Point = Tuple[float, float]


def _placement(track: Track, index: int) -> Tuple[Point, Tuple[float, float], float, float]:
    """Centre, normal and measured road extents at a centreline sample."""
    cx, cy = track.samples[index]
    normal = track.normal_at_index(index)
    left, right = track.road_extent_at_index(index)
    return (float(cx), float(cy)), normal, left, right


class TrafficLight:
    """A barrier across the road, solid only during the red phase."""

    def __init__(self, track: Track, cfg: ObstacleConfig, index: int, phase: int) -> None:
        self.cfg = cfg
        self.index = index
        self.phase = phase
        (cx, cy), (nx, ny), left, right = _placement(track, index)
        width = left + right
        n = cfg.light_barrier_circles
        radius = width / (2.0 * n)
        span_start = -left + radius
        self._circles: List[Circle] = []
        for i in range(n):
            offset = span_start + i * (width - 2 * radius) / max(1, n - 1)
            self._circles.append((cx + nx * offset, cy + ny * offset, radius))
        self.is_red = True

    def update(self, frame: int) -> None:
        position = (frame + self.phase) % self.cfg.light_period_frames
        self.is_red = position < self.cfg.light_period_frames * self.cfg.light_red_fraction

    def solid_circles(self) -> Sequence[Circle]:
        return self._circles if self.is_red else ()

    def render_circles(self) -> Tuple[Sequence[Circle], Tuple[int, int, int]]:
        color = self.cfg.red_color if self.is_red else self.cfg.green_color
        return self._circles, color


class Pedestrian:
    """A circle crossing the road back and forth, always solid."""

    def __init__(self, track: Track, cfg: ObstacleConfig, index: int, phase: int) -> None:
        self.cfg = cfg
        self.index = index
        self.phase = phase
        (cx, cy), self._normal, left, right = _placement(track, index)
        # Walk symmetrically around the true middle of the carriageway.
        middle = (right - left) / 2.0
        self._origin = (cx + self._normal[0] * middle, cy + self._normal[1] * middle)
        self._span = max(1.0, (left + right) / 2.0 - cfg.pedestrian_radius)
        self._position = self._origin

    def update(self, frame: int) -> None:
        # Triangular wave: walk across, turn around, walk back.
        period = max(1.0, 4.0 * self._span / self.cfg.pedestrian_speed)
        t = ((frame + self.phase) % period) / period
        offset = (4.0 * abs(t - 0.5) - 1.0) * self._span
        self._position = (
            self._origin[0] + self._normal[0] * offset,
            self._origin[1] + self._normal[1] * offset,
        )

    def solid_circles(self) -> Sequence[Circle]:
        return ((self._position[0], self._position[1], self.cfg.pedestrian_radius),)

    def render_circles(self) -> Tuple[Sequence[Circle], Tuple[int, int, int]]:
        return self.solid_circles(), self.cfg.pedestrian_color


class ObstacleField:
    """All hazards on the circuit, plus the queries cars need each frame."""

    def __init__(self, track: Track, cfg: ObstacleConfig) -> None:
        self.cfg = cfg
        self.hazards: List[object] = []
        self.active = cfg.enabled
        if not cfg.enabled:
            self._solid = np.zeros((0, 3))
            return

        self._track = track
        self._place(fractions=None, rng=None)
        self._clear()
        self.update(0)

    def _place(self, fractions, rng) -> None:
        """(Re)create every hazard at the given fractions of the lap."""
        cfg = self.cfg
        track = self._track
        total = cfg.num_traffic_lights + cfg.num_pedestrians
        if fractions is None:
            fractions = [(i + 0.5) / max(1, total) for i in range(total)]

        self.hazards = []
        for i in range(cfg.num_traffic_lights):
            index = track.index_at_fraction(fractions[i])
            phase = int(i * cfg.light_period_frames / max(1, cfg.num_traffic_lights))
            self.hazards.append(TrafficLight(track, cfg, index, phase))
        for i in range(cfg.num_pedestrians):
            index = track.index_at_fraction(fractions[cfg.num_traffic_lights + i])
            phase = int(rng.integers(0, 400)) if rng is not None else i * 37
            self.hazards.append(Pedestrian(track, cfg, index, phase))

    def randomize(self, rng: np.random.Generator) -> None:
        """Scatter the hazards afresh, keeping them spread around the lap.

        Fixed positions and phases are memorisable: a population can learn where
        the pedestrians stand on the training circuits instead of learning to
        avoid pedestrians, and then fails on any new layout. Re-drawing them each
        generation makes reacting to the sensors the only strategy that works.
        """
        total = self.cfg.num_traffic_lights + self.cfg.num_pedestrians
        if total == 0:
            return
        slot = 1.0 / total
        fractions = [(i + float(rng.uniform(0.15, 0.85))) * slot for i in range(total)]
        self._place(fractions, rng)

    def _clear(self) -> None:
        self._solid = np.zeros((0, 3))
        self._blocking = np.zeros((0, 3))
        self._lethal = np.zeros((0, 3))

    def set_active(self, active: bool) -> None:
        """Enable or disable every hazard, for curriculum training."""
        self.active = active and self.cfg.enabled
        if not self.active:
            self._clear()

    def update(self, frame: int) -> None:
        """Advance every hazard and rebuild the arrays of solid circles.

        A red light is *blocking*: a car that reaches it is stopped and loses
        time, but survives. A pedestrian is *lethal*: hitting one ends the run.
        """
        if not self.active:
            self._clear()
            return
        blocking: List[Circle] = []
        lethal: List[Circle] = []
        for hazard in self.hazards:
            hazard.update(frame)  # type: ignore[attr-defined]
            target = lethal if isinstance(hazard, Pedestrian) else blocking
            target.extend(hazard.solid_circles())  # type: ignore[attr-defined]
        self._blocking = np.array(blocking, dtype=float) if blocking else np.zeros((0, 3))
        self._lethal = np.array(lethal, dtype=float) if lethal else np.zeros((0, 3))
        self._solid = np.vstack([self._blocking, self._lethal])

    @staticmethod
    def _hits(circles: np.ndarray, x: float, y: float, radius: float) -> bool:
        if circles.size == 0:
            return False
        dx = circles[:, 0] - x
        dy = circles[:, 1] - y
        return bool(np.any(dx * dx + dy * dy <= (circles[:, 2] + radius) ** 2))

    def collides_blocking(self, x: float, y: float, radius: float) -> bool:
        """Hit a red light: the car is stopped, not destroyed."""
        return self._hits(self._blocking, x, y, radius)

    def collides_lethal(self, x: float, y: float, radius: float) -> bool:
        """Hit a pedestrian: the run ends."""
        return self._hits(self._lethal, x, y, radius)

    def ray_distance(self, x: float, y: float, dx: float, dy: float, max_range: float) -> float:
        """Distance to the nearest solid circle along a ray, or `max_range`."""
        if self._solid.size == 0:
            return max_range
        ox = self._solid[:, 0] - x
        oy = self._solid[:, 1] - y
        radii = self._solid[:, 2]
        along = ox * dx + oy * dy  # projection onto the ray direction
        perp_sq = ox * ox + oy * oy - along * along
        hittable = (along > 0) & (perp_sq <= radii * radii)
        if not np.any(hittable):
            return max_range
        half_chord = np.sqrt(np.maximum(0.0, radii[hittable] ** 2 - perp_sq[hittable]))
        distances = np.maximum(0.0, along[hittable] - half_chord)
        return float(min(max_range, distances.min()))

    def render_items(self) -> List[Tuple[Sequence[Circle], Tuple[int, int, int]]]:
        if not self.active:
            return []
        return [h.render_circles() for h in self.hazards]  # type: ignore[attr-defined]
