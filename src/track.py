"""Serpentine racetrack: procedural centreline, drivable mask, lap progress.

The circuit is a closed polyline — long straights joined by 180-degree hairpins,
with a return corridor down the left side — smoothed with Chaikin's algorithm and
resampled at even spacing. Progress along the lap is measured by projecting a car
onto the nearest centreline sample, so the track may take any shape; the previous
polar formulation only worked for circuits that were star-shaped about a centre.
"""

import array
import logging
import math
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pygame

from src.config import TrackConfig

logger = logging.getLogger(__name__)

Point = Tuple[float, float]

# Samples either side of the hint searched by `nearest_index`.
_SEARCH_WINDOW = 25


def _chaikin(points: Sequence[Point], passes: int) -> List[Point]:
    """Round off a closed polyline by cutting every corner, `passes` times."""
    result = list(points)
    for _ in range(passes):
        cut: List[Point] = []
        for i, (x0, y0) in enumerate(result):
            x1, y1 = result[(i + 1) % len(result)]
            cut.append((0.75 * x0 + 0.25 * x1, 0.75 * y0 + 0.25 * y1))
            cut.append((0.25 * x0 + 0.75 * x1, 0.25 * y0 + 0.75 * y1))
        result = cut
    return result


def _resample(points: Sequence[Point], spacing: float) -> List[Point]:
    """Resample a closed polyline at even arc-length spacing."""
    out: List[Point] = [points[0]]
    carry = 0.0
    for i in range(len(points)):
        a, b = points[i], points[(i + 1) % len(points)]
        segment = math.dist(a, b)
        if segment == 0.0:
            continue
        t = spacing - carry
        while t <= segment:
            ratio = t / segment
            out.append((a[0] + (b[0] - a[0]) * ratio, a[1] + (b[1] - a[1]) * ratio))
            t += spacing
        carry = (carry + segment) % spacing
    return out


class Track:
    """A closed serpentine circuit: drawn surface, drivable mask, centreline."""

    def __init__(self, cfg: TrackConfig) -> None:
        self.cfg = cfg
        self.size: Tuple[int, int] = (cfg.width, cfg.height)
        smoothed = _chaikin(self._skeleton(), cfg.smoothing_passes)
        self.samples: np.ndarray = np.array(_resample(smoothed, cfg.sample_spacing))
        segments = np.linalg.norm(np.roll(self.samples, -1, axis=0) - self.samples, axis=1)
        self.lap_length: float = float(segments.sum())
        self.surface = self._render_surface()
        self.mask = self._build_mask(self.surface)
        self.clearance = self._build_clearance()
        self.clearance_flat = self._flat_clearance()
        # Samples with the search window wrapped onto either end, so the hinted
        # lookup below is a contiguous slice instead of a modulo and a gather.
        self._wrapped = np.vstack(
            [self.samples[-_SEARCH_WINDOW:], self.samples, self.samples[: _SEARCH_WINDOW + 1]]
        )

    def _skeleton(self) -> List[Point]:
        """Corner points of the raw circuit, before smoothing."""
        builders = {
            "zigzag": self._zigzag_skeleton,
            "grid": self._grid_skeleton,
            "speedway": self._speedway_skeleton,
            "serpentine": self._serpentine_skeleton,
        }
        points = builders[self.cfg.layout]()
        if self.cfg.mirrored:
            points = [(self.cfg.width - x, y) for x, y in points]
        return points

    def _zigzag_skeleton(self) -> List[Point]:
        """A loop whose long sides are chicanes: out along the top, back along
        the bottom, with a different number of peaks on each run."""
        cfg = self.cfg
        x_left, x_right = 110.0, 890.0
        y_top, y_bottom = 185.0, 515.0
        out_peaks, back_peaks = cfg.zigzag_peaks
        amp = cfg.zigzag_amplitude
        points: List[Point] = [(x_left, y_top)]

        for i in range(out_peaks):
            ratio = (i + 0.5) / out_peaks
            points.append((x_left + (x_right - x_left) * ratio, y_top + amp * (-1) ** i))
        points.append((x_right, y_top))
        points.append((x_right + 55, (y_top + y_bottom) / 2))
        points.append((x_right, y_bottom))

        for i in range(back_peaks):
            ratio = (i + 0.5) / back_peaks
            points.append((x_right - (x_right - x_left) * ratio, y_bottom + amp * 0.75 * (-1) ** i))
        points.append((x_left, y_bottom))
        points.append((x_left - 55, (y_top + y_bottom) / 2))
        return points

    def _grid_skeleton(self) -> List[Point]:
        """A city-style circuit of right angles: straights and square corners.

        Nothing like the flowing serpentine — here the car meets abrupt 90-degree
        turns, which demand braking into the corner rather than a steady arc.

        Eight corners, not the twelve it had. Twelve was not impossible — by
        generation 40 an entrant was lapping it at 2.72 — but it was learnt far
        more slowly than the other circuits, and while that lasted the compound
        odds of twelve corners in a row decided the ranking rather than anyone's
        driving. Eight keeps the circuit's identity, which is that it has more
        corners than anywhere else and every one of them is square.
        """
        return [
            (140.0, 120.0),
            (860.0, 120.0),
            (860.0, 330.0),
            (520.0, 330.0),
            (520.0, 470.0),
            (860.0, 470.0),
            (860.0, 590.0),
            (140.0, 590.0),
        ]

    def _speedway_skeleton(self) -> List[Point]:
        """Fast sweeping curves and long straights, broken by one tight chicane.

        The opposite problem to the grid: here the car must carry speed, then
        shed it for a single sharp sequence and pick it back up.
        """
        return [
            (170.0, 120.0),
            (520.0, 100.0),
            (800.0, 150.0),
            (900.0, 300.0),
            (860.0, 460.0),
            # Chicane: three tight direction changes in quick succession.
            (700.0, 470.0),
            (660.0, 590.0),
            (540.0, 500.0),
            (430.0, 600.0),
            (300.0, 560.0),
            (140.0, 470.0),
            (100.0, 300.0),
        ]

    def _serpentine_skeleton(self) -> List[Point]:
        """Lanes alternating direction, joined by hairpins, closed by a return
        corridor down the left-hand side."""
        cfg = self.cfg
        left, right = cfg.lane_x_left, cfg.lane_x_right
        points: List[Point] = []

        for i, y in enumerate(cfg.lane_ys):
            rightwards = i % 2 == 0
            start_x, end_x = (left, right) if rightwards else (right, left)
            if i == 0:
                start_x = cfg.return_x  # the first lane starts from the corridor
            points.append((start_x, y))
            # Several bumps along the straight, alternating side, so the car is
            # steering continuously rather than running dead straight.
            for b in range(cfg.bumps_per_straight):
                ratio = (b + 1) / (cfg.bumps_per_straight + 1)
                side = 1 if (b + i) % 2 == 0 else -1
                points.append((start_x + (end_x - start_x) * ratio, y + cfg.bump_amplitude * side))
            points.append((end_x, y))
            if i < len(cfg.lane_ys) - 1:
                points.extend(self._hairpin(end_x, y, cfg.lane_ys[i + 1], rightwards))

        # From the last lane, sweep down and left into the return corridor.
        # Spread over several points so the corner stays a curve, not a kink.
        points.append((cfg.lane_x_left - 30, cfg.return_y_bottom - 75))
        points.append((cfg.lane_x_left - 105, cfg.return_y_bottom - 25))
        points.append((cfg.return_x + 25, cfg.return_y_bottom))
        points.append((cfg.return_x, cfg.return_y_bottom - 45))
        return points

    def _hairpin(self, x: float, y_from: float, y_to: float, rightwards: bool) -> List[Point]:
        """A semicircular U-turn joining two lanes, as explicit arc points.

        Describing the arc directly keeps its radius equal to half the lane
        spacing. Leaving a single corner for the smoothing pass to round off
        would instead produce a radius far too tight to drive.
        """
        radius = abs(y_to - y_from) / 2.0
        cy = (y_from + y_to) / 2.0
        direction = 1.0 if rightwards else -1.0
        steps = self.cfg.hairpin_steps
        points: List[Point] = []
        for i in range(1, steps):
            angle = -math.pi / 2 + math.pi * i / steps
            points.append((x + direction * radius * math.cos(angle), cy + radius * math.sin(angle)))
        return points

    def road_width_at_index(self, index: int) -> float:
        """Local carriageway width; it breathes along the lap."""
        phase = 2 * math.pi * self.cfg.width_frequency * index / len(self.samples)
        return self.cfg.road_width * (1.0 + self.cfg.width_variation * math.sin(phase))

    def _island_shape(self, index: int, ordinal: int) -> Optional[List[Point]]:
        """Corners of one island, sitting in the middle of the carriageway.

        Sized from the road so a drivable corridor is left on either side, and
        skipped entirely when there is no room. Shapes alternate between round
        and angular: a hexagon reads as a traffic island, while triangles and
        squares give the car hard edges to squeeze past rather than smooth arcs.
        """
        cx, cy = self.samples[index]
        heading = self.heading_at_index(index)
        available = self.road_width_at_index(index) - 2 * self.cfg.island_min_corridor
        if available < 12.0:
            return None
        size = available / 2.0
        sides = (6, 3, 4, 5)[ordinal % 4]
        # Angular shapes are rotated off-axis so an edge, not a point, faces the car.
        spin = heading + (math.pi / sides if sides % 2 == 0 else 0.0)
        return [
            (
                cx + size * math.cos(spin + 2 * math.pi * k / sides),
                cy + size * math.sin(spin + 2 * math.pi * k / sides),
            )
            for k in range(sides)
        ]

    def _draw_islands(self, surface: pygame.Surface) -> None:
        for ordinal, fraction in enumerate(self.cfg.islands):
            shape = self._island_shape(self.index_at_fraction(fraction), ordinal)
            if shape is not None:
                pygame.draw.polygon(surface, self.cfg.grass_color, shape)

    def _render_surface(self) -> pygame.Surface:
        surface = pygame.Surface(self.size)
        surface.fill(self.cfg.grass_color)
        for i, (x, y) in enumerate(self.samples):
            pygame.draw.circle(
                surface, self.cfg.road_color, (int(x), int(y)), int(self.road_width_at_index(i) / 2)
            )
        # Islands are cut out of the road, so they are part of the track
        # geometry: the wall sensors and the collision mask see them for free.
        self._draw_islands(surface)
        # The start line is *not* painted here. Cars begin somewhere different
        # every generation, so a line baked into the texture would sit at a
        # place nobody starts from; the renderer draws it where the cars
        # actually are. Keeping it off the surface also keeps it out of the
        # drivable mask, which is derived from these pixels.
        return surface

    def start_line(self, index: int) -> Tuple[Point, Point]:
        """The two ends of a line across the road at this sample."""
        nx, ny = self.normal_at_index(index)
        half = self.road_width_at_index(index) / 2
        x, y = self.samples[index]
        return (x - nx * half, y - ny * half), (x + nx * half, y + ny * half)

    def _build_mask(self, surface: pygame.Surface) -> np.ndarray:
        """Boolean array indexed [x, y]: True where the car may drive."""
        pixels = pygame.surfarray.array3d(surface)
        return np.all(pixels == np.array(self.cfg.road_color), axis=-1)

    def _build_clearance(self, max_radius: int = 72) -> np.ndarray:
        """For each pixel, a safe lower bound on the distance to the verge.

        Built by repeatedly eroding the road with a 3x3 element: a pixel that
        survives k erosions is at least k pixels from any non-road pixel. That
        makes this the chessboard distance, which never exceeds the true
        Euclidean distance — exactly what sphere tracing needs, since a ray may
        safely jump forward by this much without any chance of skipping a wall.

        Costs a few milliseconds once per circuit, and saves the ray marcher
        roughly an order of magnitude of pixel lookups on every frame after.
        """
        surviving = self.mask.copy()
        clearance = surviving.astype(np.float32)
        for _ in range(max_radius - 1):
            if not surviving.any():
                break
            eroded = surviving.copy()
            eroded[1:, :] &= surviving[:-1, :]
            eroded[:-1, :] &= surviving[1:, :]
            eroded[:, 1:] &= surviving[:, :-1]
            eroded[:, :-1] &= surviving[:, 1:]
            eroded[1:, 1:] &= surviving[:-1, :-1]
            eroded[:-1, :-1] &= surviving[1:, 1:]
            eroded[1:, :-1] &= surviving[:-1, 1:]
            eroded[:-1, 1:] &= surviving[1:, :-1]
            surviving = eroded
            clearance += surviving
        return clearance

    def _flat_clearance(self) -> array.array:
        """The clearance field as a flat array of Python floats, row-major.

        The ray marcher reads one cell per step and does nothing else with it,
        and it is the hottest loop in the project by a distance. Reading a numpy
        array there hands back a `float32` scalar, and every comparison and
        addition that follows is then numpy arithmetic on a zero-dimensional
        object rather than on a machine float — 2.3x the cost of the same loop
        over an `array.array`, measured.

        The values are erosion counts, so they are integers from 0 to 49, exact
        in either width, and the marcher only ever adds integers to them. What
        changes is the ray's position arithmetic, which becomes float64 instead
        of float32; over four thousand sampled casts not one reading moved.
        """
        flat = array.array("d")
        flat.frombytes(np.ascontiguousarray(self.clearance, dtype=np.float64).tobytes())
        return flat

    def clearance_at(self, x: float, y: float) -> float:
        """Distance a ray may advance from here without leaving the road."""
        ix, iy = int(x), int(y)
        if ix < 0 or iy < 0 or ix >= self.cfg.width or iy >= self.cfg.height:
            return 0.0
        return float(self.clearance[ix, iy])

    def is_on_road(self, x: float, y: float) -> bool:
        ix, iy = int(x), int(y)
        if ix < 0 or iy < 0 or ix >= self.cfg.width or iy >= self.cfg.height:
            return False
        return bool(self.mask[ix, iy])

    def tangent_at_index(self, index: int) -> Tuple[float, float]:
        n = len(self.samples)
        ax, ay = self.samples[(index - 1) % n]
        bx, by = self.samples[(index + 1) % n]
        dx, dy = bx - ax, by - ay
        length = math.hypot(dx, dy) or 1.0
        return dx / length, dy / length

    def normal_at_index(self, index: int) -> Tuple[float, float]:
        tx, ty = self.tangent_at_index(index)
        return -ty, tx

    def nearest_index(self, x: float, y: float, hint: int = -1) -> int:
        """Index of the closest centreline sample.

        With a `hint` from the previous frame only a local window is searched,
        which keeps this O(window) instead of O(number of samples). This runs
        once per car per frame, so the window is a preallocated constant and the
        squared distance is taken on the columns rather than through `einsum`.
        """
        if hint < 0:
            deltas = self.samples - np.array([x, y])
            return int(np.argmin(np.einsum("ij,ij->i", deltas, deltas)))
        points = self._wrapped[hint : hint + 2 * _SEARCH_WINDOW + 1]
        dx = points[:, 0] - x
        dy = points[:, 1] - y
        offset = int(np.argmin(dx * dx + dy * dy)) - _SEARCH_WINDOW
        return (hint + offset) % len(self.samples)

    def index_at_fraction(self, fraction: float) -> int:
        """Sample index at a given fraction of the way round the lap."""
        return int(fraction * len(self.samples)) % len(self.samples)

    def valid_start_index(self, rng: np.random.Generator) -> int:
        """A random point on the lap that is actually drivable.

        The centreline runs through the islands, so a random sample may sit
        inside one; walk forward until the road reappears.
        """
        n = len(self.samples)
        index = int(rng.integers(0, n))
        for offset in range(n):
            candidate = (index + offset) % n
            x, y = self.samples[candidate]
            if self.is_on_road(x, y):
                return candidate
        return 0

    def position_at_index(self, index: int) -> Point:
        return float(self.samples[index][0]), float(self.samples[index][1])

    def heading_at_index(self, index: int) -> float:
        tx, ty = self.tangent_at_index(index)
        return math.atan2(ty, tx)
