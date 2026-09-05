"""Geometric validation of a generated circuit.

A procedurally generated track can be subtly broken: two parts of the lap may
touch (giving cars a shortcut and corrupting progress measurement), a corner may
be tighter than the car can physically turn, or the layout may spill outside the
window. These checks catch all three before a training run is wasted.
"""

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from src.config import CarConfig
from src.track import Track

_NEIGHBOUR_SKIP = 60  # samples within this index distance are the same stretch


@dataclass(frozen=True)
class TrackReport:
    """Measured geometry of a circuit, with the reasons it failed (if any)."""

    name: str
    lap_length: float
    min_separation: float
    min_radius: float
    min_width: float  # narrowest drivable corridor anywhere on the lap
    bounds: Tuple[float, float, float, float]
    problems: Tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.problems

    def max_speed_in_tightest_corner(self, car: CarConfig) -> float:
        return self.min_radius * math.radians(car.max_steering)


def _min_separation(track: Track) -> float:
    """Closest approach between two non-adjacent stretches of the lap."""
    n = len(track.samples)
    indices = np.arange(n)
    closest = math.inf
    for i in range(0, n, 3):
        far = (np.abs(indices - i) > _NEIGHBOUR_SKIP) & (np.abs(indices - i) < n - _NEIGHBOUR_SKIP)
        if not far.any():
            continue
        deltas = track.samples[far] - track.samples[i]
        closest = min(closest, float(np.sqrt(np.einsum("ij,ij->i", deltas, deltas)).min()))
    return closest


def widest_corridor(track: Track, index: int) -> float:
    """Width of the widest continuous gap across the road at this point.

    An island splits the carriageway in two, so measuring outwards from the
    centre reports the island instead of the road. What matters is that at least
    one side of it is wide enough to drive through.
    """
    cx, cy = track.samples[index]
    nx, ny = track.normal_at_index(index)
    limit = track.cfg.road_width * (1.0 + track.cfg.width_variation)
    best = run = 0.0
    offset = -limit
    while offset <= limit:
        if track.is_on_road(cx + nx * offset, cy + ny * offset):
            run += 1.0
            best = max(best, run)
        else:
            run = 0.0
        offset += 1.0
    return best


def _min_radius(track: Track, span: int = 6) -> float:
    """Smallest circumscribed-circle radius along the centreline."""
    n = len(track.samples)
    smallest = math.inf
    for i in range(n):
        p0, p1, p2 = track.samples[(i - span) % n], track.samples[i], track.samples[(i + span) % n]
        area = abs((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1])) / 2
        if area < 1e-9:
            continue
        sides = math.dist(p0, p1) * math.dist(p1, p2) * math.dist(p0, p2)
        smallest = min(smallest, sides / (4 * area))
    return smallest


def check_track(track: Track, car: CarConfig, margin: float = 15.0) -> TrackReport:
    """Measure a circuit and report anything that makes it unusable."""
    cfg = track.cfg
    separation = _min_separation(track)
    radius = _min_radius(track)
    widths = [widest_corridor(track, i) for i in range(0, len(track.samples), 7)]
    xs, ys = track.samples[:, 0], track.samples[:, 1]
    bounds = (float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max()))

    problems: List[str] = []
    if separation <= cfg.road_width + 6:
        problems.append(f"lap touches itself (separation {separation:.0f}px)")
    if min(widths) < car.width * 2.5:
        problems.append(f"no drivable corridor ({min(widths):.0f}px at its narrowest)")
    half = cfg.road_width * (1 + cfg.width_variation) / 2
    if (
        bounds[0] - half < margin
        or bounds[1] + half > cfg.width - margin
        or bounds[2] - half < margin
        or bounds[3] + half > cfg.height - margin
    ):
        problems.append("layout spills outside the window")

    return TrackReport(
        name=cfg.name,
        lap_length=track.lap_length,
        min_separation=separation,
        min_radius=radius,
        min_width=min(widths),
        bounds=bounds,
        problems=tuple(problems),
    )
