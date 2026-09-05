"""Tests for the moving-hazard module.

Traffic lights and pedestrians are not wired into the simulation any more — the
obstacles in use are static islands cut into the road (see `TrackConfig.islands`).
The module is kept working and covered so the moving hazards can be brought back
by raising the counts in `ObstacleConfig`.
"""

import math

import numpy as np
import pygame
import pytest

from src.config import ObstacleConfig, track_variants
from src.obstacles import ObstacleField, Pedestrian, TrafficLight
from src.track import Track

# A configuration with hazards switched on, whatever the project default is.
POPULATED = ObstacleConfig(num_traffic_lights=3, num_pedestrians=3)


@pytest.fixture(scope="module")
def track() -> Track:
    pygame.init()
    return Track(track_variants()[0])


def test_traffic_light_cycles_red_and_green(track: Track) -> None:
    cfg = ObstacleConfig(light_period_frames=100, light_red_fraction=0.4)
    light = TrafficLight(track, cfg, index=100, phase=0)

    light.update(0)
    assert light.is_red and len(light.solid_circles()) == cfg.light_barrier_circles
    light.update(50)
    assert not light.is_red and len(light.solid_circles()) == 0
    light.update(100)
    assert light.is_red  # next cycle


def test_traffic_light_barrier_spans_the_road(track: Track) -> None:
    light = TrafficLight(track, POPULATED, index=100, phase=0)
    circles = light.solid_circles()
    span = math.dist(circles[0][:2], circles[-1][:2]) + 2 * circles[0][2]
    left, right = track.road_extent_at_index(100)
    assert span >= 0.95 * (left + right)


def test_pedestrian_stays_within_the_road(track: Track) -> None:
    walker = Pedestrian(track, POPULATED, index=200, phase=0)
    for frame in range(0, 400, 7):
        walker.update(frame)
        (x, y, _) = walker.solid_circles()[0]
        assert track.is_on_road(x, y)


def test_ray_distance_finds_a_circle_ahead_and_ignores_one_behind(track: Track) -> None:
    field = ObstacleField(track, ObstacleConfig(num_traffic_lights=0, num_pedestrians=0))
    field._solid = np.array([[100.0, 0.0, 10.0]])
    assert field.ray_distance(0.0, 0.0, 1.0, 0.0, 200.0) == pytest.approx(90.0)
    assert field.ray_distance(0.0, 0.0, -1.0, 0.0, 200.0) == 200.0
    assert field.ray_distance(0.0, 50.0, 1.0, 0.0, 200.0) == 200.0  # ray passes wide


def test_blocking_and_lethal_hazards_are_kept_apart(track: Track) -> None:
    """Red lights stop a car; pedestrians end its run. Different arrays."""
    field = ObstacleField(track, POPULATED)
    field.update(0)
    assert field._blocking.size > 0  # the lights
    assert field._lethal.size > 0  # the pedestrians
    assert len(field._solid) == len(field._blocking) + len(field._lethal)


def test_set_active_removes_every_hazard(track: Track) -> None:
    field = ObstacleField(track, POPULATED)
    field.update(0)
    assert field._solid.size > 0
    field.set_active(False)
    assert field._solid.size == 0
    assert field.render_items() == []


def test_randomize_moves_hazards_and_keeps_them_spread(track: Track) -> None:
    field = ObstacleField(track, POPULATED)
    before = [h.index for h in field.hazards]
    field.randomize(np.random.default_rng(3))
    after = [h.index for h in field.hazards]

    assert after != before
    assert len(after) == len(before)
    assert sorted(after) == after  # one per slot, never bunched in one corner
    assert min(after) >= 0 and max(after) < len(track.samples)


def test_randomize_is_reproducible_from_a_seed(track: Track) -> None:
    fields = [ObstacleField(track, POPULATED) for _ in range(2)]
    for field in fields:
        field.randomize(np.random.default_rng(7))
    assert [h.index for h in fields[0].hazards] == [h.index for h in fields[1].hazards]
