"""Every circuit must be geometrically valid before it is trained on."""

import pygame
import pytest

from src.config import CarConfig, TrackConfig, race_track, track_variants
from src.track import Track
from src.track_check import check_track, widest_corridor

ALL_TRACKS = list(track_variants()) + [race_track()]


@pytest.fixture(scope="module", autouse=True)
def pygame_ready() -> None:
    pygame.init()


@pytest.mark.parametrize("cfg", ALL_TRACKS, ids=[c.name for c in ALL_TRACKS])
def test_track_is_valid(cfg: TrackConfig) -> None:
    report = check_track(Track(cfg), CarConfig())
    assert report.ok, f"{cfg.name}: {'; '.join(report.problems)}"


@pytest.mark.parametrize("cfg", ALL_TRACKS, ids=[c.name for c in ALL_TRACKS])
def test_corners_are_drivable(cfg: TrackConfig) -> None:
    """A corner tighter than the car can turn at crawling speed is impassable."""
    car = CarConfig()
    report = check_track(Track(cfg), car)
    assert report.max_speed_in_tightest_corner(car) >= 1.0


@pytest.mark.parametrize("cfg", ALL_TRACKS, ids=[c.name for c in ALL_TRACKS])
def test_islands_block_the_road_without_sealing_it(cfg: TrackConfig) -> None:
    """An island must be solid to the sensors, and leave a way past."""
    track = Track(cfg)
    for ordinal, fraction in enumerate(cfg.islands):
        index = track.index_at_fraction(fraction)
        shape = track._island_shape(index, ordinal)
        if shape is None:
            continue  # no room here, so nothing was drawn
        centre = track.samples[index]
        assert not track.is_on_road(*centre), "island centre must not be drivable"
        assert widest_corridor(track, index) >= CarConfig().width * 2.5


def test_layouts_are_structurally_different() -> None:
    """Three circuits of the same shape would only test memorisation."""
    layouts = [cfg.layout for cfg in track_variants()]
    assert len(set(layouts)) == len(layouts)
    assert race_track().layout not in layouts


def test_serpentine_lane_count_is_even() -> None:
    """An odd number of lanes ends on the wrong side and crosses the return."""
    for cfg in track_variants():
        if cfg.layout == "serpentine":
            assert len(cfg.lane_ys) % 2 == 0, cfg.name


def test_self_touching_layout_is_reported() -> None:
    """The validator must actually catch a broken circuit, not just pass."""
    broken = TrackConfig(name="broken", lane_ys=(95.0, 245.0, 395.0), bump_amplitude=90.0)
    report = check_track(Track(broken), CarConfig())
    assert not report.ok


def test_frame_budget_grants_every_circuit_the_same_laps() -> None:
    """A flat frame budget favours short laps; the budget must scale with them."""
    from src.config import SimulationConfig
    from src.simulation import Simulation

    cfg = SimulationConfig()
    simulation = Simulation(cfg, render=False)
    granted = [
        cfg.car.max_speed * simulation.frame_budget(track) / track.lap_length
        for track in simulation.tracks
    ]
    for laps in granted:
        assert laps == pytest.approx(cfg.laps_budget, abs=0.01)


def test_random_starts_are_drivable() -> None:
    """A start point inside an island would kill the car on frame one."""
    import numpy as np

    rng = np.random.default_rng(11)
    for cfg in ALL_TRACKS:
        track = Track(cfg)
        for _ in range(40):
            index = track.valid_start_index(rng)
            assert track.is_on_road(*track.position_at_index(index)), cfg.name


def test_island_layouts_differ_between_generations() -> None:
    """Fixed island positions are memorisable; the layout must rotate."""
    from src.config import SimulationConfig
    from src.simulation import Simulation

    simulation = Simulation(SimulationConfig(), render=False)
    layouts = {variant.cfg.islands for variant in simulation.circuits[0]}
    assert len(layouts) == len(simulation.circuits[0]) > 1
