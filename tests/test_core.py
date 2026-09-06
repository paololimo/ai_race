"""Smoke tests for the genetic operators, the track and the car."""

import numpy as np
import pygame
import pytest

from src.config import CarConfig, GeneticConfig, track_variants
from src.car import Car, network_input_size
from src.genetic import crossover, mutate, next_generation, select_breeding_pool
from src.track import Track
from src.track_check import widest_corridor

INPUTS = network_input_size(CarConfig())


class StubBrain:
    """Something for a car to drive with.

    These tests are about the track, the physics and the genetic operators, so
    they need *a* brain, not any entrant in particular — and the repository is
    handed to competing agents with every entrant but their own removed, so
    naming one here would stop that copy from collecting at all.
    """

    def __init__(self, input_size: int, rng: np.random.Generator, still: bool = False) -> None:
        self.w = np.zeros((input_size, 2)) if still else rng.normal(0.0, 0.4, (input_size, 2))

    @property
    def genome_size(self) -> int:
        return self.w.size

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        return np.tanh(inputs @ self.w)

    def get_genome(self) -> np.ndarray:
        return self.w.ravel()

    def set_genome(self, genome: np.ndarray) -> None:
        self.w = genome.reshape(self.w.shape)


@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture(scope="module")
def track() -> Track:
    pygame.init()
    return Track(track_variants()[0])


def test_crossover_takes_genes_from_both_parents(rng: np.random.Generator) -> None:
    a, b = np.zeros(100), np.ones(100)
    child = crossover(a, b, rng)
    assert set(np.unique(child)) <= {0.0, 1.0}
    assert 0.0 in child and 1.0 in child


def test_mutation_changes_few_genes(rng: np.random.Generator) -> None:
    cfg = GeneticConfig(mutation_rate=0.05)
    genome = np.zeros(1000)
    mutated = mutate(genome, cfg, rng)
    changed = int(np.count_nonzero(mutated))
    assert 10 < changed < 120  # ~5% of 1000, with sampling slack


def test_selection_is_fitness_ranked() -> None:
    genomes = [np.full(4, i, dtype=float) for i in range(10)]
    pool = select_breeding_pool(genomes, list(range(10)), GeneticConfig(breeding_fraction=0.3))
    assert [p[0] for p in pool] == [9.0, 8.0, 7.0]


def test_next_generation_keeps_population_size_and_elites() -> None:
    cfg = GeneticConfig(population_size=20, elite_count=2)
    rng = np.random.default_rng(1)
    genomes = [rng.normal(size=8) for _ in range(20)]
    fitnesses = list(np.arange(20.0))
    children = next_generation(genomes, fitnesses, cfg, rng)
    assert len(children) == cfg.population_size
    assert np.allclose(children[0], genomes[19])  # best genome carried over untouched


def test_track_start_is_on_road(track: Track) -> None:
    assert track.is_on_road(*track.position_at_index(0))
    assert not track.is_on_road(1.0, 1.0)
    assert not track.is_on_road(-5.0, 10.0)


def test_centreline_samples_are_evenly_spaced(track: Track) -> None:
    gaps = np.linalg.norm(np.diff(track.samples, axis=0), axis=1)
    assert np.allclose(gaps, track.cfg.sample_spacing, atol=0.5)


def test_every_point_of_the_lap_has_a_way_through(track: Track) -> None:
    """The centreline runs through the islands, so the centre itself may be
    blocked; what must always hold is that a drivable corridor exists."""
    for index in range(0, len(track.samples), 10):
        assert widest_corridor(track, index) >= CarConfig().width * 2.5


def test_nearest_index_matches_brute_force(track: Track) -> None:
    for probe in track.samples[::97]:
        point = (probe[0] + 3.0, probe[1] - 2.0)
        brute = track.nearest_index(*point)
        local = track.nearest_index(*point, hint=brute)
        assert local == brute


def test_progress_tracks_distance_along_the_lap(track: Track, rng: np.random.Generator) -> None:
    car = Car(track, StubBrain(INPUTS, rng), CarConfig())
    # Teleport forward along the centreline; progress must follow the arc length.
    for step in range(1, 20):
        car.x, car.y = track.samples[step * 5]
        car._update_progress()
    assert car.progress == pytest.approx(95 * track.cfg.sample_spacing, abs=1.0)


def test_car_senses_and_dies_off_road(track: Track, rng: np.random.Generator) -> None:
    car = Car(track, StubBrain(INPUTS, rng), CarConfig())
    readings = car.sense()
    assert readings.shape == (INPUTS,)
    assert np.all((readings >= 0.0) & (readings <= 1.0))
    assert len(car.wall_readings) == CarConfig().num_sensors

    car.x, car.y = 1.0, 1.0
    car.update()
    assert not car.alive


def test_idle_car_is_killed(track: Track, rng: np.random.Generator) -> None:
    cfg = CarConfig(idle_frames_allowed=5)
    car = Car(track, StubBrain(INPUTS, rng, still=True), cfg)
    for _ in range(cfg.idle_frames_allowed):
        car.update()
    assert not car.alive


def test_fitness_rewards_the_weakest_circuit_not_the_sum() -> None:
    """A specialist must not beat an all-rounder with a worse total."""
    from src.config import SimulationConfig
    from src.simulation import Simulation

    simulation = Simulation(SimulationConfig(), render=False)
    #                     specialist   all-rounder
    scores = np.array([[3.4, 2.0],   # easy circuit
                       [3.3, 1.9],   # easy circuit
                       [0.2, 1.7]])  # hard circuit
    assert scores[:, 0].sum() > scores[:, 1].sum()  # specialist wins on the sum

    fitness = simulation._aggregate(scores)
    assert fitness[1] > fitness[0], "the all-rounder must win on the real fitness"


def test_clearance_field_never_overestimates(track: Track) -> None:
    """Sphere tracing is only safe if the clearance is a true lower bound.

    A ray jumps forward by the clearance at its current point, so if that value
    ever exceeded the real distance to the verge, a ray could jump straight
    through a wall and report open road beyond it.
    """
    rng = np.random.default_rng(5)
    for _ in range(60):
        index = int(rng.integers(0, len(track.samples)))
        x, y = track.samples[index]
        clearance = track.clearance_at(x, y)
        if clearance <= 1.0:
            continue
        # Everything strictly inside that radius must still be road.
        for angle in np.linspace(0, 2 * np.pi, 16, endpoint=False):
            probe = (x + np.cos(angle) * (clearance - 1), y + np.sin(angle) * (clearance - 1))
            assert track.is_on_road(*probe), f"clearance {clearance:.1f} overestimated at {x},{y}"


def test_ray_stops_at_the_verge(track: Track, rng: np.random.Generator) -> None:
    """Whatever the stepping strategy, a ray must land on the edge of the road."""
    car = Car(track, StubBrain(INPUTS, rng), CarConfig())
    for angle in np.linspace(0, 2 * np.pi, 40, endpoint=False):
        distance, _ = car._cast_wall_ray(float(np.cos(angle)), float(np.sin(angle)))
        if distance >= car.cfg.sensor_range:
            continue
        just_before = (
            car.x + np.cos(angle) * (distance - 2),
            car.y + np.sin(angle) * (distance - 2),
        )
        assert track.is_on_road(*just_before)


def test_every_crossover_mode_produces_a_legal_child(rng: np.random.Generator) -> None:
    """Whatever the operator, a child is the right size and made of its parents."""
    a, b = np.zeros(60), np.ones(60)
    for mode in ("uniform", "one-point", "two-point", "none"):
        child = crossover(a, b, rng, mode)
        assert child.size == a.size, mode
        assert set(np.unique(child)) <= {0.0, 1.0}, mode


def test_run_crossovers_keep_stretches_of_one_parent(rng: np.random.Generator) -> None:
    """The point of cutting rather than sampling: co-adapted groups survive.

    Uniform crossover changes parent at roughly every second gene, which splits
    a neuron's incoming weights between both parents and hands the child a unit
    that detects neither thing. A one-point cut switches once.
    """
    a, b = np.zeros(400), np.ones(400)
    switches = lambda g: int(np.count_nonzero(np.diff(g)))  # noqa: E731
    assert switches(crossover(a, b, rng, "one-point")) <= 1
    assert switches(crossover(a, b, rng, "two-point")) <= 2
    assert switches(crossover(a, b, rng, "uniform")) > 100


def test_mutation_only_leaves_recombination_out() -> None:
    """`none` must hand back a parent untouched, not a blend of two."""
    rng = np.random.default_rng(3)
    a, b = np.zeros(50), np.ones(50)
    assert np.array_equal(crossover(a, b, rng, "none"), a)


def test_an_unknown_crossover_mode_is_refused(rng: np.random.Generator) -> None:
    with pytest.raises(ValueError, match="Unknown crossover"):
        crossover(np.zeros(10), np.ones(10), rng, "telepathy")
