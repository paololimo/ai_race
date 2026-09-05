"""Smoke tests for the network, the genetic operators and the track."""

import numpy as np
import pygame
import pytest

from src.config import CarConfig, GeneticConfig, NetworkConfig, track_variants
from src.car import Car, network_input_size
from src.genetic import crossover, mutate, next_generation, select_breeding_pool
from src.neural_network import NeuralNetwork
from src.track import Track
from src.track_check import widest_corridor

INPUTS = network_input_size(CarConfig())


@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture(scope="module")
def track() -> Track:
    pygame.init()
    return Track(track_variants()[0])


def test_forward_shape_and_range(rng: np.random.Generator) -> None:
    net = NeuralNetwork(INPUTS, NetworkConfig(), rng)
    out = net.forward(np.ones(INPUTS))
    assert out.shape == (2,)
    assert np.all(np.abs(out) <= 1.0)


def test_genome_roundtrip(rng: np.random.Generator) -> None:
    net = NeuralNetwork(INPUTS, NetworkConfig(), rng)
    genome = net.get_genome()
    h1, h2 = NetworkConfig().hidden_sizes
    assert genome.size == INPUTS * h1 + h1 * h2 + h2 * 2 + h1 + h2 + 2
    assert INPUTS == CarConfig().num_sensors + 1  # one ray each, plus speed
    inputs = rng.random(INPUTS)
    expected = net.forward(inputs)
    net.set_genome(genome)
    assert np.allclose(net.forward(inputs), expected)


def test_set_genome_rejects_wrong_size(rng: np.random.Generator) -> None:
    net = NeuralNetwork(INPUTS, NetworkConfig(), rng)
    with pytest.raises(ValueError):
        net.set_genome(np.zeros(3))


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
    assert track.is_on_road(*track.start_position())
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
    car = Car(track, NeuralNetwork(INPUTS, NetworkConfig(), rng), CarConfig())
    # Teleport forward along the centreline; progress must follow the arc length.
    for step in range(1, 20):
        car.x, car.y = track.samples[step * 5]
        car._update_progress()
    assert car.progress == pytest.approx(95 * track.cfg.sample_spacing, abs=1.0)


def test_car_senses_and_dies_off_road(track: Track, rng: np.random.Generator) -> None:
    car = Car(track, NeuralNetwork(INPUTS, NetworkConfig(), rng), CarConfig())
    readings = car.sense()
    assert readings.shape == (INPUTS,)
    assert np.all((readings >= 0.0) & (readings <= 1.0))
    assert len(car.wall_readings) == CarConfig().num_sensors

    car.x, car.y = 1.0, 1.0
    car.update()
    assert not car.alive


def test_idle_car_is_killed(track: Track, rng: np.random.Generator) -> None:
    cfg = CarConfig(idle_frames_allowed=5)
    car = Car(track, NeuralNetwork(INPUTS, NetworkConfig(), rng), cfg)
    car.brain.weights = [np.zeros_like(w) for w in car.brain.weights]
    car.brain.biases = [np.zeros_like(b) for b in car.brain.biases]
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


def test_saved_model_carries_its_own_architecture(tmp_path) -> None:
    """A checkpoint must stay usable after the project default changes."""
    from dataclasses import replace as dc_replace

    from src.config import SimulationConfig
    from src.simulation import Simulation

    cfg = SimulationConfig()
    trainer = Simulation(cfg, render=False)
    odd = dc_replace(cfg.network, hidden_sizes=(9, 7))  # not the current default
    brain = NeuralNetwork(trainer.input_size, odd, np.random.default_rng(0))
    trainer.best_genome = brain.get_genome()
    trainer.best_fitness = 1.23
    trainer.cfg = dc_replace(cfg, network=odd, checkpoint_dir=str(tmp_path))
    path = trainer.save_best()

    # A fresh simulation on the default architecture must still load it.
    loader = Simulation(cfg, render=False)
    genome, network = loader._load_genome(path)
    assert network.hidden_sizes == (9, 7)
    assert genome.size == brain.genome_size


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
    car = Car(track, NeuralNetwork(INPUTS, NetworkConfig(), rng), CarConfig())
    for angle in np.linspace(0, 2 * np.pi, 40, endpoint=False):
        distance, _ = car._cast_wall_ray(float(np.cos(angle)), float(np.sin(angle)))
        if distance >= car.cfg.sensor_range:
            continue
        just_before = (
            car.x + np.cos(angle) * (distance - 2),
            car.y + np.sin(angle) * (distance - 2),
        )
        assert track.is_on_road(*just_before)


def test_parallel_evaluation_matches_serial() -> None:
    """Using more cores must change the speed and nothing else.

    Guards a real bug: building a network draws random weights that are then
    overwritten, and while those draws came from the run's own generator, doing
    the building in workers left the parent's generator in a different state —
    so start points and mutations diverged and the two paths gave different
    answers.
    """
    from dataclasses import replace as dc_replace

    from src.config import SimulationConfig
    from src.simulation import Simulation

    base = SimulationConfig()
    cfg = dc_replace(
        base,
        generations=2,
        island_variants=1,
        genetic=dc_replace(base.genetic, population_size=8),
    )

    serial = Simulation(cfg, render=False, workers=1)
    serial.train()
    parallel_run = Simulation(cfg, render=False, workers=2)
    parallel_run.train()

    assert parallel_run.best_fitness == pytest.approx(serial.best_fitness, abs=1e-12)
    assert parallel_run.history == pytest.approx(serial.history, abs=1e-12)
