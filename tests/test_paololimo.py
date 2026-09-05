"""Tests for this project's own entrant.

Kept apart from the rest of the suite because the repository is handed to
competing agents with every entrant file removed but their own: nothing outside
this file may import `src.brains.paololimo`, or that copy will not even collect.
"""

from dataclasses import replace

import numpy as np
import pytest

from src.brains import BrainRef
from src.brains.paololimo import NetworkConfig, NeuralNetwork, spec_from_config
from src.car import network_input_size
from src.config import CarConfig, SimulationConfig
from src.simulation import Simulation

INPUTS = network_input_size(CarConfig())
SPEC = spec_from_config(NetworkConfig())


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


def test_forward_shape_and_range(rng: np.random.Generator) -> None:
    net = NeuralNetwork(INPUTS, SPEC, rng)
    out = net.forward(np.ones(INPUTS))
    assert out.shape == (2,)
    assert np.all(np.abs(out) <= 1.0)


def test_genome_roundtrip(rng: np.random.Generator) -> None:
    net = NeuralNetwork(INPUTS, SPEC, rng)
    genome = net.get_genome()
    cfg = NetworkConfig()
    widths = (INPUTS, *cfg.hidden_sizes, 2)
    # A weight matrix and a bias per step of the chain, plus the skip's own
    # matrix straight from the inputs to the controls.
    expected = sum(a * b + b for a, b in zip(widths, widths[1:]))
    if cfg.skip:
        expected += INPUTS * 2
    assert genome.size == expected
    assert INPUTS == CarConfig().num_sensors + 1  # one ray each, plus speed
    inputs = rng.random(INPUTS)
    expected = net.forward(inputs)
    net.set_genome(genome)
    assert np.allclose(net.forward(inputs), expected)


def test_set_genome_rejects_wrong_size(rng: np.random.Generator) -> None:
    net = NeuralNetwork(INPUTS, SPEC, rng)
    with pytest.raises(ValueError):
        net.set_genome(np.zeros(3))


def test_symmetry_is_exact_when_asked_for(rng: np.random.Generator) -> None:
    """Steering must be antisymmetric and throttle symmetric under mirroring."""
    net = NeuralNetwork(INPUTS, spec_from_config(NetworkConfig(symmetric=True)), rng)
    sensors = np.array([0.1, 0.3, 0.5, 0.4, 0.9, 0.7, 0.2, 0.6])
    mirrored = np.concatenate([sensors[6::-1], sensors[7:]])
    direct, reflected = net.forward(sensors), net.forward(mirrored)
    assert direct[0] == pytest.approx(-reflected[0])
    assert direct[1] == pytest.approx(reflected[1])


def test_describe_matches_the_real_weights(rng: np.random.Generator) -> None:
    """The panel draws what `describe` says, so it has to be the truth."""
    net = NeuralNetwork(INPUTS, SPEC, rng)
    topology = net.describe()
    assert [size for _, size, _ in topology.layers] == list(net.layer_sizes)
    for source, target, weights in topology.edges:
        assert weights.shape == (topology.layers[source][1], topology.layers[target][1])


def test_saved_model_carries_its_own_architecture(tmp_path) -> None:
    """A checkpoint must stay usable after the entrant's defaults change.

    It records the architecture it was trained with, not just the layer sizes:
    `symmetric` changes what the same weights compute, so dropping it raced a
    model nobody had trained.
    """
    cfg = replace(SimulationConfig(), checkpoint_dir=str(tmp_path))
    trainer = Simulation(cfg, render=False)
    squad = next(s for s in trainer.squads if s.name == "paololimo")
    odd = NetworkConfig(hidden_sizes=(9, 7), symmetric=True)  # not the defaults
    squad.brain = BrainRef("paololimo", spec_from_config(odd))
    brain = NeuralNetwork(trainer.input_size, spec_from_config(odd), np.random.default_rng(0))
    squad.best_genome = brain.get_genome()
    squad.best_fitness = 1.23
    trainer.save_all()

    loader = Simulation(cfg, render=False)
    fresh = next(s for s in loader.squads if s.name == "paololimo")
    genome = loader.load_champion(fresh)
    assert tuple(fresh.brain.spec["hidden_sizes"]) == (9, 7)
    assert fresh.brain.spec["symmetric"] is True
    assert genome.size == brain.genome_size


VARIATIONS = (
    NetworkConfig(),
    NetworkConfig(skip=True),
    NetworkConfig(decoupled=True),
    NetworkConfig(skip=True, decoupled=True),
    NetworkConfig(hidden_sizes=(9,), skip=True, decoupled=True, symmetric=True),
    NetworkConfig(hidden_sizes=()),
)


@pytest.mark.parametrize("cfg", VARIATIONS)
def test_every_variation_drives_and_round_trips(cfg: NetworkConfig) -> None:
    """Each architectural option has to be a working network on its own."""
    rng = np.random.default_rng(1)
    net = NeuralNetwork(INPUTS, spec_from_config(cfg), rng)
    sensors = rng.random(INPUTS)

    out = net.forward(sensors)
    assert out.shape == (2,)
    assert np.all(np.abs(out) <= 1.0)

    genome = net.get_genome()
    assert genome.ndim == 1 and genome.size == net.genome_size
    net.set_genome(genome)
    assert np.allclose(net.forward(sensors), out)


@pytest.mark.parametrize("cfg", VARIATIONS)
def test_every_variation_describes_itself_truthfully(cfg: NetworkConfig) -> None:
    """The panel draws `describe()`, so it must match the real matrices."""
    net = NeuralNetwork(INPUTS, spec_from_config(cfg), np.random.default_rng(2))
    topology = net.describe()
    assert topology.layers
    for source, target, weights in topology.edges:
        assert weights.shape == (topology.layers[source][1], topology.layers[target][1])


def test_decoupling_keeps_the_controls_apart() -> None:
    """The point of separate stacks: a change to one control cannot move the other."""
    net = NeuralNetwork(INPUTS, spec_from_config(NetworkConfig(decoupled=True)), np.random.default_rng(3))
    sensors = np.linspace(0.2, 0.8, INPUTS)
    before = net.forward(sensors)

    genome = net.get_genome()
    half = genome.size // 2  # the steering stack comes first
    genome[:half] += 0.5
    net.set_genome(genome)
    after = net.forward(sensors)

    assert after[0] != pytest.approx(before[0]), "steering should have moved"
    assert after[1] == pytest.approx(before[1]), "throttle must not have moved"


def test_a_skip_is_a_real_extra_path() -> None:
    """The skip must change the output, not sit there unused."""
    spec = spec_from_config(NetworkConfig(skip=True))
    net = NeuralNetwork(INPUTS, spec, np.random.default_rng(4))
    sensors = np.linspace(0.1, 0.9, INPUTS)
    with_skip = net.forward(sensors)

    genome = net.get_genome()
    net.stacks[0].skip[:] = 0.0
    without = net.forward(sensors)
    assert not np.allclose(with_skip, without)

    net.set_genome(genome)  # and it must come back through the genome
    assert np.allclose(net.forward(sensors), with_skip)


def test_the_default_fits_the_search_budget() -> None:
    """The design is a bet on the budget, so pin what the bet actually is.

    100 genomes over 120 generations is 12 000 evaluations, and a
    derivative-free search wants roughly 100 to 1000 of them per parameter. The
    previous default asked for 366 parameters, which needs 37 000 at the most
    generous end; this one asks for 160. If someone widens the network again
    without widening the budget, this is the test that should stop them.
    """
    cfg = NetworkConfig()
    net = NeuralNetwork(INPUTS, spec_from_config(cfg), np.random.default_rng(0))

    assert net.genome_size == 160
    # A funnel: wider than the input to form features across the ray fan, then
    # narrow to compose them. Depth is affordable only at these widths.
    assert cfg.hidden_sizes == (10, 4)
    assert cfg.hidden_sizes[0] > INPUTS > cfg.hidden_sizes[1]
    assert cfg.skip, "the linear path evolution starts from"
    assert cfg.symmetric, "the mirror symmetry the circuits actually have"
    assert not cfg.decoupled, "a stack per control doubles the genome"

    evaluations = 100 * 120
    assert evaluations / net.genome_size >= 60, "too many parameters for the budget"
