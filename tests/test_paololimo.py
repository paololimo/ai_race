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
    h1, h2 = NetworkConfig().hidden_sizes
    assert genome.size == INPUTS * h1 + h1 * h2 + h2 * 2 + h1 + h2 + 2
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
