"""The brain registry: the contract a competing architecture has to satisfy.

A brain arriving from another agent is untrusted code held to a fixed
interface, so these test the interface, not the getters.
"""

import numpy as np
import pytest

from src.brains import Brain, BrainRef, build_brain, register_brain
from src.brains.baseline import config_from_spec, spec_from_config
from src.config import NetworkConfig

INPUTS = 8


@register_brain("test-stub")
class StubBrain:
    """A minimal legal brain: no hidden layer, a single weight matrix."""

    def __init__(self, input_size, spec, rng):
        self.gain = float(spec.get("gain", 1.0))
        self.w = rng.normal(0.0, 0.5, size=(input_size, 2))

    @property
    def genome_size(self) -> int:
        return self.w.size

    def forward(self, inputs):
        return np.tanh(inputs @ self.w * self.gain)

    def get_genome(self):
        return self.w.ravel()

    def set_genome(self, genome):
        self.w = genome.reshape(self.w.shape)


def test_baseline_is_registered_and_satisfies_the_protocol() -> None:
    brain = build_brain(BrainRef("baseline", {}), INPUTS, np.random.default_rng(0))
    assert isinstance(brain, Brain)


def test_unknown_brain_names_the_ones_that_exist() -> None:
    with pytest.raises(SystemExit, match="baseline"):
        build_brain(BrainRef("nonexistent", {}), INPUTS, np.random.default_rng(0))


def test_a_third_party_brain_drives_the_same_car() -> None:
    """The whole point of the registry: an outside brain needs no other change."""
    brain = build_brain(BrainRef("test-stub", {"gain": 2.0}), INPUTS, np.random.default_rng(1))
    out = brain.forward(np.zeros(INPUTS))
    assert out.shape == (2,) and np.all(np.abs(out) <= 1.0)


def test_spec_round_trips_through_a_network_config() -> None:
    """The spec is what the checkpoint will carry, so it must lose nothing."""
    cfg = NetworkConfig(hidden_sizes=(9, 7), symmetric=True, weight_init_scale=0.5)
    assert config_from_spec(spec_from_config(cfg)) == cfg
