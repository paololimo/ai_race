"""The race harness: the contract a competing architecture has to satisfy.

A brain arriving from another agent is untrusted code held to a fixed
interface, and a checkpoint that fails to describe itself silently races a
model nobody trained. These test that, not the getters.
"""

from dataclasses import replace

import numpy as np
import pytest

from src.brains import Brain, BrainRef, build_brain, register_brain
from src.brains.baseline import config_from_spec, spec_from_config
from src.config import NetworkConfig, SimulationConfig
from src.grand_prix import GrandPrix, build_grid, load_entrant
from src.simulation import Simulation

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


def test_checkpoint_preserves_symmetry(tmp_path) -> None:
    """Saving only the layer sizes raced a model that was never trained.

    `symmetric` changes what the same weights compute — the network is averaged
    against its own mirror image — so a checkpoint that drops it loads a
    different driver than the one evolution selected.
    """
    cfg = SimulationConfig()
    symmetric = replace(cfg.network, symmetric=True)
    trainer = Simulation(replace(cfg, network=symmetric, checkpoint_dir=str(tmp_path)), render=False)
    trainer.best_genome = build_brain(
        trainer.brain_ref(), trainer.input_size, np.random.default_rng(0)
    ).get_genome()
    trainer.best_fitness = 1.0
    path = trainer.save_best()

    # The loader's own default is symmetric=False: only the manifest can save it.
    loader = Simulation(replace(cfg, checkpoint_dir=str(tmp_path)), render=False)
    _, ref = loader._load_genome(path)
    assert ref.spec["symmetric"] is True

    sensors = np.linspace(0.1, 0.9, trainer.input_size)
    trained = build_brain(trainer.brain_ref(), trainer.input_size, np.random.default_rng(0))
    trained.set_genome(trainer.best_genome)
    raced = build_brain(ref, loader.input_size, np.random.default_rng(0))
    raced.set_genome(trainer.best_genome)
    assert np.allclose(trained.forward(sensors), raced.forward(sensors))


def test_checkpoint_carries_the_brain_name(tmp_path) -> None:
    """A competitor's model must not be reloaded as the baseline."""
    cfg = replace(
        SimulationConfig(),
        brain=BrainRef("test-stub", {"gain": 3.0}),
        entrant="stub",
        checkpoint_dir=str(tmp_path),
    )
    trainer = Simulation(cfg, render=False)
    trainer.best_genome = build_brain(
        cfg.brain, trainer.input_size, np.random.default_rng(0)
    ).get_genome()
    trainer.best_fitness = 0.5
    path = trainer.save_best("stub_genome.npz")

    _, ref = Simulation(cfg, render=False)._load_genome(path)
    assert ref == BrainRef("test-stub", {"gain": 3.0})


def test_grid_rejects_a_checkpoint_built_for_a_different_car(tmp_path) -> None:
    """Sensor count is the car's, not the brain's: a mismatch is not raceable."""
    cfg = replace(SimulationConfig(), checkpoint_dir=str(tmp_path))
    trainer = Simulation(cfg, render=False)
    trainer.best_genome = build_brain(
        trainer.brain_ref(), trainer.input_size, np.random.default_rng(0)
    ).get_genome()
    trainer.best_fitness = 1.0
    path = trainer.save_best()
    with pytest.raises(SystemExit, match="sensor inputs"):
        load_entrant(path, trainer.input_size + 1, (1, 2, 3))


def test_grand_prix_ranks_every_entrant_once(tmp_path) -> None:
    """Heterogeneous brains must race together and come back classified."""
    cfg = replace(SimulationConfig(), laps_budget=0.05, checkpoint_dir=str(tmp_path))
    paths = []
    for name, ref in (("human", BrainRef("baseline", {})), ("stub", BrainRef("test-stub", {}))):
        trainer = Simulation(replace(cfg, brain=ref, entrant=name), render=False)
        trainer.best_genome = build_brain(
            trainer.brain_ref(), trainer.input_size, np.random.default_rng(2)
        ).get_genome()
        trainer.best_fitness = 1.0
        paths.append(trainer.save_best(f"{name}_genome.npz"))

    grand_prix = GrandPrix(cfg, render=False)
    entrants = build_grid(paths, grand_prix.input_size)
    assert len({e.color for e in entrants}) == 2, "entrants must be distinguishable"

    results = grand_prix.run(entrants)
    grand_prix.close()
    assert [r.name for r in sorted(results, key=lambda r: r.name)] == ["human", "stub"]
    assert results == sorted(results, key=lambda r: r.laps, reverse=True)


def test_an_empty_grid_is_refused() -> None:
    with pytest.raises(SystemExit, match="No entrants"):
        build_grid([], INPUTS)
