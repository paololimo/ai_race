"""The properties the whole comparison rests on.

Every entrant in `src/brains/` is trained by the same loop under the same
conditions, so these test the traps: an entrant that cannot be discovered, one
that gets a different test than the others, populations that leak into each
other, and a broken file taking the race down with it.
"""

from dataclasses import replace

import numpy as np
import pytest

from src.brains import BRAIN_FACTORY, Brain, BrainRef, build_brain, color_of, entrants
from src.car import network_input_size
from src.config import SimulationConfig
from src.simulation import Simulation

INPUTS = network_input_size(SimulationConfig().car)


def config(tmp_path, **overrides) -> SimulationConfig:
    base = SimulationConfig()
    return replace(
        base,
        seed=42,
        generations=2,
        checkpoint_dir=str(tmp_path),
        genetic=replace(base.genetic, population_size=6),
        **overrides,
    )


def test_every_file_in_brains_is_an_entrant() -> None:
    """Dropping a file in is the whole installation — nothing else to edit."""
    found = entrants()
    assert "paololimo" in found, "the project's own entrant must be discovered"
    assert len(found) == len(BRAIN_FACTORY)


def test_every_entrant_satisfies_the_contract() -> None:
    """A competitor's file is code someone else wrote: check it, don't trust it."""
    rng = np.random.default_rng(0)
    sensors = np.linspace(0.1, 0.9, INPUTS)
    for name in entrants():
        brain = build_brain(BrainRef(name), INPUTS, rng)
        assert isinstance(brain, Brain), f"{name} does not satisfy the protocol"

        out = brain.forward(sensors)
        assert out.shape == (2,), f"{name} returned {out.shape}, not steering and throttle"
        assert np.all(np.abs(out) <= 1.0), f"{name} drove outside [-1, 1]"

        genome = brain.get_genome()
        assert genome.ndim == 1, f"{name}'s genome is not flat"
        assert genome.size == brain.genome_size
        brain.set_genome(genome)
        assert np.allclose(brain.forward(sensors), out), f"{name} does not round-trip"


def test_entrants_are_told_apart_by_colour() -> None:
    assert len({color_of(name) for name in entrants()}) == len(entrants())


def test_every_squad_faces_the_identical_stage(tmp_path) -> None:
    """One draw per evaluation, shared by everyone.

    Training entrants in separate runs made them face the same circuits only
    because both runs derived them from the seed — and that broke silently once
    genome size changed how many numbers had been drawn.
    """
    simulation = Simulation(config(tmp_path), render=False)
    for generation in range(1, 4):
        for number in range(len(simulation.circuits)):
            stage = simulation._stage(number, generation)
            cars = [simulation._car(stage, s, s.genomes[0]) for s in simulation.squads]
            assert len({(c.x, c.y, c.angle) for c in cars}) == 1, "squads started apart"
            assert len({id(c.track) for c in cars}) == 1, "squads met different circuits"


def test_populations_never_mix(tmp_path) -> None:
    """Crossing architectures is meaningless, so each entrant breeds alone."""
    simulation = Simulation(config(tmp_path), render=False)
    before = [{g.size for g in s.genomes} for s in simulation.squads]
    assert all(len(sizes) == 1 for sizes in before)
    simulation.train()
    assert [{g.size for g in s.genomes} for s in simulation.squads] == before


def test_discovery_order_cannot_change_a_result(tmp_path) -> None:
    """Each entrant draws from its own stream, so listing order is irrelevant."""
    simulation = Simulation(config(tmp_path), render=False)
    for squad in simulation.squads:
        rebuilt = np.random.default_rng(simulation.cfg.seed)
        first = build_brain(squad.brain, simulation.input_size, rebuilt).get_genome()
        assert np.array_equal(squad.genomes[0], first)


def test_scoring_on_workers_matches_one_process(tmp_path) -> None:
    """The pool must change when a number arrives, never what it is.

    Guards a real bug: building a brain draws random weights that are then
    overwritten, and while those draws came from the run's own generator, doing
    the building in workers left the parent's generator in a different state —
    so start points and mutations diverged and the two paths gave different
    answers.
    """
    serial = Simulation(config(tmp_path), render=False, workers=1)
    serial.train()
    pooled = Simulation(config(tmp_path), render=False, workers=3)
    try:
        pooled.train()
        for one, many in zip(serial.squads, pooled.squads):
            assert one.history == pytest.approx(many.history)
    finally:
        pooled.close()


def test_training_then_racing_is_the_whole_loop(tmp_path) -> None:
    """What training writes has to be exactly what the race reads."""
    simulation = Simulation(config(tmp_path, laps_budget=0.05), render=False)
    simulation.train()
    assert len(simulation.save_all()) == len(simulation.squads)

    day = Simulation(config(tmp_path, laps_budget=0.05), render=False)
    results = day.race()
    assert {name for name, *_ in results} == {s.name for s in day.squads}
    assert results == sorted(results, key=lambda r: r[1], reverse=True)


def test_racing_without_champions_says_so(tmp_path) -> None:
    with pytest.raises(SystemExit, match="run train.py first"):
        Simulation(config(tmp_path), render=False).race()


def test_a_broken_entrant_does_not_take_the_race_down(tmp_path) -> None:
    """One competitor's syntax error must not stop everyone else racing."""
    import importlib

    import src.brains as brains

    broken = tmp_path / "wreck.py"
    broken.write_text("this is not python(")
    brains.__path__.append(str(tmp_path))
    brains._DISCOVERED = False
    try:
        found = brains.entrants()
        assert "wreck" not in found
        assert "paololimo" in found
    finally:
        brains.__path__.remove(str(tmp_path))
        brains._DISCOVERED = False
        importlib.invalidate_caches()


def test_one_class_cannot_take_two_places_on_the_grid() -> None:
    """A second name would mean two colours and two entries, competing alone."""
    from src.brains import register_brain

    @register_brain("twin-a")
    class Twin:
        genome_size = 1

        def __init__(self, input_size, spec, rng):
            self.g = np.zeros(1)

        def forward(self, inputs):
            return np.zeros(2)

        def get_genome(self):
            return self.g

        def set_genome(self, genome):
            self.g = genome

    register_brain("twin-b")(Twin)
    try:
        assert "twin-b" not in BRAIN_FACTORY
        assert BRAIN_FACTORY["twin-a"] is Twin
    finally:
        BRAIN_FACTORY.pop("twin-a", None)
