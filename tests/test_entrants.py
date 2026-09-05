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
    """Dropping a file in is the whole installation — nothing else to edit.

    No entrant is named here on purpose: the repository is handed to competing
    agents with the other entries removed, so the suite has to pass on a copy
    holding nothing but the reader's own file.
    """
    found = entrants()
    assert found, "no entrants discovered in src/brains/"
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


def test_every_entrant_can_be_drawn() -> None:
    """`describe()` is required: an entry nobody can see is half a submission.

    The dashboard falls back to a measured input-to-output response for a brain
    without it, so nothing breaks — but every entry looks alike in the fallback,
    which hides exactly the differences the race is about.
    """
    rng = np.random.default_rng(0)
    missing, wrong = [], []
    for name in entrants():
        brain = build_brain(BrainRef(name), INPUTS, rng)
        describe = getattr(brain, "describe", None)
        if not callable(describe):
            missing.append(name)
            continue
        topology = describe()
        if not topology.layers:
            wrong.append(f"{name}: described no layers")
        for source, target, weights in topology.edges:
            shape = np.asarray(weights).shape
            expected = (topology.layers[source][1], topology.layers[target][1])
            if shape != expected:
                wrong.append(f"{name}: edge {source}->{target} is {shape}, not {expected}")

    assert not missing, f"no describe(): {', '.join(missing)}"
    assert not wrong, "; ".join(wrong)


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

    standing = brains.entrants()
    broken = tmp_path / "wreck.py"
    broken.write_text("this is not python(")
    brains.__path__.append(str(tmp_path))
    brains._DISCOVERED = False
    try:
        found = brains.entrants()
        assert "wreck" not in found
        assert found == standing, "a broken file must not cost anyone else their place"
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


def test_the_panel_draws_every_entrant(tmp_path) -> None:
    """Structure when a brain describes itself, measured response when not.

    The old panel read `layer_sizes` and `weights` straight off the brain, which
    only the project's own entrant has — every competitor took the window down
    on the first frame it led.
    """
    import pygame

    from src.dashboard import Dashboard, Entry

    pygame.init()
    described, measured = 0, 0
    rng = np.random.default_rng(0)
    entries = []
    for name in entrants():
        brain = build_brain(BrainRef(name), INPUTS, rng)
        if callable(getattr(brain, "describe", None)):
            topology = brain.describe()
            assert topology.layers, f"{name} described no layers"
            for source, target, weights in topology.edges:
                rows, cols = np.asarray(weights).shape
                assert rows == topology.layers[source][1], f"{name}: edge misses its source"
                assert cols == topology.layers[target][1], f"{name}: edge misses its target"
            described += 1
        else:
            measured += 1
        entries.append(Entry(name, color_of(name), 0.5, 1, 8, 1.0, brain))

    assert described + measured == len(entrants())
    panel = Dashboard(380, 700, INPUTS).render(
        generation=1, track_name="serpentine", track_number=1, track_count=3,
        frame=1, max_frames=100, entries=entries, curves=[(e.name, e.color, [0.1, 0.2]) for e in entries],
    )
    assert panel.get_size() == (380, 700)


def test_response_reads_a_brain_without_touching_its_weights() -> None:
    """The fallback view must work on any architecture, including opaque ones."""
    from src.dashboard import response

    class Opaque:
        genome_size = 2

        def __init__(self):
            self.g = np.array([0.7, -0.4])

        def forward(self, inputs):
            return np.tanh(np.array([self.g[0] * inputs[0], self.g[1] * inputs[3]]))

        def get_genome(self):
            return self.g

        def set_genome(self, genome):
            self.g = genome

    jacobian = response(Opaque(), INPUTS)
    assert jacobian.shape == (INPUTS, 2)
    assert jacobian[0, 0] > 0, "should see the positive dependence on ray 0"
    assert jacobian[3, 1] < 0, "should see the negative dependence on ray 3"
    assert abs(jacobian[5, 0]) < 1e-9, "should see no dependence where there is none"


def test_a_brain_that_cannot_be_drawn_does_not_take_the_window_down() -> None:
    """`describe()` is competitor code, so the panel must survive it raising.

    The registry already refuses to let a file that fails to import cost anyone
    the race; a panel that dies on the frame that entrant happens to lead is the
    same failure arriving later, and mid-run.
    """
    import pygame

    from src.dashboard import Dashboard, Entry

    pygame.init()

    class Exploding:
        genome_size = 1

        def describe(self):
            raise RuntimeError("boom")

        def forward(self, inputs):
            raise RuntimeError("boom")

        def get_genome(self):
            return np.zeros(1)

        def set_genome(self, genome):
            pass

    panel = Dashboard(380, 700, INPUTS)
    entries = [Entry("wreck", (200, 0, 0), 0.0, 1, 1, 0.0, Exploding())]
    surface = panel.render(
        generation=1, track_name="serpentine", track_number=1, track_count=3,
        frame=1, max_frames=100, entries=entries, curves=[("wreck", (200, 0, 0), [0.1, 0.2])],
    )
    assert surface.get_size() == (380, 700)
