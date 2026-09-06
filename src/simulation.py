"""Every entrant trains at once, then every champion races at once.

There is one draw per evaluation — one circuit, one island layout, one start
point — and every entrant is scored under it. Parity is therefore a property of
the loop rather than a convention someone has to remember to keep.

The populations never mix. Crossing one entrant's genome with another's is
meaningless: different sizes, and at equal size the genes would mean different
things. Each squad breeds alone; only the conditions are shared.
"""

import json
import logging
import time
import multiprocessing
import multiprocessing.pool
import os
import random
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pygame

from src import parallel
from src.brains import BrainRef, Color, build_brain, color_of, entrants
from src.car import Car, build_car, network_input_size
from src.config import SimulationConfig, race_track, track_variants
from src.analysis import Analysis, Series
from src.dashboard import Dashboard, Entry
from src.genetic import mutation_sigma, next_generation
from src.recorder import Recorder
from src.renderer import Layout, Renderer, fit_window
from src.track import Track

logger = logging.getLogger(__name__)

# entrant, laps, still running at the end, frames driven, frame it finished on
# (None if it never did). Ranking is finishers by time, then the rest by laps.
Result = Tuple[str, float, bool, int, Optional[int]]

# The countdown: what is shown, and for how many frames each.
_COUNTDOWN: Tuple[Tuple[str, int], ...] = (("3", 45), ("2", 45), ("1", 45), ("GO", 40))
_FLAG = (255, 210, 90)


def _clock(seconds: float) -> str:
    """`h:mm:ss` once it runs to hours, `m:ss` before that."""
    seconds = int(max(0.0, seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


@dataclass
class Squad:
    """One entrant's population, evolving on its own."""

    name: str
    brain: BrainRef
    color: Color
    genomes: List[np.ndarray]
    rng: np.random.Generator
    history: List[float] = field(default_factory=list)
    # Mean per-gene standard deviation of the population, one value per
    # generation. The diagnostic a genetic algorithm cannot be read without:
    # once it reaches zero the population is one genome in many copies and no
    # number of further generations buys anything.
    spread: List[float] = field(default_factory=list)
    # Best score on each circuit this generation. Fitness is `worst + 0.35 x
    # mean`, so the lowest of these is what is actually holding the entrant
    # back — and it was being aggregated away before anyone could see it.
    circuits: List[float] = field(default_factory=list)
    best_fitness: float = 0.0
    best_genome: Optional[np.ndarray] = None


@dataclass(frozen=True)
class Stage:
    """The conditions of one evaluation, shared by every entrant."""

    track: Track
    number: int
    variant: int
    start: Optional[int]
    budget: int


class Simulation:
    """The generational loop, run for every entrant at the same time."""

    def __init__(
        self,
        cfg: SimulationConfig,
        render: bool = True,
        workers: int = 1,
        shown: int = 10,
        entries: Optional[Sequence[BrainRef]] = None,
        record: Optional[Path] = None,
        record_every: int = 3,
        record_clip: Optional[float] = None,
        throttle: bool = True,
    ) -> None:
        """`entries` overrides who is on the grid.

        Left alone it is everyone in `src/brains/`, which is what `train.py` and
        `race.py` want. The architecture ablation passes an explicit list because
        it compares several configurations of one entrant against each other.
        """
        self.cfg = cfg
        set_seed(cfg.seed)
        self.workers = max(1, workers)
        # How many cars per entrant are drawn. The whole population is still
        # scored: several hundred cars on one circuit is a swarm nobody can
        # read, and the elites sit at the front of each population, so the few
        # that are drawn are the ones worth watching.
        self.shown = shown
        self._pool: Optional[multiprocessing.pool.Pool] = None

        if not render:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        pygame.init()

        self.input_size = network_input_size(cfg.car)
        # One stream for the start points, drawn once per evaluation and shared.
        # Taking them from the same generator as the genomes tied the sequence
        # to how many numbers had already been drawn, which depends on genome
        # size — so entrants of different sizes met different starts.
        self._start_rng = np.random.default_rng(cfg.seed)
        self._brain_rng = np.random.default_rng(0)  # see `_car`

        # Read from the configuration rather than from a built circuit, so a
        # run that never draws — a headless race, an ablation job — never pays
        # for the twelve tracks it is not going to use.
        variants = track_variants()
        self._circuit_names = [c.name for c in variants]
        size = (variants[0].width, variants[0].height)
        # The window is asked for more than a small screen has, and shrinks to
        # what this display actually offers rather than running off the edge.
        # A recording with no window on screen has no display to fit into, so
        # it gets the full-size layout rather than one shrunk for a monitor it
        # is never shown on.
        offscreen = record is not None and not throttle
        layout = (
            Layout(size, cfg.dashboard_width, cfg.strip_height, 1.0)
            if offscreen
            else fit_window(size, cfg.dashboard_width, cfg.strip_height)
        )
        self.renderer = Renderer(size, layout, cfg.fps) if render else None
        # The panel stands beside the circuit, so it is as tall as the circuit
        # is drawn; the strip runs under both, so it is as wide as the window.
        self.dashboard = (
            Dashboard(layout.panel, layout.view[1], self.input_size) if render else None
        )
        self.analysis = (
            Analysis(layout.window[0], layout.strip) if render and layout.strip else None
        )
        self.recorder = (
            Recorder(record, layout.window, fps=cfg.fps)
            if render and record is not None
            else None
        )
        # By default every drawn frame is kept: what most people mean by
        # recording is a recording. `record_clip` is the opt-in that turns a
        # two-hour file into a short one, and it works by dropping whole
        # generations rather than by thinning frames — thinning would replace
        # the motion with a slideshow, and speeding the file up afterwards
        # does exactly the same thing at exactly the same cost.
        self._film_every = max(1, record_every)
        self._film_frames = int(record_clip * cfg.fps) if record_clip else None
        # Without a window to watch there is no reason to hold 60 frames a
        # second, and holding it would make a recording take as long as the
        # training takes to watch.
        self._throttle = throttle
        self._racing = False  # a race shows a running order, not a search
        self._race_laps = 0.0
        self._started = time.monotonic()
        # How long each finished generation took. The estimate must not be
        # divided by the generation in progress: within one, the elapsed time
        # grows while the divisor does not, so the estimate climbs steadily and
        # only falls back at each boundary. Only finished ones count.
        self._per_generation: List[float] = []
        grid = list(entries) if entries is not None else [BrainRef(n) for n in entrants()]
        self.squads: List[Squad] = [self._recruit(ref) for ref in grid]
        if not self.squads:
            raise SystemExit("No entrants in src/brains/ — nothing to train.")
        logger.info("Entrants: %s", ", ".join(s.name for s in self.squads))

    # -- setup ---------------------------------------------------------------

    @cached_property
    def circuits(self) -> List[List[Track]]:
        """Every circuit in each of its island layouts, built on first use.

        Several island layouts per circuit, cycled through generation by
        generation, so no single arrangement can be memorised. Building the
        twelve of them costs about a second, so it is deferred: racing on
        `gauntlet` and most of the test suite never touch them.
        """
        return parallel.build_circuits(self.cfg)

    def _recruit(self, ref: BrainRef) -> Squad:
        """Draw an entrant's starting population from its own stream.

        Every stream is seeded identically, so the order entrants happen to be
        discovered in cannot change anybody's result.
        """
        rng = np.random.default_rng(self.cfg.seed)
        genomes = [
            build_brain(ref, self.input_size, rng).get_genome()
            for _ in range(self.cfg.genetic.population_size)
        ]
        return Squad(ref.name, ref, color_of(ref.name), genomes, rng)

    def _ensure_pool(self) -> Optional[multiprocessing.pool.Pool]:
        """Start the worker pool on first use, and keep it for the whole run.

        Each worker rebuilds the circuits once — they are derived from the seed,
        so every process ends up with identical geometry — and then only genomes
        travel between processes.
        """
        if self.workers <= 1:
            return None
        if self._pool is None:
            self._pool = multiprocessing.Pool(
                self.workers, initializer=parallel.worker_init, initargs=(self.cfg,)
            )
            logger.info("Scoring on %d worker processes", self.workers)
        return self._pool

    def close(self) -> None:
        if self.recorder is not None:
            self.recorder.close()  # it reports the file itself
            self.recorder = None
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

    # -- one evaluation ------------------------------------------------------

    @property
    def tracks(self) -> List[Track]:
        """One representative layout per circuit."""
        return [variants[0] for variants in self.circuits]

    def frame_budget(self, track: Track) -> int:
        """How long a car may drive this circuit, in frames.

        Proportional to lap length, so a short lap does not hand out more laps
        than a long one for the same amount of simulated time.
        """
        return int(self.cfg.laps_budget * track.lap_length / self.cfg.car.max_speed)

    def _stage(self, number: int, generation: int) -> Stage:
        """Pick the circuit, the island layout and the start point — once."""
        variants = self.circuits[number]
        variant = generation % len(variants)
        track = variants[variant]
        start = track.valid_start_index(self._start_rng) if self.cfg.random_start else None
        return Stage(track, number, variant, start, self.frame_budget(track))

    def _car(
        self,
        stage: Stage,
        squad: Squad,
        genome: np.ndarray,
        start: Optional[int] = None,
        lateral: float = 0.0,
    ) -> Car:
        """The same construction the pool workers do — see `car.build_car`."""
        return build_car(
            squad.brain,
            genome,
            stage.track,
            self.cfg.car,
            self.input_size,
            self._brain_rng,
            stage.start if start is None else start,
            lateral,
        )

    def _drive(
        self,
        crew: Sequence[Tuple[Squad, Car]],
        stage: Stage,
        generation: int,
    ) -> bool:
        """Step the cars frame by frame, drawing them. False if the user quit.

        Each car is carried with the squad that entered it, which is what the
        panel needs to colour it and to group the standings. While the pool is
        scoring the full populations, `crew` is only the cars on screen: those
        are simulated twice, here and in a worker, which costs a few dozen cars'
        arithmetic and gives identical results — every step is deterministic
        given the track, the start and the genome.
        """
        for frame in range(stage.budget):
            for _, car in crew:
                car.update()
            if self.renderer is not None:
                if self.renderer.poll_quit():
                    return False
                self._draw(crew, stage, generation, frame)
            if not any(car.alive for _, car in crew):
                break
        return True

    def _draw(
        self,
        crew: Sequence[Tuple[Squad, Car]],
        stage: Stage,
        generation: int,
        frame: int,
        banner: Optional[Tuple[str, float]] = None,
    ) -> None:
        assert self.renderer is not None and self.dashboard is not None
        self.renderer.draw_track(
            stage.track,
            [car for _, car in crew],
            [squad.color for squad, _ in crew],
            start_index=stage.start or 0,
        )
        if banner is not None:
            self.renderer.banner(banner[0], _FLAG, banner[1])
        entries = []
        for squad in self.squads:
            mine = [car for owner, car in crew if owner is squad]
            # The panel draws this squad's leader, so the diagram is of a brain
            # actually on the track rather than of last generation's champion.
            leader = max((c for c in mine if c.alive), key=lambda c: c.fitness, default=None)
            leader = leader or (mine[0] if mine else None)
            entries.append(
                Entry(
                    name=squad.name,
                    color=squad.color,
                    laps=max((c.laps for c in mine), default=0.0),
                    alive=sum(1 for c in mine if c.alive),
                    shown=len(mine),
                    best=squad.best_fitness,
                    params=len(squad.genomes[0]) if squad.genomes else 0,
                    brain=leader.brain if leader else None,
                )
            )
        series = [
            Series(s.name, s.color, s.history, s.spread, s.circuits) for s in self.squads
        ]
        panel = self.dashboard.render(
            generation=generation,
            track_name=stage.track.cfg.name,
            track_number=stage.number + 1,
            circuit_count=len(self._circuit_names),
            frame=frame,
            max_frames=stage.budget,
            entries=entries,
        )
        # In a race the three training charts have no history to draw and would
        # all read "collecting..." for its whole twenty seconds. The running
        # order is what a race has to show instead.
        order = (
            sorted(
                ((e.name, e.color, e.laps, e.alive > 0) for e in entries),
                key=lambda row: -row[2],
            )
            if self._racing
            else None
        )
        strip = (
            self.analysis.render(
                series, self._circuit_names, self._stats(generation), order, self._race_laps
            )
            if self.analysis
            else None
        )
        self.renderer.present(panel, strip, throttle=self._throttle)
        if self.recorder is not None and self._filming(generation, frame):
            self.recorder.capture(self.renderer.screen)

    def _filming(self, generation: int, frame: int) -> bool:
        """Whether this frame belongs in the video.

        Not one frame in N: at that spacing a car crosses a corner between two
        frames and the motion is gone. Whole generations are left out instead,
        and the ones kept are kept unthinned.

        With no `record_clip` there is nothing to decide: every frame is kept.
        Otherwise the last generation is still filmed to the end — every other
        clip is the first few seconds off the line, which is the right way to
        compare one generation against another but means the video would never
        once show a trained car completing anything.
        """
        if self._film_frames is None or generation >= self.cfg.generations:
            return True
        return (generation - 1) % self._film_every == 0 and frame < self._film_frames

    def _countdown(self, crew: Sequence[Tuple[Squad, Car]], stage: Stage) -> bool:
        """Hold the grid for three seconds before the flag. False if quit.

        Nothing about the result depends on it — no car is stepped — but a race
        that begins mid-frame with four cars already moving reads as a clip
        starting late. It also gives the panel a moment to be looked at before
        anything happens, and lands in the recording like a real start.
        """
        if self.renderer is None:
            return True
        for text, length in _COUNTDOWN:
            for step in range(length):
                if self.renderer.poll_quit():
                    return False
                self._draw(crew, stage, 0, 0, banner=(text, step / length))
        return True

    def _stats(self, generation: int) -> List[Tuple[str, str]]:
        """Numbers about the run that no curve here shows.

        The mutation size is a different number every generation now that it
        anneals, and it is the single knob that decides whether late
        generations can refine anything. The evaluation count is the quantity
        the whole argument about how many parameters an entrant can afford
        turns on. Neither is visible in any chart, and both are one line.
        """
        if self._racing:
            # Generations, mutation size and evaluations spent mean nothing in
            # a race; what it is and how long it has run do.
            return [
                ("distance", f"{self._race_laps:.0f} laps"),
                ("elapsed", _clock(time.monotonic() - self._started)),
            ]
        total = max(1, self.cfg.generations)
        progress = (generation - 1) / max(1, total - 1)
        done = max(0, generation - 1)
        scored = done * self.cfg.genetic.population_size * len(self.squads)
        # The mean of the last few finished generations, which is stable inside
        # a generation and follows the machine if it speeds up or slows down.
        recent = self._per_generation[-5:]
        left = sum(recent) / len(recent) * (total - done) if recent else None
        return [
            ("generation", f"{generation} / {total}"),
            ("mutation", f"{mutation_sigma(self.cfg.genetic, progress):.3f}"),
            ("evaluated", f"{scored:,}".replace(",", " ")),
            ("elapsed", _clock(time.monotonic() - self._started)),
            ("remaining", _clock(left) if left is not None else "--:--"),
        ]

    def _evaluate(self, stage: Stage, generation: int) -> Tuple[Dict[int, List[float]], bool]:
        """Score every entrant's full population under one set of conditions."""
        pool = self._ensure_pool()
        lap = stage.track.lap_length

        if pool is None:
            cars = {
                i: [self._car(stage, s, g) for g in s.genomes]
                for i, s in enumerate(self.squads)
            }
            crew = [(self.squads[i], car) for i, mine in cars.items() for car in mine]
            keep = self._drive(crew, stage, generation)
            return {i: [c.fitness / lap for c in cs] for i, cs in cars.items()}, keep

        jobs = [
            parallel.Job(
                i, stage.number, stage.variant, stage.start, stage.budget, block, squad.brain
            )
            for i, squad in enumerate(self.squads)
            for block in parallel.split(squad.genomes, self.workers)
        ]
        pending = pool.map_async(parallel.evaluate, jobs)
        keep = True
        if self.renderer is not None:
            shown = [
                (squad, self._car(stage, squad, genome))
                for squad in self.squads
                for genome in squad.genomes[: self.shown]
            ]
            keep = self._drive(shown, stage, generation)

        # `map_async` hands the results back in job order, which is squad by
        # squad and block by block — the order the populations were cut in.
        scores: Dict[int, List[float]] = {i: [] for i in range(len(self.squads))}
        for squad_index, values in pending.get():
            scores[squad_index].extend(values)
        return scores, keep

    # -- training ------------------------------------------------------------

    def train(self) -> None:
        self._started = time.monotonic()
        try:
            self._loop()
        except KeyboardInterrupt:
            # Ctrl+C during a recorded run would otherwise leave the video
            # unfinished, and a run is hours long: stopping it early has to be
            # allowed to keep what it has.
            logger.info("Stopped by hand — saving what there is")
        self.save_all()
        self.close()
        pygame.quit()

    def _loop(self) -> None:
        for generation in range(1, self.cfg.generations + 1):
            began = time.monotonic()
            # How far through the run we are, which is what the mutation size is
            # annealed against. Shared by every squad, like everything else the
            # comparison rests on.
            progress = (generation - 1) / max(1, self.cfg.generations - 1)
            per_track: Dict[int, List[List[float]]] = {i: [] for i in range(len(self.squads))}
            for number in range(len(self.circuits)):
                stage = self._stage(number, generation)
                scores, keep = self._evaluate(stage, generation)
                for i, values in scores.items():
                    per_track[i].append(values)
                if not keep:
                    logger.info("Interrupted at generation %d", generation)
                    return

            for i, squad in enumerate(self.squads):
                totals = self._aggregate(np.array(per_track[i]))
                best = int(np.argmax(totals))
                if totals[best] > squad.best_fitness:
                    squad.best_fitness = float(totals[best])
                    squad.best_genome = squad.genomes[best]
                squad.history.append(float(totals[best]))
                # Measured on the population that was just scored, before
                # breeding replaces it.
                squad.spread.append(float(np.std(np.asarray(squad.genomes), axis=0).mean()))
                squad.circuits = [float(np.max(scores)) for scores in per_track[i]]
                squad.genomes = next_generation(
                    squad.genomes, list(totals), self.cfg.genetic, squad.rng, progress
                )

            self._per_generation.append(time.monotonic() - began)
            logger.info(
                "gen %3d | %s",
                generation,
                "   ".join(f"{s.name} {s.history[-1]:5.2f}" for s in self.squads),
            )

    def _aggregate(self, per_track_scores: np.ndarray) -> np.ndarray:
        """Combine one genome's scores across the circuits into one fitness.

        Summing them lets a genome ignore the hardest circuit: braking enough to
        survive it costs more laps on the easy ones than it gains on the hard
        one, so "flat out everywhere" wins on the sum while dying instantly on
        anything tight. Leading with the *worst* circuit makes competence
        everywhere the only way up; the mean stays in as a tie-breaker, and as
        the signal that guides early generations, when every worst score is 0.
        """
        return per_track_scores.min(axis=0) + self.cfg.mean_weight * per_track_scores.mean(axis=0)

    # -- checkpoints ---------------------------------------------------------

    def save_all(self) -> List[Path]:
        """Write one champion per entrant, which is what `race.py` reads."""
        out_dir = Path(self.cfg.checkpoint_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        written = []
        for squad in self.squads:
            if squad.best_genome is None:
                continue
            path = out_dir / f"{squad.name}.npz"
            np.savez(
                path,
                genome=squad.best_genome,
                fitness=squad.best_fitness,
                input_size=self.input_size,
                # The entrant's name and its full spec, so the checkpoint can be
                # rebuilt exactly. Saving only the layer sizes dropped every
                # other choice an entrant makes, some of which change what the
                # same weights compute — and raced a model nobody trained.
                brain=squad.brain.name,
                spec=json.dumps(dict(squad.brain.spec)),
                seed=self.cfg.seed,
                generations=self.cfg.generations,
                population=self.cfg.genetic.population_size,
                history=np.array(squad.history),
            )
            logger.info("%-12s %5.2f  ->  %s", squad.name, squad.best_fitness, path)
            written.append(path)
        return written

    def load_champion(self, squad: Squad) -> Optional[np.ndarray]:
        """Read this entrant's saved champion, if it has one."""
        path = Path(self.cfg.checkpoint_dir) / f"{squad.name}.npz"
        if not path.exists():
            return None
        data = np.load(path, allow_pickle=False)
        saved = int(data["input_size"])
        if saved != self.input_size:
            raise SystemExit(
                f"{path.name} expects {saved} sensor inputs but the car now produces "
                f"{self.input_size}. The sensor layout changed — retrain."
            )
        squad.brain = BrainRef(str(data["brain"]), json.loads(str(data["spec"])))
        squad.best_fitness = float(data["fitness"])
        return data["genome"]

    # -- the race ------------------------------------------------------------

    def race(self, track: Optional[Track] = None, laps: float = 3.0) -> List[Result]:
        """Race every champion over `laps` laps; return the classification.

        The circuit is `gauntlet` unless told otherwise: nobody trained on it, so
        finishing it shows general driving rather than a memorised layout.

        It ends at the flag rather than at the clock. Ranking by distance
        covered in a fixed number of frames picks the same winner as a real race
        only while nobody crashes, which is not the usual case; first past a
        fixed distance is the question anyone watching is actually asking.
        """
        self._film_frames = None  # one lap of one circuit: nothing to leave out
        self._racing = True
        self._race_laps = laps
        circuit = track or Track(race_track())
        # One fixed test: no island rotation, no random start. The time limit is
        # twice what the distance takes at top speed — cars average well under
        # half of it through corners, so the flag is what ends a race and the
        # clock only catches a field that has all crashed. Deriving it from the
        # training budget instead made a short race absurdly short: at 0.05 laps
        # it came out as thirty frames.
        limit = int(2.0 * laps * circuit.lap_length / self.cfg.car.max_speed)
        stage = Stage(circuit, 0, 0, None, max(60, limit))

        entered = [
            (squad, genome)
            for squad in self.squads
            for genome in [self.load_champion(squad)]
            if genome is not None or logger.warning(
                "%s has no champion yet and sits this one out", squad.name
            )
        ]
        if not entered:
            raise SystemExit("No champions in outputs/ — run train.py first.")

        # A grid across the carriageway rather than a queue along it. Every car
        # starts on the same centreline point, so the one in front on screen is
        # the one in front in the results — which staggering them along the lap
        # made false, since a car 72 px back could be ahead on distance covered.
        room = circuit.road_width_at_index(0) / 2 - self.cfg.car.width
        slots = len(entered)
        crew: List[Tuple[Squad, Car]] = [
            (
                squad,
                self._car(
                    stage,
                    squad,
                    genome,
                    start=0,
                    lateral=0.0 if slots < 2 else room * (2 * i / (slots - 1) - 1),
                ),
            )
            for i, (squad, genome) in enumerate(entered)
        ]

        logger.info(
            "Race on '%s', never trained on, %.0f laps: %s",
            stage.track.cfg.name,
            laps,
            ", ".join(squad.name for squad, _ in crew),
        )
        for _, car in crew:
            car.finish_laps = laps
        if not self._countdown(crew, stage):
            pygame.quit()
            return []
        self._drive(crew, stage, 0)

        results: List[Result] = [
            (s.name, c.laps, c.alive, c.frames, c.frames if c.finished else None)
            for s, c in crew
        ]
        # Finishers first, by how long they took; then everyone else by distance.
        results.sort(key=lambda r: (r[4] is None, r[4] if r[4] is not None else -r[1]))
        pygame.quit()
        return results


def format_results(results: Sequence[Result]) -> str:
    """The classification as a table, for the log.

    Finishers are timed and the rest are placed by distance, so the table reads
    as a result rather than as a column of numbers to compare by eye.
    """
    lines = [f"{'':>2}  {'entrant':<14} {'laps':>6}  result", "-" * 52]
    won = next((r[4] for r in results if r[4] is not None), None)
    for rank, (name, laps, alive, frames, finished) in enumerate(results, start=1):
        if finished is None:
            state = "still going at the time limit" if alive else f"out on frame {frames}"
        elif rank == 1:
            state = f"WINNER — {finished} frames"
        else:
            state = f"+{finished - won} frames"
        lines.append(f"{rank:>2}. {name:<14} {laps:6.2f}  {state}")
    return "\n".join(lines)


__all__ = ["Result", "Simulation", "Squad", "Stage", "format_results", "set_seed"]
