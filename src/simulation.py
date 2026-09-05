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
import multiprocessing
import multiprocessing.pool
import os
import random
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pygame

from src import parallel
from src.brains import BrainRef, Color, build_brain, color_of, entrants
from src.car import Car, network_input_size
from src.config import SimulationConfig, race_track
from src.dashboard import Dashboard, Entry
from src.genetic import next_generation
from src.renderer import Renderer
from src.track import Track

logger = logging.getLogger(__name__)

Result = Tuple[str, float, bool, int]  # entrant, laps, still running, frames


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

        # Several island layouts per circuit, cycled through generation by
        # generation, so no single arrangement can be memorised.
        self.circuits: List[List[Track]] = parallel.build_circuits(cfg)
        self.input_size = network_input_size(cfg.car)
        # One stream for the start points, drawn once per evaluation and shared.
        # Taking them from the same generator as the genomes tied the sequence
        # to how many numbers had already been drawn, which depends on genome
        # size — so entrants of different sizes met different starts.
        self._start_rng = np.random.default_rng(cfg.seed)
        self._brain_rng = np.random.default_rng(0)  # see `_car`
        self._owner: Dict[int, Squad] = {}

        size = self.circuits[0][0].size
        self.renderer = Renderer(size, cfg.dashboard_width, cfg.fps) if render else None
        self.dashboard = Dashboard(cfg.dashboard_width, size[1])
        grid = list(entries) if entries is not None else [BrainRef(n) for n in entrants()]
        self.squads: List[Squad] = [self._recruit(ref) for ref in grid]
        if not self.squads:
            raise SystemExit("No entrants in src/brains/ — nothing to train.")
        logger.info("Entrants: %s", ", ".join(s.name for s in self.squads))

    # -- setup ---------------------------------------------------------------

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

    def _stage(self, number: int, generation: int, track: Optional[Track] = None) -> Stage:
        """Pick the circuit, the island layout and the start point — once."""
        if track is None:
            variants = self.circuits[number]
            variant = generation % len(variants)
            track = variants[variant]
            start = track.valid_start_index(self._start_rng) if self.cfg.random_start else None
        else:
            variant, start = 0, None  # the race is one fixed test
        return Stage(track, number, variant, start, self.frame_budget(track))

    def _car(
        self, stage: Stage, squad: Squad, genome: np.ndarray, start: Optional[int] = None
    ) -> Car:
        # A dedicated generator, not the run's own: building a brain draws random
        # weights that `set_genome` immediately overwrites, so those draws change
        # nothing — but taking them from a shared stream would advance it, and
        # then the starts and mutations that follow would depend on how many cars
        # happened to be built. That is what made parallel runs diverge.
        driver = build_brain(squad.brain, self.input_size, self._brain_rng)
        driver.set_genome(genome)
        car = Car(stage.track, driver, self.cfg.car, stage.start if start is None else start)
        self._owner[id(car)] = squad
        return car

    def _drive(self, cars: Sequence[Car], stage: Stage, generation: int, waiting=None) -> bool:
        """Step the cars frame by frame, drawing them. False if the user quit.

        When `waiting` is given these are only the cars on screen: the pool is
        scoring the full populations meanwhile, so the cores do the work while
        the window shows a readable subset. Those cars are simulated twice, here
        and in a worker, which costs a few dozen cars' arithmetic and gives
        identical results — every step is deterministic given the track, the
        start and the genome.
        """
        for frame in range(stage.budget):
            for car in cars:
                car.update()
            if self.renderer is not None:
                if self.renderer.poll_quit():
                    return False
                self._draw(cars, stage, generation, frame)
            if not any(car.alive for car in cars):
                break
            if waiting is not None and self.renderer is None and waiting.ready():
                break
        return True

    def _draw(self, cars: Sequence[Car], stage: Stage, generation: int, frame: int) -> None:
        assert self.renderer is not None
        self.renderer.draw_track(
            stage.track,
            cars,
            [self._owner[id(c)].color for c in cars],
            start_index=stage.start or 0,
        )
        entries = []
        for squad in self.squads:
            mine = [c for c in cars if self._owner[id(c)] is squad]
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
                    brain=leader.brain if leader else None,
                )
            )
        self.renderer.present(
            self.dashboard.render(
                generation=generation,
                track_name=stage.track.cfg.name,
                track_number=stage.number + 1,
                track_count=len(self.circuits),
                frame=frame,
                max_frames=stage.budget,
                entries=entries,
                curves=[(s.name, s.color, s.history) for s in self.squads],
            )
        )

    def _evaluate(self, stage: Stage, generation: int) -> Tuple[Dict[int, List[float]], bool]:
        """Score every entrant's full population under one set of conditions."""
        pool = self._ensure_pool()
        lap = stage.track.lap_length

        if pool is None:
            cars = {
                i: [self._car(stage, s, g) for g in s.genomes]
                for i, s in enumerate(self.squads)
            }
            keep = self._drive(list(chain.from_iterable(cars.values())), stage, generation)
            return {i: [c.fitness / lap for c in cs] for i, cs in cars.items()}, keep

        jobs = [
            ((i, b), stage.number, stage.variant, stage.start, block, squad.brain)
            for i, squad in enumerate(self.squads)
            for b, block in enumerate(parallel.split(squad.genomes, self.workers))
        ]
        pending = pool.map_async(parallel.evaluate, jobs)
        keep = True
        if self.renderer is not None:
            shown = [
                self._car(stage, squad, genome)
                for squad in self.squads
                for genome in squad.genomes[: self.shown]
            ]
            keep = self._drive(shown, stage, generation, waiting=pending)

        by_key = dict(pending.get())
        scores: Dict[int, List[float]] = {i: [] for i in range(len(self.squads))}
        for key, *_ in jobs:
            scores[key[0]].extend(by_key[key])
        return scores, keep

    # -- training ------------------------------------------------------------

    def train(self) -> None:
        for generation in range(1, self.cfg.generations + 1):
            per_track: Dict[int, List[List[float]]] = {i: [] for i in range(len(self.squads))}
            for number in range(len(self.circuits)):
                stage = self._stage(number, generation)
                scores, keep = self._evaluate(stage, generation)
                for i, values in scores.items():
                    per_track[i].append(values)
                if not keep:
                    logger.info("Interrupted at generation %d", generation)
                    self.save_all()
                    self.close()
                    pygame.quit()
                    return

            for i, squad in enumerate(self.squads):
                totals = self._aggregate(np.array(per_track[i]))
                best = int(np.argmax(totals))
                if totals[best] > squad.best_fitness:
                    squad.best_fitness = float(totals[best])
                    squad.best_genome = squad.genomes[best]
                squad.history.append(float(totals[best]))
                squad.genomes = next_generation(
                    squad.genomes, list(totals), self.cfg.genetic, squad.rng
                )

            logger.info(
                "gen %3d | %s",
                generation,
                "   ".join(f"{s.name} {s.history[-1]:5.2f}" for s in self.squads),
            )

        self.save_all()
        self.close()
        pygame.quit()

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
                # other choice — `symmetric` above all, which changes what the
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

    def race(self, track: Optional[Track] = None) -> List[Result]:
        """Put every champion on the grid together; return the classification.

        The circuit is `gauntlet` unless told otherwise: nobody trained on it, so
        finishing it shows general driving rather than a memorised layout.
        """
        stage = self._stage(0, 0, track or Track(race_track()))
        # Cars do not collide with one another — the physics never modelled it —
        # so they are spaced along the centreline only to stay visible. Distance
        # is measured from each car's own start, so the stagger costs nobody
        # progress; it does mean they meet each corner a few frames apart.
        spacing = int(26.0 / stage.track.cfg.sample_spacing)
        cars: List[Car] = []
        for position, squad in enumerate(self.squads):
            genome = self.load_champion(squad)
            if genome is None:
                logger.warning("%s has no champion yet and sits this one out", squad.name)
                continue
            start = (-len(cars) * spacing) % len(stage.track.samples)
            cars.append(self._car(stage, squad, genome, start=start))
        if not cars:
            raise SystemExit("No champions in outputs/ — run train.py first.")

        logger.info(
            "Race on '%s', never trained on: %s",
            stage.track.cfg.name,
            ", ".join(self._owner[id(c)].name for c in cars),
        )
        self._drive(cars, stage, self.cfg.generations)
        results = [(self._owner[id(c)].name, c.laps, c.alive, c.frames) for c in cars]
        results.sort(key=lambda r: r[1], reverse=True)
        pygame.quit()
        return results


def format_results(results: Sequence[Result]) -> str:
    """The classification as a table, for the log."""
    lines = [f"{'':>2}  {'entrant':<14} {'laps':>6}  status", "-" * 44]
    for rank, (name, laps, alive, frames) in enumerate(results, start=1):
        state = "running at the flag" if alive else f"out on frame {frames}"
        lines.append(f"{rank:>2}. {name:<14} {laps:6.2f}  {state}")
    return "\n".join(lines)


__all__ = ["Result", "Simulation", "Squad", "Stage", "format_results", "set_seed"]
