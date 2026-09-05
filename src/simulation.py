"""Training loop: evaluate every genome on each training circuit, then evolve."""

import logging
import multiprocessing
import multiprocessing.pool
import os
import random
from dataclasses import replace
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pygame

from src.car import Car, network_input_size
from src.config import NetworkConfig, SimulationConfig, race_track, track_variants
from src.dashboard import Dashboard
from src.genetic import next_generation
from src import parallel
from src.neural_network import NeuralNetwork
from src.renderer import Renderer
from src.track import Track

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


class Simulation:
    """Runs the generational loop over the training circuits."""

    def __init__(
        self, cfg: SimulationConfig, render: bool = True, workers: int = 1
    ) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self._brain_rng = np.random.default_rng(0)  # see _new_car
        set_seed(cfg.seed)
        # Rendering has to stay in one process, so the pool is for headless runs.
        self.workers = 1 if render else max(1, workers)
        self._pool: Optional[multiprocessing.pool.Pool] = None

        if not render:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        pygame.init()

        # Several island layouts per circuit, cycled through generation by
        # generation, so no single arrangement can be memorised.
        self.circuits: List[List[Track]] = parallel.build_circuits(cfg)
        self.tracks: List[Track] = [variants[0] for variants in self.circuits]
        self.input_size = network_input_size(cfg.car)
        self.renderer: Optional[Renderer] = None
        if render:
            self.renderer = Renderer(self.tracks[0].size, cfg.dashboard_width, cfg.fps)
        self.dashboard = Dashboard(cfg.dashboard_width, self.tracks[0].size[1])

        self.best_fitness = 0.0
        self.best_genome: Optional[np.ndarray] = None
        self.history: List[float] = []  # best fitness per generation

    def _ensure_pool(self) -> Optional["multiprocessing.pool.Pool"]:
        """Start the worker pool on first use, and keep it for the whole run.

        Each worker rebuilds the circuits once — they are derived from the seed,
        so every process ends up with identical geometry — and then only genomes
        travel between processes, a few hundred kilobytes per generation.
        """
        if self.workers <= 1:
            return None
        if self._pool is None:
            self._pool = multiprocessing.Pool(
                self.workers,
                initializer=parallel.worker_init,
                initargs=(self.cfg,),
            )
            logger.info("Evaluating on %d worker processes", self.workers)
        return self._pool

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

    def _island_variants(self, cfg) -> List[Track]:
        """Build the same circuit with the islands shifted around the lap."""
        count = max(1, self.cfg.island_variants)
        built = [Track(cfg)]
        jitter = np.random.default_rng(self.cfg.seed)
        for _ in range(count - 1):
            shifted = tuple(
                float((fraction + jitter.uniform(0.05, 0.95)) % 1.0) for fraction in cfg.islands
            )
            built.append(Track(replace(cfg, islands=shifted)))
        return built

    def _new_car(
        self,
        track: Track,
        genome: np.ndarray,
        network: Optional[NetworkConfig] = None,
        start_index: Optional[int] = None,
    ) -> Car:
        # A dedicated generator, not the run's own: building a network draws
        # random weights that `set_genome` immediately overwrites, so those
        # draws change nothing — but taking them from `self.rng` would advance
        # it, and then the start points and the mutations that follow would
        # differ depending on how many cars happened to be built. That is what
        # made parallel runs diverge from serial ones.
        brain = NeuralNetwork(self.input_size, network or self.cfg.network, self._brain_rng)
        brain.set_genome(genome)
        return Car(track, brain, self.cfg.car, start_index)

    def _random_genomes(self) -> List[np.ndarray]:
        blank = NeuralNetwork(self.input_size, self.cfg.network, self.rng)
        genomes = [blank.get_genome()]
        for _ in range(self.cfg.genetic.population_size - 1):
            genomes.append(NeuralNetwork(self.input_size, self.cfg.network, self.rng).get_genome())
        return genomes

    def run_on_track(
        self,
        genomes: Sequence[np.ndarray],
        track_number: int,
        generation: int,
        network: Optional[NetworkConfig] = None,
    ) -> Tuple[List[float], bool]:
        """Race every genome on one circuit; return their scores."""
        variants = self.circuits[track_number] if track_number < len(self.circuits) else None
        variant_index = generation % len(variants) if variants else 0
        if variants:
            # Rotate through the island layouts, one per generation.
            track = variants[variant_index]
            self.tracks[track_number] = track
        else:
            track = self.tracks[track_number]

        start = track.valid_start_index(self.rng) if self.cfg.random_start else None

        pool = self._ensure_pool()
        if pool is not None:
            blocks = parallel.split(genomes, self.workers)
            jobs = [
                (i, track_number, variant_index, start, block, network)
                for i, block in enumerate(blocks)
            ]
            results = dict(pool.map(parallel.evaluate, jobs))
            scores: List[float] = []
            for i in range(len(blocks)):
                scores.extend(results[i])
            return scores, True

        cars = [self._new_car(track, g, network, start) for g in genomes]
        budget = self.frame_budget(track)

        for frame in range(budget):
            for car in cars:
                car.update()
            if self.renderer is not None:
                if self.renderer.poll_quit():
                    return self._scores(cars, track), False
                self._draw(track, cars, generation, track_number, frame, budget)
            if not any(car.alive for car in cars):
                break
        return self._scores(cars, track), True

    def frame_budget(self, track: Track) -> int:
        """How long a genome may drive this circuit, in frames.

        Proportional to lap length, so a short lap does not hand out more laps
        than a long one for the same amount of simulated time.
        """
        return int(self.cfg.laps_budget * track.lap_length / self.cfg.car.max_speed)

    @staticmethod
    def _scores(cars: Sequence[Car], track: Track) -> List[float]:
        """Score a circuit in laps, not pixels.

        The training circuits differ in length, so summing raw distance would
        let a genome ignore the hardest track and optimise only the fast ones.
        Laps put every circuit on the same scale.
        """
        return [c.fitness / track.lap_length for c in cars]

    def _draw(
        self,
        track: Track,
        cars: Sequence[Car],
        generation: int,
        track_number: int,
        frame: int,
        budget: int,
    ) -> None:
        assert self.renderer is not None
        leader = self.renderer.draw_track(track, cars)
        panel = self.dashboard.render(
            generation=generation,
            track_name=track.cfg.name,
            track_number=track_number + 1,
            track_count=len(self.tracks),
            frame=frame,
            max_frames=budget,
            cars=cars,
            leader=leader,
            best_ever=self.best_fitness,
            history=self.history,
        )
        self.renderer.present(panel)

    def _aggregate(self, per_track_scores: np.ndarray) -> np.ndarray:
        """Combine one genome's scores across the circuits into one fitness.

        Summing them lets a genome ignore the hardest circuit: braking enough to
        survive it costs more laps on the easy ones than it gains on the hard
        one, so "flat out everywhere" wins on the sum while dying instantly on
        anything tight. Leading with the *worst* circuit makes competence
        everywhere the only way up; the mean stays in as a tie-breaker, and as
        the signal that guides early generations, when every worst score is 0.

        `per_track_scores` is indexed [track, genome].
        """
        weakest = per_track_scores.min(axis=0)
        average = per_track_scores.mean(axis=0)
        return weakest + self.cfg.mean_weight * average

    def train(self) -> None:
        genomes = self._random_genomes()
        for generation in range(1, self.cfg.generations + 1):
            per_track_scores = np.zeros((len(self.tracks), len(genomes)))
            for track_number in range(len(self.tracks)):
                scores, keep_going = self.run_on_track(genomes, track_number, generation)
                per_track_scores[track_number] = scores
                if not keep_going:
                    logger.info("Training interrupted by user at generation %d", generation)
                    self.save_best()
                    pygame.quit()
                    return

            totals = self._aggregate(per_track_scores)
            best_index = int(np.argmax(totals))
            if totals[best_index] > self.best_fitness:
                self.best_fitness = float(totals[best_index])
                self.best_genome = genomes[best_index]
            self.history.append(float(totals[best_index]))

            logger.info(
                "gen %3d | fitness %5.2f | weakest %5.2f | best-of laps %s",
                generation,
                totals[best_index],
                float(per_track_scores[:, best_index].min()),
                " ".join(f"{s:5.2f}" for s in per_track_scores[:, best_index]),
            )
            genomes = next_generation(genomes, list(totals), self.cfg.genetic, self.rng)

        self.save_best()
        self.close()
        pygame.quit()

    def save_best(self) -> Optional[Path]:
        if self.best_genome is None:
            return None
        out_dir = Path(self.cfg.checkpoint_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "best_genome.npz"
        np.savez(
            path,
            genome=self.best_genome,
            fitness=self.best_fitness,
            input_size=self.input_size,
            hidden_sizes=np.array(self.cfg.network.hidden_sizes),
            history=np.array(self.history),
        )
        logger.info("Saved best genome (fitness %.1f) to %s", self.best_fitness, path)
        return path

    def _load_genome(self, genome_path: Path) -> Tuple[np.ndarray, NetworkConfig]:
        """Load a saved model, rebuilding the architecture it was trained with.

        The checkpoint carries its own layer sizes, so a model stays usable
        after the project's default architecture changes. Only the input count
        must still match: that one is fixed by the car's sensors, not by the
        network, so a mismatch means the car itself has changed.
        """
        data = np.load(genome_path)
        saved_inputs = int(data["input_size"]) if "input_size" in data else -1
        if saved_inputs != self.input_size:
            raise SystemExit(
                f"This model expects {saved_inputs} sensor inputs but the car now "
                f"produces {self.input_size}. The sensor layout changed since it was "
                f"trained — retrain with train.py."
            )
        hidden = tuple(int(n) for n in data["hidden_sizes"]) if "hidden_sizes" in data else None
        network = replace(self.cfg.network, hidden_sizes=hidden) if hidden else self.cfg.network
        if hidden and hidden != self.cfg.network.hidden_sizes:
            logger.info("Model architecture %s differs from the current default %s — using the model's.",
                        hidden, self.cfg.network.hidden_sizes)
        return data["genome"], network

    def replay_best(self, genome_path: Path, track_number: int = 0) -> None:
        """Watch a saved genome drive one training circuit on its own."""
        genome, network = self._load_genome(genome_path)
        self.run_on_track([genome], track_number, self.cfg.generations, network)
        pygame.quit()

    def race(self, genome_path: Path) -> None:
        """Run the saved genome on the race circuit, which it never trained on."""
        genome, network = self._load_genome(genome_path)
        self.tracks = [Track(race_track())]
        self.circuits = []  # no rotation, no random start: the race is one fixed test
        logger.info("Final race on '%s' - a circuit never seen in training", self.tracks[0].cfg.name)
        scores, _ = self.run_on_track([genome], 0, self.cfg.generations, network)
        logger.info("Race result: %.2f laps of '%s'", scores[0], self.tracks[0].cfg.name)
        pygame.quit()
