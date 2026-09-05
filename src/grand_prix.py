"""The multi-entrant race: every champion on the gauntlet at the same time.

One circuit none of them trained on, one frame budget, one grid. Each entrant
brings only a brain — the car, the sensors, the physics and the budget are the
harness's — so what separates the finishers is the architecture and the weights
evolution found for it.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pygame

from src.brains import BrainRef, build_brain
from src.car import Car, network_input_size
from src.config import SimulationConfig, race_track
from src.renderer import Renderer
from src.track import Track

logger = logging.getLogger(__name__)

# One colour per entrant, in the order they are given on the command line.
GRID_COLORS: Tuple[Tuple[int, int, int], ...] = (
    (255, 176, 60),   # amber
    (90, 200, 250),   # cyan
    (140, 230, 130),  # green
    (232, 110, 190),  # magenta
    (250, 240, 120),  # yellow
    (200, 200, 210),  # silver
)


@dataclass(frozen=True)
class Entrant:
    """One competitor: a name, the brain it evolved, and its champion genome."""

    name: str
    brain: BrainRef
    genome: np.ndarray
    color: Tuple[int, int, int]
    source: Path


@dataclass(frozen=True)
class Result:
    """Where an entrant got to, and whether it was still running at the flag."""

    name: str
    laps: float
    finished: bool  # still alive when the frame budget ran out
    frames: int


def load_entrant(
    path: Path, input_size: int, color: Tuple[int, int, int], name: Optional[str] = None
) -> Entrant:
    """Read a checkpoint written by `train.py` into a grid entry."""
    data = np.load(path, allow_pickle=False)
    saved_inputs = int(data["input_size"]) if "input_size" in data else -1
    if saved_inputs != input_size:
        raise SystemExit(
            f"{path.name} expects {saved_inputs} sensor inputs, this car produces "
            f"{input_size}. It was trained against a different car — retrain it."
        )
    if "brain" not in data:
        raise SystemExit(
            f"{path.name} carries no brain manifest: it predates the race harness. "
            f"Retrain it with train.py so the checkpoint records its architecture."
        )
    ref = BrainRef(str(data["brain"]), json.loads(str(data["spec"])))
    label = name or (str(data["entrant"]) if "entrant" in data else path.stem)
    return Entrant(label, ref, data["genome"], color, path)


def build_grid(paths: Sequence[Path], input_size: int) -> List[Entrant]:
    """Load every checkpoint, assigning each a colour in the order given."""
    if not paths:
        raise SystemExit("No entrants: pass at least one checkpoint to race.")
    if len(paths) > len(GRID_COLORS):
        raise SystemExit(f"At most {len(GRID_COLORS)} entrants can be told apart on the grid.")
    grid = [load_entrant(p, input_size, GRID_COLORS[i]) for i, p in enumerate(paths)]

    # Two entrants under one name would make the classification unreadable, and
    # it happens easily: every run of `train.py` without `--entrant` labels its
    # champion after the brain it evolved.
    seen: Dict[str, int] = {}
    for e in grid:
        seen[e.name] = seen.get(e.name, 0) + 1
    return [
        e if seen[e.name] == 1 else Entrant(f"{e.name}:{e.source.stem}", e.brain, e.genome, e.color, e.source)
        for e in grid
    ]


class GrandPrix:
    """Runs every entrant together on the race circuit."""

    # Cars have no collision with one another — the physics never modelled it —
    # so they are spaced along the centreline purely so they stay visible. Each
    # one's fitness is measured from where it started, so the stagger costs
    # nobody distance; it does mean they meet the same corners a few frames
    # apart, which is why the rigorous number is still the solo run.
    grid_spacing: float = 26.0

    def __init__(self, cfg: SimulationConfig, render: bool = True) -> None:
        self.cfg = cfg
        self.render = render
        if not render:
            import os

            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        pygame.init()
        self.track = Track(race_track())
        self.input_size = network_input_size(cfg.car)
        self.renderer = Renderer(self.track.size, cfg.dashboard_width, cfg.fps) if render else None
        self.font = pygame.font.SysFont("menlo,consolas,monospace", 15)
        self.title_font = pygame.font.SysFont("menlo,consolas,monospace", 17, bold=True)

    def _build_cars(self, entrants: Sequence[Entrant]) -> List[Car]:
        rng = np.random.default_rng(0)  # weights are overwritten; see Simulation._new_car
        spacing = int(self.grid_spacing / self.track.cfg.sample_spacing)
        cars: List[Car] = []
        for position, entrant in enumerate(entrants):
            driver = build_brain(entrant.brain, self.input_size, rng)
            driver.set_genome(entrant.genome)
            start = (-position * spacing) % len(self.track.samples)
            cars.append(Car(self.track, driver, self.cfg.car, start))
        return cars

    def _budget(self) -> int:
        return int(self.cfg.laps_budget * self.track.lap_length / self.cfg.car.max_speed)

    def run(self, entrants: Sequence[Entrant]) -> List[Result]:
        """Race them all; return the classification, best first."""
        cars = self._build_cars(entrants)
        colors = [e.color for e in entrants]
        budget = self._budget()
        logger.info(
            "Grand Prix on '%s' (%d frames, %.1f laps at top speed): %s",
            self.track.cfg.name,
            budget,
            self.cfg.laps_budget,
            ", ".join(f"{e.name} [{e.brain.name}]" for e in entrants),
        )

        for frame in range(budget):
            for car in cars:
                car.update()
            if self.renderer is not None:
                if self.renderer.poll_quit():
                    break
                self.renderer.draw_track(self.track, cars, colors)
                self.renderer.present(self._panel(entrants, cars, frame, budget))
            if not any(car.alive for car in cars):
                break

        results = [
            Result(e.name, c.laps, c.alive, c.frames) for e, c in zip(entrants, cars)
        ]
        results.sort(key=lambda r: r.laps, reverse=True)
        return results

    def _panel(
        self, entrants: Sequence[Entrant], cars: Sequence[Car], frame: int, budget: int
    ) -> pygame.Surface:
        """Live classification down the right-hand side."""
        panel = pygame.Surface((self.cfg.dashboard_width, self.track.size[1]))
        panel.fill((18, 18, 22))
        y = 18
        panel.blit(self.title_font.render("GRAND PRIX", True, (240, 240, 240)), (18, y))
        y += 26
        panel.blit(self.font.render(f"{self.track.cfg.name} - unseen", True, (150, 150, 160)), (18, y))
        y += 20
        panel.blit(self.font.render(f"frame {frame}/{budget}", True, (150, 150, 160)), (18, y))
        y += 34

        order = sorted(range(len(cars)), key=lambda i: cars[i].laps, reverse=True)
        for rank, i in enumerate(order, start=1):
            car, entrant = cars[i], entrants[i]
            pygame.draw.rect(panel, entrant.color, pygame.Rect(18, y + 3, 10, 10))
            status = "" if car.alive else "  OUT"
            text = f"{rank}. {entrant.name[:12]:<12} {car.laps:5.2f}{status}"
            shade = (235, 235, 235) if car.alive else (120, 120, 128)
            panel.blit(self.font.render(text, True, shade), (36, y))
            y += 22
            panel.blit(
                self.font.render(f"    {entrant.brain.name}", True, (110, 110, 120)), (36, y)
            )
            y += 24
        return panel

    def close(self) -> None:
        pygame.quit()


def format_results(results: Sequence[Result]) -> str:
    """The classification as a table, for the log and for `outputs/`."""
    lines = [f"{'':>2}  {'entrant':<16} {'laps':>6}  status", "-" * 42]
    for rank, r in enumerate(results, start=1):
        lines.append(
            f"{rank:>2}. {r.name:<16} {r.laps:6.2f}  "
            f"{'running at the flag' if r.finished else f'out on frame {r.frames}'}"
        )
    return "\n".join(lines)


def save_results(results: Sequence[Result], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            [
                {"rank": i, "name": r.name, "laps": r.laps, "finished": r.finished, "frames": r.frames}
                for i, r in enumerate(results, start=1)
            ],
            indent=2,
        )
    )
    logger.info("Classification written to %s", path)


__all__ = ["Entrant", "GrandPrix", "Result", "build_grid", "format_results", "load_entrant", "save_results"]
