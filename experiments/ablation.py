"""Compare network architectures on equal terms.

The project's architecture was inherited, not measured. This runs several of
them through identical training and reports two numbers each: how well they
drive the circuits they trained on, and how far they get on the race circuit
they have never seen — which is the number that actually matters.

Every configuration runs on several seeds, because a genetic algorithm is
stochastic: a single run says almost nothing. Results are reported as medians.

The runs are independent, so they go out to a process pool: this is the one
place in the project where parallelism is free — no shared state, no ordering,
nothing to merge. Python threads would not help (the GIL serialises CPU-bound
work), hence processes.

    python experiments/ablation.py --generations 40 --seeds 3 --workers 6
"""

import argparse
import json
import logging
import multiprocessing
import os
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.brains import BrainRef, build_brain  # noqa: E402
from src.brains.paololimo import NetworkConfig, spec_from_config  # noqa: E402
from src.config import SimulationConfig  # noqa: E402
from src.simulation import Simulation  # noqa: E402

logger = logging.getLogger("ablation")


@dataclass(frozen=True)
class Variant:
    name: str
    hidden: Tuple[int, ...]
    symmetric: bool = False

    def ref(self) -> BrainRef:
        """This variant as an entrant the harness can put on the grid."""
        cfg = NetworkConfig(hidden_sizes=self.hidden, symmetric=self.symmetric)
        return BrainRef("paololimo", spec_from_config(cfg))


VARIANTS: Tuple[Variant, ...] = (
    Variant("linear", ()),
    Variant("8", (8,)),
    Variant("14", (14,)),
    Variant("14-14", (14, 14)),  # the current default
    Variant("24-24", (24, 24)),
    Variant("14-14 sym", (14, 14), symmetric=True),
)


def parameter_count(variant: Variant, cfg: SimulationConfig, inputs: int) -> int:
    return build_brain(variant.ref(), inputs, np.random.default_rng(0)).genome_size


def run_job(job: Tuple[Variant, int, int, int]) -> Tuple[str, int, float, float]:
    """One (variant, seed) pair, for the worker pool."""
    variant, seed, generations, population = job
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    fitness, laps = run_one(variant, seed, generations, population)
    return variant.name, seed, fitness, laps


def run_one(variant: Variant, seed: int, generations: int, population: int) -> Tuple[float, float]:
    """Train one variant on one seed; return (training fitness, race laps).

    Only this variant is on the grid: the point is to compare configurations of
    one entrant against each other, not against the other competitors.
    """
    base = SimulationConfig()
    with tempfile.TemporaryDirectory() as out:
        cfg = replace(
            base,
            seed=seed,
            generations=generations,
            checkpoint_dir=out,
            genetic=replace(base.genetic, population_size=population),
        )
        simulation = Simulation(cfg, render=False, entries=[variant.ref()])
        simulation.train()
        squad = simulation.squads[0]
        if squad.best_genome is None:
            return 0.0, 0.0
        # `race` reads the champion back from the checkpoint, so save it first.
        simulation.save_all()
        results = Simulation(cfg, render=False, entries=[variant.ref()]).race()
        return squad.best_fitness, results[0][1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Architecture ablation")
    parser.add_argument("--generations", type=int, default=40)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--population", type=int, default=100)
    parser.add_argument("--out", type=Path, default=Path("outputs/ablation.json"))
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 2),
        help="parallel processes (default: cores - 2, leaving room for the rest)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    base = SimulationConfig()
    inputs = base.car.num_sensors + 1

    jobs = [
        (variant, seed, args.generations, args.population)
        for variant in VARIANTS
        for seed in range(1, args.seeds + 1)
    ]
    print(
        f"{len(VARIANTS)} variants x {args.seeds} seeds x {args.generations} generations, "
        f"population {args.population}"
    )
    print(f"{len(jobs)} runs across {args.workers} worker processes\n")

    started = time.time()
    with multiprocessing.Pool(args.workers) as pool:
        collected = {}
        for done, (name, seed, fitness, laps) in enumerate(
            pool.imap_unordered(run_job, jobs), start=1
        ):
            collected.setdefault(name, []).append((seed, fitness, laps))
            print(f"  [{done}/{len(jobs)}] {name} seed {seed}: race {laps:.2f} laps", flush=True)
    print(f"\nfinished in {(time.time() - started) / 60:.1f} minutes\n")

    header = (
        f"{'variant':<12} {'params':>6} {'train (median)':>16} "
        f"{'race (median)':>16}  {'race, per seed'}"
    )
    print(header)
    print("-" * len(header))

    results = []
    for variant in VARIANTS:
        runs = sorted(collected.get(variant.name, []))
        training = [fitness for _, fitness, _ in runs]
        racing = [laps for _, _, laps in runs]
        if not runs:
            continue

        row = {
            "variant": variant.name,
            "hidden": list(variant.hidden),
            "symmetric": variant.symmetric,
            "parameters": parameter_count(variant, base, inputs),
            "training": training,
            "race": racing,
        }
        results.append(row)
        print(
            f"{variant.name:<12} {row['parameters']:>6} "
            f"{statistics.median(training):>16.2f} {statistics.median(racing):>16.2f}  "
            + " ".join(f"{v:.2f}" for v in racing)
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nFull results in {args.out}")


if __name__ == "__main__":
    main()
