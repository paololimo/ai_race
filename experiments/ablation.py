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
    """One thing to measure, on one of two independent axes.

    `hidden` and `symmetric` are the *architecture*, which lives in one
    entrant's own file and is the only thing a competitor may vary. `crossover`
    is the *harness*, shared by everyone, so changing it changes the experiment
    for all entrants equally and advantages nobody. They are varied here in one
    script only because both are questions about the same run; do not read a
    crossover result as an argument about any one architecture.
    """

    name: str
    hidden: Tuple[int, ...] = (14, 14)
    symmetric: bool = False
    skip: bool = False
    decoupled: bool = False
    crossover: str = "uniform"

    def ref(self) -> BrainRef:
        """This variant as an entrant the harness can put on the grid."""
        cfg = NetworkConfig(
            hidden_sizes=self.hidden,
            symmetric=self.symmetric,
            skip=self.skip,
            decoupled=self.decoupled,
        )
        return BrainRef("paololimo", spec_from_config(cfg))


# Architecture, one question per group.
#
# Size: is the default too big to search in the generations available? The
# earlier run hinted that it is, but on three seeds the hint means nothing.
#
# Symmetry: the problem is mirror-symmetric, so imposing the symmetry halves the
# search without losing capacity — if the halving is worth more than the two
# extra forward passes it costs.
#
# Skip: a direct input-to-output path gives evolution a linear controller it can
# refine immediately, instead of one it must first build through two tanh layers.
#
# Decoupling: separate stacks per control, so a mutation that improves braking
# lands in weights steering never reads and cannot damage it.
ARCHITECTURES: Tuple[Variant, ...] = (
    Variant("linear", ()),
    Variant("8", (8,)),
    Variant("14", (14,)),
    Variant("14-14", (14, 14)),  # the current default
    Variant("24-24", (24, 24)),
    Variant("14 sym", (14,), symmetric=True),
    Variant("14-14 sym", (14, 14), symmetric=True),
    Variant("14 skip", (14,), skip=True),
    Variant("14-14 skip", (14, 14), skip=True),
    Variant("14 split", (14,), decoupled=True),
    Variant("14-14 split", (14, 14), decoupled=True),
    Variant("14 all", (14,), symmetric=True, skip=True, decoupled=True),
    Variant("14-14 all", (14, 14), symmetric=True, skip=True, decoupled=True),
)

# Harness: uniform crossover splits a neuron's incoming weights between parents,
# so recombination behaves closer to heavy mutation than to inheritance. Cutting
# the vector into runs keeps co-adapted groups together; dropping crossover
# altogether leaves the work to mutation, which on small genomes is often no
# worse. Held at the default architecture so only the operator moves.
OPERATORS: Tuple[Variant, ...] = (
    Variant("uniform", crossover="uniform"),
    Variant("one-point", crossover="one-point"),
    Variant("two-point", crossover="two-point"),
    Variant("mutation only", crossover="none"),
)

VARIANTS: Tuple[Variant, ...] = ARCHITECTURES + OPERATORS


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
            genetic=replace(
                base.genetic, population_size=population, crossover=variant.crossover
            ),
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

    # Median and spread together, never the median alone. A genetic algorithm on
    # this problem swings from 0.04 to 3.05 laps on the same architecture with
    # only the seed changed, so a table of medians invites a conclusion the data
    # does not support. If the spreads overlap, the difference is not measured.
    results = []
    for group, title in ((ARCHITECTURES, "ARCHITECTURE"), (OPERATORS, "CROSSOVER (harness-wide)")):
        header = (
            f"{title:<16} {'params':>6} {'train med':>10} "
            f"{'race med':>9} {'race range':>14}  {'per seed'}"
        )
        print(f"\n{header}\n" + "-" * len(header))
        for variant in group:
            runs = sorted(collected.get(variant.name, []))
            if not runs:
                continue
            training = [fitness for _, fitness, _ in runs]
            racing = [laps for _, _, laps in runs]

            row = {
                "variant": variant.name,
                "hidden": list(variant.hidden),
                "symmetric": variant.symmetric,
                "skip": variant.skip,
                "decoupled": variant.decoupled,
                "crossover": variant.crossover,
                "parameters": parameter_count(variant, base, inputs),
                "training": training,
                "race": racing,
            }
            results.append(row)
            print(
                f"{variant.name:<16} {row['parameters']:>6} "
                f"{statistics.median(training):>10.2f} {statistics.median(racing):>9.2f} "
                f"{min(racing):>6.2f}-{max(racing):<7.2f}  "
                + " ".join(f"{v:.2f}" for v in racing)
            )

    if args.seeds < 5:
        print(
            f"\nOnly {args.seeds} seeds: the spread will swamp any difference between "
            f"variants. Use --seeds 7 or more before concluding anything."
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nFull results in {args.out}")


if __name__ == "__main__":
    main()
