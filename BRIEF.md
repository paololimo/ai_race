# Brief for a competing agent

You are entering a race. Several agents each write one neural network for the
same self-driving-car problem. Every entry is trained by the same genetic
algorithm, for the same generations, on the same three circuits, from the same
seed — all at the same time, in the same run — and then all the champions race
together on a fourth circuit none of them has ever seen.

Your entry is **one file**: `src/brains/<yourname>.py`. That is the only file
you may create, and you may not modify any other file in the repository.
Dropping it in that directory is the whole installation: the next
`python train.py` trains you alongside everyone else and the next
`python race.py` puts you on the grid, with no other change anywhere.

Everything the comparison depends on — circuits, car physics, ray sensors,
fitness, selection, crossover, mutation, frame budget, seed, and the start point
of every evaluation — belongs to the harness and is identical for every entrant.
That is what makes the result a comparison between architectures rather than
between training setups.

## The problem

A car drives a closed circuit. It only moves forward. Steering authority grows
with speed up to 1.5 px/frame and then saturates, so the turning radius is
`speed / turn rate` — 76 px at top speed, 19 px at a crawl. **Braking is what
buys a tight corner**, and the circuits contain 180° hairpins, right-angle city
corners and continuous chicanes, plus static islands sitting in the carriageway
that the car must pick a side of.

Fitness is distance travelled along the centreline, normalised to laps, scored
across the three training circuits as `worst + 0.35 × mean`. Leading with the
worst circuit is deliberate: on a plain sum, "flat out everywhere" wins by
ignoring the hard circuit. You are selected on competence everywhere.

Each generation starts from a different point on the lap and meets a different
island layout, so a network that memorises a sequence of manoeuvres scores well
in training and fails on the race circuit. Generalisation is what is measured.

## The contract

Write a class with these five members and register it:

```python
"""One line on what this architecture is."""

from typing import Any, Mapping

import numpy as np

from src.brains import register_brain


@register_brain("yourname")   # your name on the grid, and the checkpoint filename
class YourBrain:
    def __init__(self, input_size: int, spec: Mapping[str, Any], rng: np.random.Generator) -> None:
        """`input_size` is 8. Read hyperparameters from `spec`, each with a
        default — `spec` is empty on a normal run and is written verbatim into
        the checkpoint, so it must stay JSON-serialisable. Draw the initial
        parameters from `rng` and from nothing else."""
        hidden = int(spec.get("hidden", 16))
        scale = 1.0 / np.sqrt(input_size)
        self.w1 = rng.normal(0.0, scale, size=(input_size, hidden))
        self.b1 = rng.normal(0.0, scale, size=hidden)
        self.w2 = rng.normal(0.0, scale, size=(hidden, 2))
        self.b2 = rng.normal(0.0, scale, size=2)

    @property
    def genome_size(self) -> int:
        """Total learnable parameters. Never changes for a given `spec`."""
        return self.w1.size + self.b1.size + self.w2.size + self.b2.size

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Sensors in, `[steering, throttle]` out, both within [-1, 1]."""
        hidden = np.tanh(inputs @ self.w1 + self.b1)
        return np.tanh(hidden @ self.w2 + self.b2)

    def get_genome(self) -> np.ndarray:
        """Every parameter, flattened into one 1-D float array."""
        return np.concatenate([self.w1.ravel(), self.b1, self.w2.ravel(), self.b2])

    def set_genome(self, genome: np.ndarray) -> None:
        """The exact inverse of `get_genome`."""
        if genome.size != self.genome_size:
            raise ValueError(f"Genome size {genome.size} does not match {self.genome_size}")
        offset = 0
        for name, target in (("w1", self.w1), ("b1", self.b1), ("w2", self.w2), ("b2", self.b2)):
            setattr(self, name, genome[offset : offset + target.size].reshape(target.shape))
            offset += target.size
```

**Inputs** — `forward` receives a float array of length 8:

| index | meaning | range |
|---|---|---|
| 0–6 | distance along 7 rays spread over 180° around the heading, nearest thing ahead, normalised by a 200 px range | 0.0 (wall touching) … 1.0 (nothing within range) |
| 7 | own speed, normalised by top speed | 0.0 … 1.0 |

Ray 0 is the leftmost, ray 6 the rightmost, ray 3 straight ahead. Islands are
cut into the road mask itself, so the rays already see them — there is no
separate obstacle channel.

**Outputs** — exactly 2 floats, each within `[-1, 1]`:

| index | meaning |
|---|---|
| 0 | steering: −1 hard left, +1 hard right |
| 1 | throttle: positive accelerates, **negative brakes** |

Values outside `[-1, 1]` are not clipped for you and would give you physics no
other entrant is subject to. Bound them yourself.

**Genome** — the genetic algorithm treats your flat vector as opaque: uniform
crossover gene by gene, then gaussian noise (σ 0.3) on 5% of genes. Two
consequences worth designing around:

- Neighbouring genes are recombined independently, so a layout that keeps
  related parameters adjacent survives crossover better than one that scatters
  them.
- `genome_size` must be identical for every instance built with the same `spec`,
  and must not change over an instance's lifetime.

## Rules

1. One new file, `src/brains/<yourname>.py`. No edits anywhere else.
2. **No machine-learning libraries.** numpy only — that is the whole point of the
   project. No torch, no tensorflow, no jax, no scikit-learn.
3. **One `@register_brain` per class.** A second name would put you on the grid
   twice, under two colours, competing with yourself. The harness ignores the
   extra name and logs a warning.
4. No training of your own. You supply the architecture; the harness's genetic
   algorithm finds the weights. No backpropagation, no pretrained weights, no
   reading or writing files.
5. No randomness of your own: use the `rng` you are given, never
   `np.random.seed`, `random` or `time`. The harness scores populations across
   worker processes and reproduces a single-process run exactly.
6. No state between `forward` calls unless your architecture is deliberately
   recurrent — and if it is, `set_genome` must reset it, or your behaviour will
   depend on which cars ran before you.
7. `forward` runs once per car per frame, millions of times per generation.
   Something ten times slower than a dense 8→14→14→2 will be a problem.
8. It must import cleanly and pass `python -m pytest tests -q`. A file that
   raises on import is logged and left out of the race rather than stopping it.

## What you are judged on

Laps completed on `gauntlet`, the circuit that appears in no training run.
Nothing else. A larger network is not automatically better: 366 parameters is
the incumbent, and the genetic algorithm's search gets harder as the genome
grows. An inductive bias that shrinks the search without losing capacity is
worth more than raw width — the incumbent offers one, an optional left/right
symmetry that averages the network against its own mirror image, since the
response to a wall on the left should be the mirror of the response to a wall on
the right.

## Check your work

```bash
python -m pytest tests -q                    # must stay green, and it checks your entry
python -c "from src.brains import entrants; print(entrants())"
python train.py --generations 5 --population 20 --headless
python race.py --headless
```

The test suite runs your file against the contract: shape and range of your
outputs, that your genome is flat and round-trips, and that you face the same
conditions as everyone else.

## Deliverable

The single file `src/brains/<yourname>.py`, plus a short note (5–10 lines) on
what the architecture is and why you expect it to generalise to a circuit it has
not seen. No other changes.
