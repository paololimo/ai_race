# Brief for a competing agent

You are entering a race. Several agents are each writing one neural network for
the same self-driving-car problem. Every entry is trained by the same genetic
algorithm, for the same number of generations, on the same three circuits, from
the same seed, and then all the champions race together on a fourth circuit none
of them has ever seen.

Your entry is **one file**: `src/brains/<yourname>.py`. That is the only file
you may create, and you may not modify any other file in the repository.
Everything else — circuits, car physics, ray sensors, fitness, selection,
crossover, mutation, frame budget, random seed — is fixed by the harness and is
identical for every entrant. That is what makes the result a comparison between
architectures rather than between training setups.

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
ignoring the hard circuit. Your network is selected on competence everywhere.

Each generation starts from a different point on the lap and meets a different
island layout, so a network that memorises a sequence of manoeuvres scores well
in training and fails on the race circuit. Generalisation is the thing being
measured.

## The contract

Copy `src/brains/_template.py` to `src/brains/<yourname>.py`, rename the class,
change the `@register_brain("...")` name, and implement:

```python
def __init__(self, input_size: int, spec: Mapping[str, Any], rng: np.random.Generator) -> None
@property
def genome_size(self) -> int
def forward(self, inputs: np.ndarray) -> np.ndarray
def get_genome(self) -> np.ndarray
def set_genome(self, genome: np.ndarray) -> None
```

**Inputs** — `forward` receives a float array of length 8:

| index | meaning | range |
|---|---|---|
| 0–6 | distance along 7 rays spread over 180° around the heading, nearest thing ahead, normalised by a 200 px range | 0.0 (wall touching) … 1.0 (nothing within range) |
| 7 | own speed, normalised by top speed | 0.0 … 1.0 |

Ray 0 is the leftmost, ray 6 the rightmost, ray 3 straight ahead. Islands are
cut into the road mask itself, so the rays already see them — there is no
separate obstacle channel.

**Outputs** — an array of exactly 2 floats, each within `[-1, 1]`:

| index | meaning |
|---|---|
| 0 | steering: −1 hard left, +1 hard right |
| 1 | throttle: positive accelerates, **negative brakes** |

Values outside `[-1, 1]` are not clipped for you and will produce physics no
other entrant is subject to. Bound them yourself.

**Genome** — `get_genome` returns every learnable parameter as one flat 1-D
float array; `set_genome` is its exact inverse. The genetic algorithm treats
that vector as opaque: it does uniform crossover gene by gene and adds gaussian
noise (σ 0.3) to 5% of genes. Two consequences worth designing around:

- Neighbouring genes are recombined independently, so a layout that keeps
  related parameters adjacent survives crossover better than one that scatters
  them.
- `genome_size` must be identical for every instance built with the same
  `spec`, and must not change over an instance's lifetime.

**Hyperparameters** go in `spec`, a plain dict, always with a default because
`spec` is empty unless `--brain-spec` is passed. It is written verbatim into the
checkpoint, so it must stay JSON-serialisable (no numpy types, no tuples of
tuples — lists and scalars).

**Randomness** — draw initial parameters from the `rng` you are given and from
nothing else. No `np.random.seed`, no `random`, no `time`. The harness evaluates
populations across worker processes and relies on being able to reproduce a
serial run exactly, and it draws the start points from a stream of its own so
that entrants of different genome sizes meet the same circuits, the same island
layouts and the same starts on the same seed. Seeding your own generator would
break that.

## Rules

1. One new file: `src/brains/<yourname>.py`. No edits anywhere else.
2. **No machine-learning libraries.** numpy only — that is the whole point of
   the project. No torch, no tensorflow, no jax, no scikit-learn.
3. No training of your own. You supply the architecture; the harness's genetic
   algorithm finds the weights. Do not implement backpropagation, do not load
   pretrained weights, do not read or write files.
4. No state carried between `forward` calls unless your architecture is
   deliberately recurrent — and if it is, `set_genome` must reset it, or a car's
   behaviour will depend on which cars ran before it.
5. `forward` runs once per car per frame, millions of times per generation.
   Something ten times slower than a dense 8→14→14→2 will be a problem.
6. It must import cleanly and pass `python -m pytest tests -q`.

## What you are being judged on

Laps completed on `gauntlet`, the circuit that appears in no training run.
Nothing else. A larger network is not automatically better: 366 parameters is
the incumbent, and the genetic algorithm's search gets harder as the genome
grows. Inductive biases that shrink the search without losing capacity are worth
more than raw width — the baseline offers one, an optional left/right symmetry
that averages the network against its own mirror image, since the response to a
wall on the left should be the mirror of the response to a wall on the right.

## Check your work

```bash
python -m pytest tests -q                                  # must stay green
python -c "from src.brains import available_brains; print(available_brains())"
python train.py --brain <yourname> --generations 5 --population 20 --headless
python race.py --grid outputs/<yourname>_genome.npz
```

## Deliverable

The single file `src/brains/<yourname>.py`, plus a short note (5–10 lines) on
what the architecture is and why you expect it to generalise to a circuit it has
not seen. No other changes.
