# cars_ai

Cars that learn to drive, with the neural network and the genetic algorithm
written from scratch. No machine learning libraries: numpy for the linear
algebra, pygame for drawing and collisions, nothing else.

Training runs on **three structurally different circuits**; the final race is
run on a fourth circuit the network has never seen.

Inspired by the project described in `../Can I Make a Better AI Than AI.md`.

## Layout

```
src/
├── config.py           # frozen dataclasses: circuits, car, network, GA, run
├── neural_network.py   # feedforward 8 → 14 → 14 → 2, flat genome
├── genetic.py          # selection, uniform crossover, mutation, elitism
├── track.py            # procedural circuit generation + drivable mask
├── track_check.py      # geometric validation of a generated circuit
├── obstacles.py        # traffic lights and pedestrians: working, not wired in
├── car.py              # physics, 7 ray sensors, braking, fitness
├── renderer.py         # window: circuit on the left, panel on the right
├── dashboard.py        # panel: status, fitness chart, leader's network
├── simulation.py       # generational loop across circuits
├── parallel.py         # population evaluation across worker processes
└── cli.py              # shared arguments and configuration
train.py                # train on the three circuits
race.py                 # take the champion to the unseen circuit
tests/
├── test_core.py        # network, genetic operators, track, car
├── test_obstacles.py   # moving-hazard module, in isolation
└── test_tracks.py      # geometric validity of every circuit
experiments/
└── ablation.py         # architecture comparison across seeds
```

## Setup

Python 3.12 is required: on 3.14 the `pygame.font` module is unavailable
(`ImportError: cannot import name 'Font'`) and the window never opens.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python train.py                                # train with the dashboard (ESC to quit)
python train.py --headless --generations 200   # fast training, no window
python race.py                                 # race on the circuit never trained on
python race.py --track 1                       # replay the champion on a training circuit
python -m pytest tests -q                      # tests
```

`race.py` uses `outputs/best_genome.npz` unless given another path. Headless
training spreads the population across worker processes (`--workers`, default
`cores - 2`); rendering always runs in a single process.

## How it works

**Perception.** Each car casts 7 rays across a 180° fan around its heading;
every ray advances until it leaves the asphalt. The network receives 7
normalised distances plus its own speed: 8 inputs. Islands are cut into the
road mask itself, so the rays see them as walls with no dedicated sensor
channel.

**Brain.** Dense `8 → 14 → 14 → 2` with `tanh` on every layer (366 parameters).
The outputs are steering (−1…1) and throttle, where a negative value brakes.
The weights are never trained by gradient descent: they are a flat genome that
the genetic algorithm recombines. `genetic.py` does not import
`neural_network.py`, so changing the architecture never touches the
evolutionary operators.

**Physics.** The car only moves forward. Steering authority grows with speed up
to 1.5 px/frame and then saturates, which makes the turning radius
`speed / turn rate`: 76 px at top speed, 19 px at a crawl. Braking is therefore
what buys a tight corner. With a constant radius, tight hairpins would not be
difficult but impossible at any speed.

**Circuits.** The centreline is a closed polyline, smoothed with Chaikin's
algorithm and resampled at 4 px spacing. Four layouts:

| Circuit | Use | Lap | Problem it poses |
|---|---|---|---|
| serpentine | training | 3 670 px | 180° hairpins, continuous curves |
| grid-city | training | 3 107 px | right angles: brake *into* the corner |
| speedway | training | 2 083 px | carry speed, then shed it for the chicane |
| gauntlet | **race only** | 2 366 px | continuous chicanes, never trained on |

Hairpins are emitted as explicit semicircular arcs: leaving a single corner for
the smoothing pass to round off yields a radius far too tight to drive (14 px
instead of 75).

**Islands.** Four or five per circuit, alternating hexagons, triangles, squares
and pentagons. They are sized from the local road width so a corridor of at
least 26 px is left on either side, and skipped where there is no room — so a
sealed track cannot be generated.

**Geometric validation** (`track_check.py`), run by the tests on every circuit:

- two stretches touching would hand out shortcuts and corrupt progress
- a corner tighter than the turning radius is impassable
- the narrowest drivable corridor must stay drivable
- the layout must not spill outside the window

Two constraints were found this way: serpentine lanes must be **even** in number
(with an odd count the last one ends on the wrong side and the lap crosses
itself), and bump amplitude must stay below
`(lane spacing − road width − 10) / 2`, or adjacent lanes merge.

**Fitness.** Distance travelled along the centreline, measured by projecting the
car onto the nearest sample with a local window search (constant cost per
frame). Each circuit's score is normalised to **laps**, not pixels: summing raw
distance would let a genome ignore the hardest circuit and optimise only the
fast ones.

Across circuits the fitness is `worst + 0.35 × mean`. Summing let a specialist
win: braking enough to survive the hard circuit costs more laps on the easy ones
than it gains on the hard one, so "flat out everywhere" won on the sum while
dying at full speed in every tight corridor. Leading with the worst circuit took
grid-city from 0.47 to 2.91 laps.

**Frame budget.** Derived per circuit so each one grants the same number of laps
(`laps_budget`, 3.0 at top speed). A flat budget was 4.0 laps of speedway
against 2.3 of serpentine, quietly favouring the short circuit.

**Evolution.** The top 30% form the breeding pool; the best four carry over
untouched (elitism), the rest come from uniform crossover with gaussian mutation
on 5% of genes. Every genome is evaluated on all three circuits before selection.

**Against memorisation.** Each generation starts from a different point on the
lap and meets a different island layout (four pre-built arrangements per
circuit, rotated). Without this the population learned the sequence of
manoeuvres for those exact laps rather than learning to drive: it scored 2.9
laps in training and **0.12** on the unseen circuit, dying on the very first
island. With it, training and race scores match.

## Results

| | Training circuits | Race circuit (unseen) |
|---|---|---|
| Fixed starts and islands | 3.04 · 2.91 · 2.90 | **0.12** |
| Randomised starts and islands | 3.06 · 2.98 · 3.23 | **3.23** |

The network is identical in both rows — same 366 parameters, same architecture,
same fitness. Only the training protocol changed.

## Dashboard

1380×700 window: circuit on the left, panel on the right with the generation,
current circuit (`TRACK k/3`), progress through the frame budget, run
statistics, the best-fitness history and the leader's network — green edges for
positive weights, red for negative, intensity proportional to magnitude. The
diagram is cached and redrawn only when the leader changes.

## Moving hazards (not active)

`obstacles.py` implements timed traffic lights and crossing pedestrians, with
analytic ray-circle intersection. They are no longer wired into the simulation
but the module stays covered by tests: raise `num_traffic_lights` /
`num_pedestrians` in `ObstacleConfig` to bring them back.

Both were removed for measured reasons. Traffic lights made waiting at a red
barrier dominate run time. Pedestrians proved unlearnable for a memoryless
network: seeing only instantaneous distance, it cannot tell one walking into its
path from one standing still — seven of eight race attempts ended in a
pedestrian collision, with zero off-track deaths.

## Differences from the original project

- Circuits are generated procedurally rather than drawn in Photoshop, so the
  project is reproducible with no external assets.
- Continuous outputs (steering, throttle) instead of discrete actions.
- Islands in the carriageway are an addition; the original had only the verges.
- No multi-model race against Claude/ChatGPT/Gemini: a single population here.
