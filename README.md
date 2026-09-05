# cars_ai

Cars that learn to drive, with the neural network and the genetic algorithm
written from scratch. No machine learning libraries: numpy for the linear
algebra, pygame for drawing and collisions, nothing else.

Several networks compete: one written by hand, the others by AI agents. They
**train together**, in the same run, on the same three circuits, and then race
together on a fourth circuit none of them has ever seen.

```bash
python train.py     # everyone evolves side by side, with the dashboard
python race.py      # every champion on the grid, on the unseen circuit
```

Neither command takes an entrant argument. The grid is whatever is in
`src/brains/`.

## Layout

```
src/
├── brains/             # the competitors — one file each, and nothing else
│   ├── __init__.py     # the registry: Brain protocol, discovery, colours
│   ├── paololimo.py    # the hand-written network: 8 → 10 → 4 → 2, with a skip
│   ├── claude.py
│   ├── codex.py
│   └── gemini.py
├── config.py           # frozen dataclasses: circuits, car, GA, run
├── genetic.py          # rank-weighted selection, crossover, annealed mutation
├── track.py            # procedural circuit generation + drivable mask
├── track_check.py      # geometric validation of a generated circuit
├── car.py              # physics, 7 ray sensors, braking, fitness
├── renderer.py         # window layout, circuit scaling, car drawing
├── dashboard.py        # panel: standings and each entrant's own network
├── analysis.py         # strip: progress, genetic spread, per-circuit bars
├── simulation.py       # the generational loop, and the race
├── parallel.py         # population scoring across worker processes
├── recorder.py         # frames piped into ffmpeg, for a video of a run
└── cli.py              # shared arguments and configuration
train.py                # train everyone on the three circuits
race.py                 # take every champion to the unseen circuit
tests/
├── test_core.py        # network, genetic operators, track, car
├── test_entrants.py    # the race harness: discovery, contract, parity
└── test_tracks.py      # geometric validity of every circuit
experiments/
└── ablation.py         # architecture comparison across seeds
BRIEF.md                # the spec handed to a competing agent
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
python train.py                                # dashboard, everyone at once (ESC to quit)
python train.py --headless --generations 200   # fast training, all cores
python train.py --shown 20                     # draw more cars per entrant
python train.py --headless --record run.mp4    # every frame, ~70 min of video
python train.py --headless --record run.mp4 --record-clip 3   # ~6 min instead
python race.py --headless --record race.mp4    # the race, filmed whole
python race.py                                 # the race, on the unseen circuit
python race.py --track 1                       # the same grid on a training circuit
python -m pytest tests -q                      # tests
```

Each champion is written to `outputs/<name>.npz`, which is what `race.py` reads.

## Recording it

```bash
python train.py --headless --record outputs/run.mp4
```

**Every frame is kept.** A 120-generation run is around 70 minutes of video and
700 MB — enough to speed up afterwards to whatever suits, which is simpler than
deciding in advance and better than any rule this program could pick for you:

```bash
ffmpeg -i outputs/run.mp4 -filter:v "setpts=PTS/6" -an outputs/run-6x.mp4
```

At 4-6x the result is still smooth: a car covers 4 px a frame, so 6x moves it
24 px between frames on a 1000 px circuit. Past about 8x it starts to stutter,
and at 20x — which is what "one frame in twenty" amounts to, whether the
thinning happens at capture or at playback — a car crosses a whole corner
between frames and the motion is gone.

`--record-clip S` is the shortcut for when 70 minutes is not wanted at all. It
films only the first S seconds of each circuit, in one generation in
`--record-every`, cutting between them: at `--record-clip 3` a 120-generation
run is six minutes. Each clip starts on the line, so the same situation is
directly comparable across generations, and the panel carries the generation
number so the cuts read as progress. The last generation is filmed to the end
either way — every other clip is three seconds off the line, which would mean
the video never once shows a trained car completing anything.

`x264` at `crf 16`, preset `slow`, `yuv420p` and `+faststart`: thin lines and
small text are what h264 destroys first and they are most of this picture, so
it is encoded for screen capture rather than for camera footage.

`--headless` lifts the 60 fps cap — with no window there is nothing to hold the
loop back, so the recording takes as long as the drawing does rather than as
long as watching would have — and gives the full 1380x980 layout instead of one
shrunk to fit a monitor it is never shown on. `--record` works with the window
open too; it just runs at watching speed.

A race is one lap of one circuit, so `race.py --record` films it whole.

## Adding a competitor

Write one file, `src/brains/<name>.py`, holding a class that satisfies the
`Brain` protocol — `forward` (8 sensor readings in, steering and throttle out),
plus `get_genome` / `set_genome` to expose its parameters as one flat vector —
and decorate it with `@register_brain("<name>")`. That is the whole
installation: every module in `src/brains/` is imported on startup, so the next
`train.py` trains it alongside the others, the next `race.py` puts it on the
grid, and the dashboard grows a row and a curve for it. No other file changes.

`BRIEF.md` is the spec to hand to the agent writing it. The test suite checks
every entrant it finds against the contract, so a broken entry fails the tests
rather than the race — and an entry that raises on *import* is logged and left
out rather than stopping everyone else.

Colours are assigned in `src/brains/__init__.py`, not by the entrants, so no
competitor can pick one that clashes with another's: paololimo red, gemini blue,
codex off-white, claude orange, and anything else takes the next free colour
from a palette.

## How it works

**Perception.** Each car casts 7 rays across a 180° fan around its heading;
every ray advances until it leaves the asphalt. The network receives 7
normalised distances plus its own speed: 8 inputs. Islands are cut into the
road mask itself, so the rays see them as walls with no dedicated sensor
channel.

**Brains.** Each entrant is a flat genome the genetic algorithm recombines; the
weights are never trained by gradient descent. There is no gradient to take: the
fitness is distance travelled before crashing, and reaching it means passing
through a boolean road mask and a discrete death condition, neither of which has
a useful derivative. `genetic.py` knows nothing about any architecture, which is
why one algorithm can breed all of them.

The hand-written entrant is `8 → 10 → 4 → 2` with `tanh`, a direct linear path
from the sensors to the controls alongside the hidden ones, and left/right
symmetry imposed rather than learned: 160 parameters. It was `8 → 14 → 14 → 2`
at 366 until the arithmetic was done — 100 genomes over 120 generations is
12 000 evaluations, and a derivative-free search wants 100 to 1000 of them per
parameter, so 366 parameters were never going to be searched, only sampled.

The hidden path is a funnel rather than one wide layer, at a cost of ten
parameters: reading the ray fan is comparing each ray against its neighbours to
find the gap, and *then* weighing that against own speed to set steering and
braking, which is two stages one matrix would have to do at once.

**Physics.** The car only moves forward. Steering authority grows with speed up
to 1.5 px/frame and then saturates, which makes the turning radius
`speed / turn rate`: 76 px at top speed, 19 px at a crawl. Braking is therefore
what buys a tight corner. With a constant radius, tight hairpins would not be
difficult but impossible at any speed.

**Circuits.** The centreline is a closed polyline, smoothed with Chaikin's
algorithm and resampled at 4 px spacing. Four layouts:

| Circuit | Use | Lap | Corners forcing < ½ speed | Lap slowed | Problem it poses |
|---|---|---|---|---|---|
| serpentine | training | 3 759 px | 2 | 14% | hairpins, then a six-bump slalom down every straight |
| grid-city | training | 2 701 px | 8 | 12% | right angles: brake *into* the corner, eight times |
| speedway | training | 2 119 px | 6 | 13% | carry speed, then shed it for a real chicane |
| gauntlet | **race only** | 2 473 px | 5 | 21% | continuous chicanes, never trained on — the hardest |

They were not balanced, and the imbalance was invisible until the panel grew a
per-circuit breakdown. Measured, the first three had **one, twelve and one**
corner forcing below half speed: grid-city was the only circuit that
discriminated between entrants, speedway was saturated at the frame budget so
it separated nobody, and the race circuit was the *easiest* of the four — the
one the entrants are ranked on was the one they were least prepared for.

The levers were geometric, not arbitrary. A hairpin's radius is half the lane
spacing, so serpentine's were 75 px and takeable at 98% of top speed; the
difficulty had to come from the straights instead. Speedway's chicane was drawn
with real direction changes and then smoothed away by three Chaikin passes.
Gauntlet gained two peaks and 18 px of amplitude. Grid-city lost four of its
twelve right angles — not because twelve was impossible, an entrant reached
2.72 laps on it, but because it was learnt far more slowly than the rest, and
until it was, the compound odds of twelve corners in a row decided the ranking
rather than anyone's driving.

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
untouched (elitism), the rest come from crossover of two parents drawn from the
pool, with gaussian mutation on 5% of genes. Every genome is evaluated on all
three circuits before selection.

Two things about that search were wrong for a long time, both in `genetic.py`
and so shared by every entrant equally.

Parents were drawn *uniformly* from the pool, which threw away the ranking
selection had just produced: the best genome in the pool and the one scraping
the cut-off were equally likely to breed. They are now weighted by
log-decreasing rank, the weighting the evolution-strategy literature settles on
— the population's shift from one generation to the next is a noisy estimate of
the direction fitness improves in, and this is the weighting that estimates it
best. The effective pool size (`1 / Σw²`) goes from 30 to 16.6, so the ranking
is used without the diversity collapsing.

Mutation size was a fixed 0.3 against a typical weight of 0.27 — a mutated gene
was not perturbed but effectively redrawn, and it was as violent at generation
120 as at generation 1, so nothing could ever be refined and only the untouched
elites carried progress. It now decays geometrically to 0.03 across the run:
`σ/|w|` runs 1.11 → 0.11. Geometrically rather than linearly because the useful
quantity is the ratio of the step to the weights, so equal fractions of the run
should shrink it by equal factors.

**Against memorisation.** Each generation starts from a different point on the
lap and meets a different island layout (four pre-built arrangements per
circuit, rotated). Without this the population learned the sequence of
manoeuvres for those exact laps rather than learning to drive: it scored 2.9
laps in training and **0.12** on the unseen circuit, dying on the very first
island. With it, training and race scores match.

## What makes the comparison fair

Each entrant keeps its own population — crossing one entrant's genome with
another's is meaningless, since the sizes differ and at equal size the genes
would mean different things — but there is **one draw per evaluation**: one
circuit, one island layout, one start point, shared by everyone. Parity is a
property of the loop rather than a convention someone has to remember.

Training each entrant in its own run did not give this, even on one seed. Start
points came from the same generator as the genomes and the breeding, so how many
numbers had already been drawn — which depends on genome size — decided where
the cars started, and entrants of different sizes met different starts from
generation 1. Start points now come from a stream of their own.

**Checkpoints name their own entrant.** `outputs/<name>.npz` carries the
registry name and the full spec, not just the layer sizes. Saving only
`hidden_sizes` silently dropped every other choice — `symmetric` above all,
which changes what the same weights compute — so a genome trained one way was
raced another.

## Watching it, and the cores

With several entrants the population is a few hundred cars, which is a swarm
nobody can read, and drawing it every frame is slow. So the worker pool scores
every full population across all cores while the window draws a readable subset
— `--shown`, eight per entrant by default, taken from the front of each
population, where elitism keeps the best genomes. Those cars are simulated twice,
once on screen and once in a worker, which costs a few dozen cars' arithmetic
and gives identical numbers: every step is deterministic given the track, the
start and the genome.

What the cores cannot fix is that watching 120 generations takes 120 generations
of wall-clock time. For a real run use `--headless`.

## Dashboard

```
+--------------------------+---------+
|         circuit          | panel   |
+--------------------------+---------+
|      analyses, full width          |
+------------------------------------+
```

1380×980 given the room: the circuit and the standings side by side, the
analyses across the whole width beneath both.

**It rarely has the room.** That window is taller than a 1440×900 laptop can
show, so `renderer.fit_window` scales the circuit down until the strip fits —
on a 900 px display, to 76%, giving a 1137×810 window. The circuit gives way
rather than the analyses, because the analyses are what the strip is for; only
when even a floored circuit leaves nothing readable is the strip dropped. The
circuit is drawn at its native 1000×700 onto an off-screen surface and scaled
once on the way out, so no car, ray or start line has to know about it.

**Panel.** Generation, current circuit (`TRACK k/3`), progress through the
frame budget, then one card per entrant in its colour, ordered by who is
furthest round: laps, parameter count, best so far, how many of the drawn cars
are alive — and its own network beside the text. Beside, not under: nodes are
laid out down the card, so a diagram wants height, and beside the text it gets
the whole card rather than what three rows of text leave over.

**Strip.** Three questions a neuroevolution run cannot be read without, plus
the numbers that are not curves:

- **Best so far.** The running maximum per entrant, which is monotonic and is
  the only line here that means progress. The raw per-generation best is kept
  faintly behind it: it swings with the draw rather than with the search — all
  the entrants rise and fall together on it — but the *differences* between the
  faint lines are where one entrant handles a hard draw better than another.
- **Genetic spread.** Mean per-gene standard deviation of the population,
  normalised to each entrant's own first generation so different genome sizes
  compare on one axis. A line on the floor means the population has converged
  to one genome in many copies and every further generation buys nothing.
- **Best per circuit.** Grouped by circuit, one bar per entrant. Fitness is
  `worst + 0.35 × mean`, so an entrant's solid bar is its weakest circuit —
  the one actually leading its number, and until now summed away before anyone
  could see it.
- **Run.** Generation, the *current* mutation size (it anneals, so it is a
  different number every generation), evaluations spent, elapsed and remaining.
  The estimate divides by *finished* generations: dividing by the one in
  progress made it climb steadily and jump back at each boundary, because the
  elapsed time grows while the divisor does not.

Everything grows by itself when an entrant is added.

## The race

Every champion goes onto `gauntlet` together, one colour each, with the
classification written to the log. Cars have no collision with one another — the
physics never modelled it — so they are spaced 26 px apart on the grid purely to
stay visible; distance is measured from each car's own start, so the stagger
costs nobody progress, but it does mean they meet each corner a few frames apart.

## Moving hazards, and why there are none

Timed traffic lights and crossing pedestrians were built, measured and removed.
Traffic lights made waiting at a red barrier dominate the run time, so most of
each evaluation was spent stationary. Pedestrians proved unlearnable for a
memoryless network: seeing only instantaneous distance, it cannot tell one
walking into its path from one standing still — seven of eight race attempts
ended in a pedestrian collision, with zero off-track deaths.

The hazards the cars do face are static: islands cut into the carriageway, which
the ray sensors see as walls for free. The implementation of the moving ones sat
unreachable behind a config flag for a long time; it is in the history rather
than in the tree — `git log -- src/obstacles.py` — because a module no code path
can reach is not a feature, it is a claim the tests can no longer check.

## Differences from the original project

- Circuits are generated procedurally rather than drawn in Photoshop, so the
  project is reproducible with no external assets.
- Continuous outputs (steering, throttle) instead of discrete actions.
- Islands in the carriageway are an addition; the original had only the verges.
