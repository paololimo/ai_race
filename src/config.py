"""Immutable configuration for the evolutionary self-driving car project."""

from dataclasses import dataclass, field
from typing import Tuple


# One asphalt tone across every circuit: only the surroundings change colour.
_ASPHALT = (44, 44, 52)


@dataclass(frozen=True)
class TrackConfig:
    """Geometry of the generated serpentine racetrack.

    Lanes run alternately left and right at the `lane_ys` heights, joined by
    180-degree hairpins, with a return corridor at `return_x` closing the lap.
    """

    width: int = 1000
    height: int = 700
    road_width: float = 78.0
    # Road width breathes along the lap: narrow chicanes, wide sweepers.
    width_variation: float = 0.18
    width_frequency: int = 5
    # Serpentine layout: lanes run left/right, joined by 180-degree hairpins,
    # with a return corridor down the left-hand side.
    lane_x_left: float = 340.0
    lane_x_right: float = 855.0
    # An even number of lanes, so the last one ends on the left where the
    # return corridor starts; an odd count makes the circuit cross itself.
    lane_ys: Tuple[float, ...] = (95.0, 245.0, 395.0, 545.0)
    return_x: float = 150.0
    hairpin_steps: int = 10  # arc resolution of each 180-degree turn
    return_y_bottom: float = 640.0
    # Bumps break up the long straights so they are not trivially fast. Keep
    # below (lane spacing - road_width - 10) / 2, or adjacent lanes merge.
    # Several bumps per straight keep the car turning continuously, which is
    # what the race circuit demands; a single bump leaves the lap mostly
    # straight and the network never learns sustained cornering.
    bump_amplitude: float = 30.0
    bumps_per_straight: int = 3
    # Chaikin passes: more passes round the hairpins off more smoothly.
    smoothing_passes: int = 4
    sample_spacing: float = 4.0
    mirrored: bool = False  # flip horizontally, for a different-feeling circuit
    # "serpentine" (lanes + hairpins), "grid" (right-angle city circuit),
    # "speedway" (fast sweepers + one chicane) or "zigzag" (chicanes throughout)
    layout: str = "serpentine"
    # Zigzag layout only: peaks on the outward and return runs, and their reach.
    zigzag_peaks: Tuple[int, int] = (5, 7)
    zigzag_amplitude: float = 90.0
    # Static islands sitting in the carriageway: the car must pick a side.
    # Given as fractions of the lap; shapes alternate round and angular.
    islands: Tuple[float, ...] = ()
    # Islands are sized from the local road width so that a corridor at least
    # this wide is left on each side; where that leaves no room, the island is
    # skipped rather than sealing the track off.
    island_min_corridor: float = 26.0
    name: str = "track"

    grass_color: Tuple[int, int, int] = (32, 90, 48)
    road_color: Tuple[int, int, int] = _ASPHALT
    line_color: Tuple[int, int, int] = (220, 220, 220)


def track_variants() -> Tuple[TrackConfig, ...]:
    """The three circuits every generation is evaluated on.

    They differ in structure, not just in scale: flowing hairpins, right-angle
    city corners, and fast sweepers broken by a chicane. Training on one shape
    and racing on another only proves the network memorised that shape.
    """
    return (
        TrackConfig(
            name="serpentine",
            layout="serpentine",
            road_width=88.0,
            islands=(0.14, 0.34, 0.56, 0.78, 0.92),
            grass_color=(32, 90, 48),
            road_color=_ASPHALT,
        ),
        TrackConfig(
            name="grid-city",
            layout="grid",
            road_width=92.0,
            width_variation=0.08,
            smoothing_passes=1,  # keep the corners square, not rounded away
            islands=(0.08, 0.29, 0.47, 0.66, 0.84),
            grass_color=(58, 96, 172),
            road_color=_ASPHALT,
        ),
        TrackConfig(
            name="speedway",
            layout="speedway",
            road_width=96.0,
            width_variation=0.16,
            smoothing_passes=3,
            islands=(0.18, 0.44, 0.62, 0.88),
            grass_color=(198, 84, 76),
            road_color=_ASPHALT,
        ),
    )


def race_track() -> TrackConfig:
    """The circuit used only for the final race: never trained on.

    A different shape again — chicanes from start to finish — so finishing it
    demonstrates general driving rather than a memorised layout.
    """
    return TrackConfig(
        name="gauntlet",
        layout="zigzag",
        road_width=90.0,
        width_variation=0.14,
        zigzag_peaks=(4, 5),
        zigzag_amplitude=72.0,
        smoothing_passes=2,
        islands=(0.12, 0.35, 0.58, 0.81),
        grass_color=(86, 62, 140),
        road_color=_ASPHALT,
    )


@dataclass(frozen=True)
class ObstacleConfig:
    """Timed hazards on the circuit: traffic lights and crossing pedestrians."""

    enabled: bool = True
    # Both moving hazard types are switched off. Traffic lights made waiting at
    # a red barrier dominate the run time; pedestrians proved unlearnable for a
    # memoryless network, which sees only instantaneous distance and so cannot
    # tell one walking into its path from one standing still. Obstacles are now
    # static islands cut into the carriageway (see TrackConfig.islands). The
    # classes below are kept intact — raise these counts to bring them back.
    num_traffic_lights: int = 0
    num_pedestrians: int = 0
    # Traffic light: a barrier across the road, solid only while red.
    light_period_frames: int = 260
    light_red_fraction: float = 0.45
    light_barrier_circles: int = 4
    # Pedestrian: a circle crossing the road back and forth.
    pedestrian_radius: float = 11.0
    pedestrian_speed: float = 0.9
    red_color: Tuple[int, int, int] = (220, 70, 70)
    green_color: Tuple[int, int, int] = (70, 200, 110)
    pedestrian_color: Tuple[int, int, int] = (240, 200, 90)


@dataclass(frozen=True)
class CarConfig:
    """Physical properties and sensor layout of a car."""

    num_sensors: int = 7
    sensor_range: float = 200.0
    sensor_spread: float = 180.0
    length: float = 18.0
    width: float = 10.0
    max_speed: float = 6.0
    acceleration: float = 0.25
    braking: float = 0.35
    friction: float = 0.04
    max_steering: float = 4.5
    # Above this speed the turn rate stops growing, so the turning radius is
    # `speed / max_steering`: going slower lets a car take a tighter corner.
    steering_ref_speed: float = 1.5
    idle_speed_threshold: float = 0.2
    idle_frames_allowed: int = 60


@dataclass(frozen=True)
class GeneticConfig:
    """Selection, crossover and mutation parameters."""

    population_size: int = 100
    elite_count: int = 4
    breeding_fraction: float = 0.3
    mutation_rate: float = 0.05
    mutation_scale: float = 0.3
    # "uniform", "one-point", "two-point" or "none". Shared by every entrant, so
    # changing it changes the experiment for all of them equally and advantages
    # nobody. See `genetic.crossover` for what each costs a neural genome.
    crossover: str = "uniform"


@dataclass(frozen=True)
class SimulationConfig:
    """Training-run level settings."""

    generations: int = 200
    # Frame budget is derived per circuit rather than fixed, so every track
    # grants the same number of *laps* instead of the same number of frames.
    # A flat budget favoured the short circuits: 1400 frames was 4.0 laps of
    # speedway but only 2.3 of serpentine, which distorted the comparison
    # between tracks. This is laps at top speed; in practice cars must slow for
    # corners and reach roughly two thirds of it.
    laps_budget: float = 3.0
    # Fitness across circuits is `worst + mean_weight * mean`: leading with the
    # worst circuit forces competence everywhere, while the mean term breaks
    # ties and gives early generations a gradient to follow.
    mean_weight: float = 0.35
    # Anti-memorisation. With fixed circuits, fixed island positions and a fixed
    # start, a population can learn the sequence of manoeuvres for those exact
    # laps instead of learning to drive — it scored 2.9 laps in training and
    # 0.12 on an unseen circuit, dying on the very first island. Each generation
    # now starts somewhere else on the lap and meets a different island layout.
    island_variants: int = 4
    random_start: bool = True
    fps: int = 60
    seed: int = 42
    # Curriculum: learn to drive first, then face the hazards.
    obstacle_start_generation: int = 30
    dashboard_width: int = 380
    checkpoint_dir: str = "outputs"
    # Everything a comparison rests on lives here and is shared by every
    # entrant. What an entrant brings is only its architecture, in one file
    # under `src/brains/`.
    track: TrackConfig = field(default_factory=TrackConfig)
    obstacles: ObstacleConfig = field(default_factory=ObstacleConfig)
    car: CarConfig = field(default_factory=CarConfig)
    genetic: GeneticConfig = field(default_factory=GeneticConfig)
