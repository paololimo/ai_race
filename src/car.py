"""Car physics, ray sensors and fitness accounting."""

import math
from typing import List, Optional, Tuple

import numpy as np

from src.brains import Brain, BrainRef, build_brain
from src.config import CarConfig
from src.track import Track

_HIT_THRESHOLD = 1.0  # clearance at or below this counts as the verge
_MIN_STEP = 1.0


def _sensor_fan(cfg: CarConfig) -> Tuple[float, Tuple[float, ...]]:
    """The fixed ray fan: a half-spread offset and one step per ray.

    Kept in the two pieces the caller adds to its heading, rather than folded
    into a single offset, so the arithmetic stays `heading - half + step` — the
    same association, and so the same floats, as evaluating it inline.
    """
    n = cfg.num_sensors
    if n == 1:
        return 0.0, (0.0,)
    spread = math.radians(cfg.sensor_spread)
    step = spread / (n - 1)
    return spread / 2, tuple(i * step for i in range(n))


def network_input_size(cfg: CarConfig) -> int:
    """One distance per ray, plus own speed.

    Static islands are cut into the road itself, so the wall rays already see
    them. A separate obstacle channel existed while hazards moved; it is no
    longer needed and the smaller genome evolves faster.
    """
    return cfg.num_sensors + 1


class Car:
    """A single agent: sensors feed a network, the network drives the car."""

    def __init__(
        self,
        track: Track,
        brain: Brain,
        cfg: CarConfig,
        start_index: Optional[int] = None,
    ) -> None:
        self.cfg = cfg
        self.track = track
        self.brain = brain
        index = 0 if start_index is None else start_index
        self.x, self.y = track.position_at_index(index)
        self.angle = track.heading_at_index(index)
        self.speed = 0.0
        self.alive = True
        self.frames = 0
        self._idle_frames = 0
        self._index = track.nearest_index(self.x, self.y)
        self.progress = 0.0  # distance travelled along the centreline, in pixels
        self.wall_readings: List[float] = [1.0] * cfg.num_sensors
        self.sensor_endpoints: List[Tuple[float, float]] = []
        # Bound once: the ray marcher below reads these on every step of every
        # ray, and at that volume the attribute chains cost more than the
        # arithmetic. Nothing here changes over a car's lifetime.
        self._field = track.clearance_flat
        self._width, self._height = track.cfg.width, track.cfg.height
        self._reach = cfg.sensor_range
        self._half_spread, self._ray_steps = _sensor_fan(cfg)

    @property
    def fitness(self) -> float:
        """Distance travelled along the track, plus a small tie-breaking bonus."""
        return max(0.0, self.progress) + 0.01 * self.frames

    @property
    def laps(self) -> float:
        return self.progress / self.track.lap_length

    def _cast_wall_ray(self, dx: float, dy: float) -> Tuple[float, Tuple[float, float]]:
        """Sphere-trace a ray to the verge; return the distance and the hit point.

        Instead of stepping a fixed 3px at a time, each step jumps by the
        precomputed clearance at the current point — the distance guaranteed to
        be free of any verge. Wide open road resolves in two or three jumps
        where marching took dozens; near a wall the clearance shrinks and the
        steps become fine again, so precision is kept exactly where it matters.
        """
        # Local names and a flat array: this loop runs millions of times per
        # generation, and at that volume the attribute lookups and the function
        # call cost more than the arithmetic inside it. The clearance field is
        # read one cell at a time and nothing else, so it is held as a flat
        # `array.array` of Python floats — indexing a numpy array here returns a
        # float32 scalar and turns every comparison and addition that follows
        # into numpy arithmetic on a zero-dimensional object, which measured 2.3
        # times the cost of the same loop over machine floats.
        reach = self._reach
        field, width, height = self._field, self._width, self._height
        x, y = self.x, self.y

        distance = 0.0
        while distance < reach:
            px = x + dx * distance
            py = y + dy * distance
            ix, iy = int(px), int(py)
            if ix < 0 or iy < 0 or ix >= width or iy >= height:
                return distance, (px, py)
            clearance = field[ix * height + iy]
            if clearance <= _HIT_THRESHOLD:
                return distance, (px, py)
            step = clearance - _HIT_THRESHOLD
            distance += step if step > _MIN_STEP else _MIN_STEP
        return reach, (x + dx * reach, y + dy * reach)

    def sense(self) -> np.ndarray:
        """Build the network input: one distance per ray, plus own speed."""
        walls: List[float] = []
        endpoints: List[Tuple[float, float]] = []

        heading = self.angle - self._half_spread
        for ray_step in self._ray_steps:
            angle = heading + ray_step
            distance, endpoint = self._cast_wall_ray(math.cos(angle), math.sin(angle))
            walls.append(distance / self._reach)
            # Back to Python floats. The clearance field is float32, so a ray
            # that took at least one step from it returns float32 coordinates,
            # which pygame refuses to draw — and only some rays do, which is why
            # the window died on one frame and not the next.
            endpoints.append((float(endpoint[0]), float(endpoint[1])))

        self.wall_readings = walls
        self.sensor_endpoints = endpoints
        return np.array(walls + [self.speed / self.cfg.max_speed], dtype=float)

    def _apply_controls(self, steering: float, throttle: float) -> None:
        if throttle >= 0.0:
            self.speed += self.cfg.acceleration * throttle
        else:
            self.speed += self.cfg.braking * throttle  # negative throttle brakes
        self.speed -= self.cfg.friction
        self.speed = min(max(self.speed, 0.0), self.cfg.max_speed)
        # Steering authority grows with speed up to a low reference speed, then
        # saturates. Past that point the turning radius is speed / turn rate, so
        # slowing down is what buys a tighter corner.
        authority = min(1.0, self.speed / self.cfg.steering_ref_speed)
        turn = math.radians(self.cfg.max_steering) * steering * authority
        self.angle += turn
        self.x += math.cos(self.angle) * self.speed
        self.y += math.sin(self.angle) * self.speed

    def _update_progress(self) -> None:
        """Project onto the centreline and accumulate the distance moved along it."""
        n = len(self.track.samples)
        current = self.track.nearest_index(self.x, self.y, hint=self._index)
        step = (current - self._index + n // 2) % n - n // 2  # shortest signed step
        self.progress += step * self.track.cfg.sample_spacing
        self._index = current

    def _update_idle(self) -> None:
        """Standing still must not be a way to survive without driving."""
        if self.speed >= self.cfg.idle_speed_threshold:
            self._idle_frames = 0
            return
        self._idle_frames += 1
        if self._idle_frames >= self.cfg.idle_frames_allowed:
            self.alive = False

    def update(self) -> None:
        """Advance one frame: sense, think, move, then check for a crash."""
        if not self.alive:
            return
        outputs = self.brain.forward(self.sense())
        self._apply_controls(steering=float(outputs[0]), throttle=float(outputs[1]))
        self._update_progress()
        self.frames += 1

        # Islands are part of the road mask, so hitting one counts as off-road.
        if not self.track.is_on_road(self.x, self.y):
            self.alive = False
        else:
            self._update_idle()


def build_car(
    ref: BrainRef,
    genome: np.ndarray,
    track: Track,
    cfg: CarConfig,
    input_size: int,
    rng: np.random.Generator,
    start_index: Optional[int],
) -> Car:
    """Put one genome on the grid.

    The serial path and the pooled workers both need this, and the guarantee
    that a parallel run reproduces a serial one exactly rests on them doing it
    the same way — including the dedicated `rng`, whose draws `set_genome`
    immediately overwrites but which must not come from a shared stream.
    """
    driver = build_brain(ref, input_size, rng)
    driver.set_genome(genome)
    return Car(track, driver, cfg, start_index)
