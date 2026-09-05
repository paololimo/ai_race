"""Claude's entrant: a mirror-symmetric network built in the symmetry basis itself.

Rather than averaging a free network against its own reflection — which imposes
the symmetry but still pays for the redundant half of the weights — the inputs
are split into even and odd parts up front and the network is wired so that
steering can only be odd and throttle can only be even. The symmetry is exact by
construction, the genome is about half the size, and the two physical priors the
circuits actually turn on — brake before a closing gap, steer towards the open
one — are handed over as features instead of being rediscovered on every track.
"""

from typing import Any, Mapping

import numpy as np

from src.brains import register_brain

# Ray directions as a fraction of the half-spread: -1 leftmost, 0 ahead, +1
# rightmost. Used to weight the free-space centroid, which is why the vector is
# exactly antisymmetric.
_RAY_AXIS = np.array([-1.0, -2.0 / 3.0, -1.0 / 3.0, 0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0], dtype=float)

_EVEN_FEATURES = 7
_ODD_FEATURES = 4
# Per hidden unit: 7 even weights, 4 odd weights, 1 bias, 1 steering readout,
# 1 throttle readout. One unit is therefore one contiguous 14-gene block.
_GENES_PER_UNIT = _EVEN_FEATURES + _ODD_FEATURES + 3
# Tail: the linear skip to steering (4), the linear skip to throttle (7), and
# the throttle bias (1). Steering gets no bias — a constant is even, and an even
# term in an odd output is exactly the drift that makes a car circle.
_TAIL = _ODD_FEATURES + _EVEN_FEATURES + 1

_EPS = 0.05


@register_brain("claude")
class ClaudeBrain:
    """Symmetry-adapted single-hidden-layer network with linear skips."""

    def __init__(
        self, input_size: int, spec: Mapping[str, Any], rng: np.random.Generator
    ) -> None:
        self.hidden = int(spec.get("hidden", 12))
        if self.hidden < 1:
            raise ValueError("hidden must be at least 1")
        self.n_rays = input_size - 1  # the last input is own speed
        self._axis = _RAY_AXIS if self.n_rays == 7 else np.linspace(-1.0, 1.0, self.n_rays)
        self._half = self.n_rays // 2
        init_scale = float(spec.get("weight_init_scale", 1.0))
        # A forward throttle bias at generation zero. A car that never opens the
        # throttle is killed as idle after 60 frames, so a population that starts
        # half-stationary spends its first generations learning only to move.
        self._drive_bias = float(spec.get("drive_bias", 0.3))

        # Column scales for one hidden unit's block, in gene order.
        scales = np.concatenate(
            [
                np.full(_EVEN_FEATURES, 1.0 / np.sqrt(_EVEN_FEATURES)),
                np.full(_ODD_FEATURES, 1.0 / np.sqrt(_ODD_FEATURES)),
                [0.3],                                  # hidden bias
                np.full(2, 1.0 / np.sqrt(self.hidden)),  # steering, throttle readouts
            ]
        )
        self.units = rng.normal(0.0, init_scale, size=(self.hidden, _GENES_PER_UNIT)) * scales
        tail_scales = np.concatenate(
            [
                np.full(_ODD_FEATURES, 1.0 / np.sqrt(_ODD_FEATURES)),
                np.full(_EVEN_FEATURES, 1.0 / np.sqrt(_EVEN_FEATURES)),
                [0.3],
            ]
        )
        self.tail = rng.normal(0.0, init_scale, size=_TAIL) * tail_scales
        self.tail[-1] += self._drive_bias

        # Scratch buffers, overwritten in full on every call. They carry nothing
        # from one frame to the next: this network is memoryless, so `set_genome`
        # has no state to reset.
        self._even = np.empty(_EVEN_FEATURES)
        self._odd = np.empty(_ODD_FEATURES)

    @property
    def genome_size(self) -> int:
        return self.units.size + self.tail.size

    def _split(self, inputs: np.ndarray) -> None:
        """Project the sensors onto the mirror-even and mirror-odd subspaces.

        Mirroring the car swaps ray i with ray n-1-i and leaves speed alone, so
        the half-sums below are invariant under it and the half-differences flip
        sign. Every downstream feature is built to keep that split intact.
        """
        rays = inputs[: self.n_rays]
        speed = inputs[self.n_rays]
        half = self._half
        left = rays[:half]
        right = rays[::-1][:half]

        even, odd = self._even, self._odd
        even[:half] = 0.5 * (left + right)
        even[half] = rays[half]                       # the centre ray is its own mirror
        odd[:half] = 0.5 * (left - right)

        # Physics, not geometry: the corner radius the car can hold is
        # speed / turn rate, so braking is what buys a tight corner and the
        # quantity that decides it is how much room is left ahead relative to
        # how fast that room is being consumed. Bounded in [0, 1) and free of
        # any units, so it means the same thing on a circuit never seen.
        ahead = float(np.min(rays[half - 1 : half + 2])) if self.n_rays >= 3 else float(rays[0])
        even[_EVEN_FEATURES - 3] = speed
        even[_EVEN_FEATURES - 2] = ahead
        even[_EVEN_FEATURES - 1] = speed / (speed + ahead + _EPS)

        # Follow-the-gap: the free-space centroid over the ray fan, squared to
        # weight open directions over blocked ones. This is the one steering
        # primitive that transfers unchanged between layouts — it points at the
        # widest opening whether that opening is a hairpin exit or the side of
        # an island — and it is not a linear combination of the half-differences,
        # so the hidden layer cannot cheaply build it for itself.
        weights = rays * rays
        odd[_ODD_FEATURES - 1] = float(weights @ self._axis) / (float(weights.sum()) + _EPS)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Sensors in, `[steering, throttle]` out, both within [-1, 1]."""
        self._split(inputs)
        even, odd, units = self._even, self._odd, self.units

        # One pre-activation for the unit and one for its mirror twin: they share
        # the even weights and differ only in the sign of the odd contribution.
        common = units[:, :_EVEN_FEATURES] @ even + units[:, _EVEN_FEATURES + _ODD_FEATURES]
        skew = units[:, _EVEN_FEATURES : _EVEN_FEATURES + _ODD_FEATURES] @ odd
        up = np.tanh(common + skew)
        down = np.tanh(common - skew)

        # The half-sum of a twin pair is even, the half-difference odd. Steering
        # reads only the odd half and throttle only the even half, so the mirror
        # image of any situation gets exactly the mirrored steering and exactly
        # the same throttle — no averaging pass, and no weights that the
        # symmetry would silently discard.
        steering = 0.5 * (up - down) @ units[:, -2] + odd @ self.tail[:_ODD_FEATURES]
        throttle = (
            0.5 * (up + down) @ units[:, -1]
            + even @ self.tail[_ODD_FEATURES:-1]
            + self.tail[-1]
        )
        return np.tanh(np.array([steering, throttle], dtype=float))

    def get_genome(self) -> np.ndarray:
        """Every parameter, flattened into one 1-D float array.

        Row-major, so a hidden unit's incoming weights, bias and both readouts
        stay in one contiguous block: crossover recombines whole units rather
        than halves of unrelated ones.
        """
        return np.concatenate([self.units.ravel(), self.tail])

    def set_genome(self, genome: np.ndarray) -> None:
        """The exact inverse of `get_genome`."""
        if genome.size != self.genome_size:
            raise ValueError(f"Genome size {genome.size} does not match {self.genome_size}")
        split = self.units.size
        self.units = genome[:split].reshape(self.units.shape)
        self.tail = genome[split:].copy()


__all__ = ["ClaudeBrain"]
