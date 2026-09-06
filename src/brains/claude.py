"""A mirror-equivariant dense net: one shared policy evaluated on the road as
seen and on its mirror image, recombined so steering is odd and throttle even."""

from typing import Any, Mapping

import numpy as np

from src.brains import FlatGenome, Topology, register_brain


@register_brain("claude")
class MirrorBrain(FlatGenome):
    """Sensors to controls through a single small net used twice.

    The track does not care which way round it is drawn: mirror a corner and the
    right answer is the same throttle with the opposite steering. That is a
    symmetry of the physics, not a fact about any particular circuit, so it is
    worth spending architecture on rather than genes.

    One dense stream — inputs to a tanh hidden layer, plus a direct linear skip
    from the inputs to the two controls — is evaluated on the sensor vector and
    on the left-right reflection of it. The two answers are then combined
    antisymmetrically for steering and symmetrically for throttle. Every weight
    is free and evolution sets all of them; what the wiring fixes is only that
    whatever policy evolution finds is applied identically to both hands of a
    corner. Nothing about braking, gap-picking or turn direction is computed
    here — the constraint is a symmetry, not a decision.
    """

    PARAMS = ("w1", "b1", "w2", "ws", "b2")

    def __init__(self, input_size: int, spec: Mapping[str, Any], rng: np.random.Generator) -> None:
        hidden = int(spec.get("hidden", 12))
        self.input_size = int(input_size)
        # Every input but the last is a ray, and the rays run leftmost to
        # rightmost, so reflecting the car's view is reversing that block and
        # leaving own speed — which has no handedness — alone.
        self.rays = self.input_size - 1

        scale = 1.0 / np.sqrt(self.input_size)
        self.w1 = rng.normal(0.0, scale, size=(self.input_size, hidden))
        self.b1 = rng.normal(0.0, scale, size=hidden)
        self.w2 = rng.normal(0.0, 1.0 / np.sqrt(hidden), size=(hidden, 2))
        # The skip is a whole linear policy in eighteen genes: coarse, but it
        # drives, and uniform crossover mixes two linear policies far more
        # gently than it mixes two sets of co-adapted hidden units.
        self.ws = rng.normal(0.0, scale, size=(self.input_size, 2))
        self.b2 = rng.normal(0.0, scale, size=2)

        # Scratch buffer for the two views; `forward` runs millions of times a
        # generation and this keeps it to two matmuls and no allocation.
        self._pair = np.empty((2, self.input_size), dtype=float)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Sensors in, `[steering, throttle]` out, both within [-1, 1]."""
        pair = self._pair
        pair[0] = inputs
        pair[1, : self.rays] = pair[0, self.rays - 1 :: -1]
        pair[1, self.rays :] = pair[0, self.rays :]

        hidden = np.tanh(pair @ self.w1 + self.b1)
        out = np.tanh(hidden @ self.w2 + pair @ self.ws + self.b2)

        # Half the difference and half the sum of two numbers already in
        # [-1, 1] are themselves in [-1, 1], so the bound needs no clipping:
        # it holds for every genome, which is the point of taking it from the
        # structure rather than from a guard at the end.
        steering = 0.5 * (out[0, 0] - out[1, 0])
        throttle = 0.5 * (out[0, 1] + out[1, 1])
        return np.array([steering, throttle])

    def describe(self) -> Topology:
        """The shape the dashboard draws: one stream, one skip, applied twice."""
        hidden = self.b1.size
        return Topology(
            layers=(
                ("sensors", self.input_size, 0),
                ("hidden", hidden, 1),
                ("drive", 2, 2),
            ),
            edges=((0, 1, self.w1), (1, 2, self.w2), (0, 2, self.ws)),
        )
