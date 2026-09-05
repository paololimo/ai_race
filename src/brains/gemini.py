"""Decoupled Residual Symmetric Brain for self-driving car evolutionary race."""

from typing import Any, Mapping

import numpy as np

from src.brains import register_brain


@register_brain("gemini")
class GeminiFlashBrain:
    """Decoupled dual-stream network with residual skip connections and exact reflection symmetry.

    Steering and throttle/braking are decoupled into dedicated sub-networks to prevent
    destructive pleiotropy during mutation and crossover. Direct skip connections from
    sensors to outputs provide an immediate linear baseline (proportional wall-following
    and lookahead speed control), while nonlinear hidden layers learn corner anticipation
    and obstacle negotiation. Bilateral symmetry enforces exact anti-symmetric steering
    and symmetric throttle.
    """

    def __init__(
        self, input_size: int, spec: Mapping[str, Any], rng: np.random.Generator
    ) -> None:
        """Initialize parameters using the harness rng."""
        self.input_size = input_size
        self._mirror_span = input_size - 1  # 7 rays, index 7 is speed

        hidden = int(spec.get("hidden", 10))
        self.hidden = hidden

        scale = float(spec.get("weight_init_scale", 1.0)) / np.sqrt(input_size)
        h_scale = 1.0 / np.sqrt(hidden)

        # Steering stream: dedicated weights
        self.w_s1 = rng.normal(0.0, scale, size=(input_size, hidden))
        self.b_s1 = rng.normal(0.0, scale, size=hidden)
        self.w_s2 = rng.normal(0.0, h_scale, size=(hidden, 1))
        self.w_s_skip = rng.normal(0.0, scale, size=(input_size, 1))
        self.b_s = rng.normal(0.0, scale, size=1)

        # Throttle stream: dedicated weights
        self.w_t1 = rng.normal(0.0, scale, size=(input_size, hidden))
        self.b_t1 = rng.normal(0.0, scale, size=hidden)
        self.w_t2 = rng.normal(0.0, h_scale, size=(hidden, 1))
        self.w_t_skip = rng.normal(0.0, scale, size=(input_size, 1))
        self.b_t = rng.normal(0.0, scale, size=1)

        self._param_names = (
            "w_s1",
            "b_s1",
            "w_s2",
            "w_s_skip",
            "b_s",
            "w_t1",
            "b_t1",
            "w_t2",
            "w_t_skip",
            "b_t",
        )

    @property
    def genome_size(self) -> int:
        """Total number of learnable parameters."""
        return sum(getattr(self, name).size for name in self._param_names)

    def _raw_forward(self, inputs: np.ndarray) -> np.ndarray:
        """Un-symmetrised forward pass computing [raw_steer, raw_throttle]."""
        # Steering pathway
        h_s = np.tanh(inputs @ self.w_s1 + self.b_s1)
        steer = np.tanh(h_s @ self.w_s2 + inputs @ self.w_s_skip + self.b_s)[0]

        # Throttle pathway
        h_t = np.tanh(inputs @ self.w_t1 + self.b_t1)
        throttle = np.tanh(h_t @ self.w_t2 + inputs @ self.w_t_skip + self.b_t)[0]

        return np.array([steer, throttle], dtype=float)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Symmetrised forward pass returning [steering, throttle] in [-1, 1]."""
        # Mirror inputs left-to-right across the 7 ray sensors
        mirrored = inputs.copy()
        mirrored[: self._mirror_span] = mirrored[self._mirror_span - 1 :: -1]

        direct = self._raw_forward(inputs)
        reflected = self._raw_forward(mirrored)

        # Enforce exact antisymmetry for steering and symmetry for throttle
        steering = (direct[0] - reflected[0]) / 2.0
        throttle = (direct[1] + reflected[1]) / 2.0

        return np.array([steering, throttle], dtype=float)

    def get_genome(self) -> np.ndarray:
        """Flatten all learnable weights into a single 1-D vector."""
        return np.concatenate([getattr(self, name).ravel() for name in self._param_names])

    def set_genome(self, genome: np.ndarray) -> None:
        """Unpack a flat 1-D genome back into the network weights."""
        if genome.size != self.genome_size:
            raise ValueError(
                f"Genome size {genome.size} does not match expected {self.genome_size}"
            )
        offset = 0
        for name in self._param_names:
            target = getattr(self, name)
            size = target.size
            setattr(self, name, genome[offset : offset + size].reshape(target.shape))
            offset += size
