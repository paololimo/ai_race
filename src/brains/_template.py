"""Copy this file to `src/brains/<yourname>.py` and make it drive.

Leading underscore: `src/brains/__init__` skips it when it auto-imports the
package, so the template itself is never registered.

Nothing outside this file may be edited. Circuits, physics, sensors, fitness,
the genetic algorithm and the seed are the harness's, identical for every
entrant, which is what makes the race a comparison between architectures.
"""

from typing import Any, Mapping

import numpy as np

from src.brains import register_brain


@register_brain("template")  # the name passed to `train.py --brain <name>`
class TemplateBrain:
    """One-line description of the architecture and why it should drive well."""

    def __init__(
        self, input_size: int, spec: Mapping[str, Any], rng: np.random.Generator
    ) -> None:
        """`input_size` is 8: seven ray distances in [0, 1], then speed in [0, 1].

        Read every hyperparameter from `spec`, with a default — `spec` is empty
        unless `train.py --brain-spec` is given, and it is written verbatim into
        the checkpoint, so it must stay JSON-serialisable.

        Draw the initial parameters from `rng` and from nothing else: the
        harness owns every other source of randomness, and a brain that seeds
        its own breaks reproducibility across worker processes.
        """
        hidden = int(spec.get("hidden", 16))
        scale = 1.0 / np.sqrt(input_size)
        self.w1 = rng.normal(0.0, scale, size=(input_size, hidden))
        self.b1 = rng.normal(0.0, scale, size=hidden)
        self.w2 = rng.normal(0.0, scale, size=(hidden, 2))
        self.b2 = rng.normal(0.0, scale, size=2)

    @property
    def genome_size(self) -> int:
        """Total number of learnable parameters. Must never change per instance."""
        return self.w1.size + self.b1.size + self.w2.size + self.b2.size

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Sensor readings in, `[steering, throttle]` out, both within [-1, 1].

        Called once per car per frame — millions of times per generation — so
        this is the one place where speed matters. No state may be carried
        between calls unless the architecture is deliberately recurrent, and
        even then it must be reset by `set_genome`.
        """
        hidden = np.tanh(inputs @ self.w1 + self.b1)
        return np.tanh(hidden @ self.w2 + self.b2)

    def get_genome(self) -> np.ndarray:
        """Every parameter, flattened into one 1-D float array."""
        return np.concatenate([self.w1.ravel(), self.b1, self.w2.ravel(), self.b2])

    def set_genome(self, genome: np.ndarray) -> None:
        """Load a vector from `get_genome` back in. Must be its exact inverse."""
        if genome.size != self.genome_size:
            raise ValueError(f"Genome size {genome.size} does not match {self.genome_size}")
        offset = 0
        for name, target in (("w1", self.w1), ("b1", self.b1), ("w2", self.w2), ("b2", self.b2)):
            setattr(self, name, genome[offset : offset + target.size].reshape(target.shape))
            offset += target.size
