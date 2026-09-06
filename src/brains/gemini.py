"""Dual-stream residual network with decoupled steering and throttle paths."""

from typing import Any, Mapping

import numpy as np

from src.brains import Topology, register_brain


@register_brain("gemini")
class GeminiBrain:
    """Decoupled dual-stream architecture with direct linear skip connections."""

    def __init__(self, input_size: int, spec: Mapping[str, Any], rng: np.random.Generator) -> None:
        steer_hidden = int(spec.get("steer_hidden", 6))
        throt_hidden = int(spec.get("throt_hidden", 6))

        scale_in = 1.0 / np.sqrt(input_size)
        scale_steer = 1.0 / np.sqrt(steer_hidden)
        scale_throt = 1.0 / np.sqrt(throt_hidden)

        self.w_steer1 = rng.normal(0.0, scale_in, size=(input_size, steer_hidden))
        self.b_steer = rng.normal(0.0, scale_in, size=steer_hidden)

        self.w_throt1 = rng.normal(0.0, scale_in, size=(input_size, throt_hidden))
        self.b_throt = rng.normal(0.0, scale_in, size=throt_hidden)

        self.w_steer2 = rng.normal(0.0, scale_steer, size=(steer_hidden, 2))
        self.w_throt2 = rng.normal(0.0, scale_throt, size=(throt_hidden, 2))

        self.w_skip = rng.normal(0.0, scale_in, size=(input_size, 2))
        self.b_out = rng.normal(0.0, scale_in, size=2)

    def _targets(self) -> tuple[tuple[str, np.ndarray], ...]:
        """The parameters, in genome order. The one place that order is written."""
        return (
            ("w_steer1", self.w_steer1),
            ("b_steer", self.b_steer),
            ("w_throt1", self.w_throt1),
            ("b_throt", self.b_throt),
            ("w_steer2", self.w_steer2),
            ("w_throt2", self.w_throt2),
            ("w_skip", self.w_skip),
            ("b_out", self.b_out),
        )

    @property
    def genome_size(self) -> int:
        return sum(target.size for _, target in self._targets())

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        h_steer = np.tanh(inputs @ self.w_steer1 + self.b_steer)
        h_throt = np.tanh(inputs @ self.w_throt1 + self.b_throt)
        out = (
            inputs @ self.w_skip
            + h_steer @ self.w_steer2
            + h_throt @ self.w_throt2
            + self.b_out
        )
        return np.tanh(out)

    def get_genome(self) -> np.ndarray:
        return np.concatenate([target.ravel() for _, target in self._targets()])

    def set_genome(self, genome: np.ndarray) -> None:
        if genome.size != self.genome_size:
            raise ValueError(f"Genome size {genome.size} does not match {self.genome_size}")
        offset = 0
        for name, target in self._targets():
            setattr(self, name, genome[offset : offset + target.size].reshape(target.shape))
            offset += target.size

    def describe(self) -> Topology:
        return Topology(
            layers=(
                ("sensors", self.w_steer1.shape[0], 0),
                ("steer_hidden", self.w_steer1.shape[1], 1),
                ("throt_hidden", self.w_throt1.shape[1], 1),
                ("controls", 2, 2),
            ),
            edges=(
                (0, 1, self.w_steer1),
                (0, 2, self.w_throt1),
                (1, 3, self.w_steer2),
                (2, 3, self.w_throt2),
                (0, 3, self.w_skip),
            ),
        )
