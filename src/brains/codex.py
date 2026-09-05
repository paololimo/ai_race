"""Geometry-biased memoryless controller for the race harness."""

from typing import Any, Mapping

import numpy as np

from src.brains import register_brain


@register_brain("codex")
class CodexBrain:
    """Small symmetric tanh MLP over raw rays plus local road-shape features."""

    _FEATURES = 16

    def __init__(
        self, input_size: int, spec: Mapping[str, Any], rng: np.random.Generator
    ) -> None:
        hidden1 = int(spec.get("hidden1", 12))
        hidden2 = int(spec.get("hidden2", 8))

        self.w1 = rng.normal(0.0, 1.0 / np.sqrt(self._FEATURES), size=(self._FEATURES, hidden1))
        self.b1 = rng.normal(0.0, 1.0 / np.sqrt(self._FEATURES), size=hidden1)
        self.w2 = rng.normal(0.0, 1.0 / np.sqrt(hidden1), size=(hidden1, hidden2))
        self.b2 = rng.normal(0.0, 1.0 / np.sqrt(hidden1), size=hidden2)
        self.w3 = rng.normal(0.0, 1.0 / np.sqrt(hidden2), size=(hidden2, 2))
        self.b3 = rng.normal(0.0, 1.0 / np.sqrt(hidden2), size=2)

    @property
    def genome_size(self) -> int:
        return (
            self.w1.size
            + self.b1.size
            + self.w2.size
            + self.b2.size
            + self.w3.size
            + self.b3.size
        )

    def _features(self, inputs: np.ndarray) -> np.ndarray:
        rays = inputs[:7]
        speed = inputs[7]
        left = float(np.mean(rays[:3]))
        front = float(np.mean(rays[2:5]))
        right = float(np.mean(rays[4:7]))
        tight_front = 1.0 - float(np.min(rays[2:5]))
        tight_all = 1.0 - float(np.min(rays))
        return np.array(
            [
                rays[0],
                rays[1],
                rays[2],
                rays[3],
                rays[4],
                rays[5],
                rays[6],
                speed,
                speed * speed,
                float(np.mean(rays)),
                left,
                front,
                right,
                right - left,
                tight_front,
                tight_all,
            ],
            dtype=float,
        )

    def _pass(self, inputs: np.ndarray) -> np.ndarray:
        x = self._features(inputs)
        h1 = np.tanh(x @ self.w1 + self.b1)
        h2 = np.tanh(h1 @ self.w2 + self.b2)
        return np.tanh(h2 @ self.w3 + self.b3)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        mirrored = inputs.copy()
        mirrored[:7] = mirrored[6::-1]
        direct = self._pass(inputs)
        reflected = self._pass(mirrored)
        return np.array(
            [(direct[0] - reflected[0]) * 0.5, (direct[1] + reflected[1]) * 0.5],
            dtype=float,
        )

    def get_genome(self) -> np.ndarray:
        return np.concatenate(
            [
                self.w1.ravel(),
                self.b1,
                self.w2.ravel(),
                self.b2,
                self.w3.ravel(),
                self.b3,
            ]
        )

    def set_genome(self, genome: np.ndarray) -> None:
        if genome.size != self.genome_size:
            raise ValueError(f"Genome size {genome.size} does not match {self.genome_size}")
        offset = 0
        for name, target in (
            ("w1", self.w1),
            ("b1", self.b1),
            ("w2", self.w2),
            ("b2", self.b2),
            ("w3", self.w3),
            ("b3", self.b3),
        ):
            setattr(self, name, genome[offset : offset + target.size].reshape(target.shape))
            offset += target.size
