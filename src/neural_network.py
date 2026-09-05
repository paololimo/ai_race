"""Feedforward neural network written from scratch (no ML libraries)."""

from typing import List, Tuple

import numpy as np

from src.config import NetworkConfig


class NeuralNetwork:
    """Dense feedforward network with tanh activations.

    Weights are never trained by gradient descent: they are treated as a genome
    that the genetic algorithm recombines and mutates.
    """

    def __init__(self, input_size: int, cfg: NetworkConfig, rng: np.random.Generator) -> None:
        self.layer_sizes: Tuple[int, ...] = (input_size, *cfg.hidden_sizes, cfg.output_size)
        self.symmetric = cfg.symmetric
        # Every input but the last is a sensor ray; the last one is own speed,
        # which is unchanged by mirroring the car left-to-right.
        self._mirror_span = input_size - 1
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        for n_in, n_out in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            scale = cfg.weight_init_scale / np.sqrt(n_in)
            self.weights.append(rng.normal(0.0, scale, size=(n_in, n_out)))
            self.biases.append(rng.normal(0.0, scale, size=n_out))

    def _pass(self, inputs: np.ndarray) -> np.ndarray:
        activation = inputs
        for weight, bias in zip(self.weights, self.biases):
            activation = np.tanh(activation @ weight + bias)
        return activation

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Run one forward pass, returning outputs in the range (-1, 1)."""
        if not self.symmetric:
            return self._pass(inputs)

        # Left/right symmetry, imposed rather than learned. The track poses a
        # mirror-symmetric problem: the response to a wall on the left is the
        # mirror of the response to a wall on the right. Averaging the network
        # against its own mirror image makes steering exactly antisymmetric and
        # throttle exactly symmetric, so the weights never have to learn the
        # same rule twice — the search space shrinks without losing capacity.
        mirrored = inputs.copy()
        mirrored[: self._mirror_span] = mirrored[self._mirror_span - 1 :: -1]
        direct = self._pass(inputs)
        reflected = self._pass(mirrored)
        return np.array(
            [(direct[0] - reflected[0]) / 2.0, (direct[1] + reflected[1]) / 2.0]
        )

    @property
    def genome_size(self) -> int:
        return sum(w.size for w in self.weights) + sum(b.size for b in self.biases)

    def get_genome(self) -> np.ndarray:
        """Flatten all weights and biases into a single 1-D vector."""
        parts = [w.ravel() for w in self.weights] + [b.ravel() for b in self.biases]
        return np.concatenate(parts)

    def set_genome(self, genome: np.ndarray) -> None:
        """Load a flat vector produced by `get_genome` back into the network."""
        if genome.size != self.genome_size:
            raise ValueError(f"Genome size {genome.size} does not match network {self.genome_size}")
        offset = 0
        for i, weight in enumerate(self.weights):
            self.weights[i] = genome[offset : offset + weight.size].reshape(weight.shape)
            offset += weight.size
        for i, bias in enumerate(self.biases):
            self.biases[i] = genome[offset : offset + bias.size].reshape(bias.shape)
            offset += bias.size
