"""Paolo's entrant: a feedforward network written from scratch, no ML libraries.

The weights are never trained by gradient descent: they are a flat genome that
the harness's genetic algorithm recombines and mutates. Four architectural
choices are exposed through the spec so the ablation can measure them against
each other rather than have them settled by argument — `hidden_sizes`,
`symmetric`, `skip` and `decoupled`. The defaults are the incumbent design.
"""

from dataclasses import dataclass
from typing import Any, List, Mapping, Tuple

import numpy as np

from src.brains import Topology, register_brain


@dataclass(frozen=True)
class NetworkConfig:
    """Architecture of this entrant. Overridable through the checkpoint spec."""

    hidden_sizes: Tuple[int, ...] = (14, 14)
    output_size: int = 2
    weight_init_scale: float = 1.0
    # Impose left/right symmetry instead of letting the weights discover it.
    symmetric: bool = False
    # A direct input-to-output matrix alongside the hidden path. Evolution then
    # starts from a linear controller it can refine, instead of having to build
    # one through the hidden layers before anything works at all.
    skip: bool = False
    # A separate hidden stack per control. A mutation that improves braking then
    # cannot damage steering, because it lands in weights steering never reads.
    decoupled: bool = False


def spec_from_config(cfg: NetworkConfig) -> dict:
    """Serialisable description of a `NetworkConfig`, for the checkpoint."""
    return {
        "hidden_sizes": list(cfg.hidden_sizes),
        "output_size": cfg.output_size,
        "weight_init_scale": cfg.weight_init_scale,
        "symmetric": cfg.symmetric,
        "skip": cfg.skip,
        "decoupled": cfg.decoupled,
    }


def config_from_spec(spec: Mapping[str, Any]) -> NetworkConfig:
    """Rebuild a `NetworkConfig` from a spec, falling back to the defaults."""
    base = NetworkConfig()
    return NetworkConfig(
        hidden_sizes=tuple(int(n) for n in spec.get("hidden_sizes", base.hidden_sizes)),
        output_size=int(spec.get("output_size", base.output_size)),
        weight_init_scale=float(spec.get("weight_init_scale", base.weight_init_scale)),
        symmetric=bool(spec.get("symmetric", base.symmetric)),
        skip=bool(spec.get("skip", base.skip)),
        decoupled=bool(spec.get("decoupled", base.decoupled)),
    )


class _Stack:
    """One feedforward chain, optionally with a direct input-to-output skip."""

    def __init__(
        self, sizes: Tuple[int, ...], skip: bool, scale: float, rng: np.random.Generator
    ) -> None:
        self.sizes = sizes
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        for n_in, n_out in zip(sizes[:-1], sizes[1:]):
            spread = scale / np.sqrt(n_in)
            self.weights.append(rng.normal(0.0, spread, size=(n_in, n_out)))
            self.biases.append(rng.normal(0.0, spread, size=n_out))
        self.skip = (
            rng.normal(0.0, scale / np.sqrt(sizes[0]), size=(sizes[0], sizes[-1]))
            if skip
            else None
        )

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        activation = inputs
        last = len(self.weights) - 1
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            total = activation @ weight + bias
            if i == last and self.skip is not None:
                total = total + inputs @ self.skip
            activation = np.tanh(total)
        return activation

    def parts(self) -> List[np.ndarray]:
        """Every parameter array, in the order the genome stores them."""
        parts = list(self.weights) + list(self.biases)
        return parts + ([self.skip] if self.skip is not None else [])

    def load(self, genome: np.ndarray, offset: int) -> int:
        for store in (self.weights, self.biases):
            for i, target in enumerate(store):
                store[i] = genome[offset : offset + target.size].reshape(target.shape)
                offset += target.size
        if self.skip is not None:
            self.skip = genome[offset : offset + self.skip.size].reshape(self.skip.shape)
            offset += self.skip.size
        return offset


@register_brain("paololimo")
class NeuralNetwork:
    """Dense feedforward network with tanh activations."""

    def __init__(
        self, input_size: int, spec: Mapping[str, Any], rng: np.random.Generator
    ) -> None:
        cfg = config_from_spec(spec)
        self.cfg = cfg
        self.symmetric = cfg.symmetric
        # Every input but the last is a sensor ray; the last one is own speed,
        # which is unchanged by mirroring the car left-to-right.
        self._mirror_span = input_size - 1
        # One stack per control when decoupled, otherwise a single shared one.
        widths = 1 if cfg.decoupled else cfg.output_size
        count = cfg.output_size if cfg.decoupled else 1
        self.stacks = [
            _Stack((input_size, *cfg.hidden_sizes, widths), cfg.skip, cfg.weight_init_scale, rng)
            for _ in range(count)
        ]

    # -- the incumbent's own view, kept for the panel and the tests ----------

    @property
    def layer_sizes(self) -> Tuple[int, ...]:
        return self.stacks[0].sizes

    @property
    def weights(self) -> List[np.ndarray]:
        return [w for stack in self.stacks for w in stack.weights]

    @property
    def biases(self) -> List[np.ndarray]:
        return [b for stack in self.stacks for b in stack.biases]

    def _pass(self, inputs: np.ndarray) -> np.ndarray:
        return np.concatenate([stack.forward(inputs) for stack in self.stacks])

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

    def describe(self) -> Topology:
        """Layers and weights for the panel: one band per stack, skips included."""
        depth = len(self.layer_sizes)
        layers: List[Tuple[str, int, int]] = [("sensors", self.layer_sizes[0], 0)]
        edges: List[Tuple[int, int, np.ndarray]] = []
        names = ("steer", "throttle") if len(self.stacks) > 1 else ("drive",)

        for s, stack in enumerate(self.stacks):
            previous = 0  # every stack reads the one sensor layer
            for column, size in enumerate(stack.sizes[1:], start=1):
                label = names[s] if column == depth - 1 else "hidden"
                layers.append((label, size, column))
                index = len(layers) - 1
                edges.append((previous, index, stack.weights[column - 1]))
                previous = index
            if stack.skip is not None:
                edges.append((0, previous, stack.skip))
        return Topology(layers=tuple(layers), edges=tuple(edges))

    # -- genome --------------------------------------------------------------

    @property
    def genome_size(self) -> int:
        return sum(part.size for stack in self.stacks for part in stack.parts())

    def get_genome(self) -> np.ndarray:
        """Flatten every parameter into a single 1-D vector."""
        return np.concatenate(
            [part.ravel() for stack in self.stacks for part in stack.parts()]
        )

    def set_genome(self, genome: np.ndarray) -> None:
        """Load a flat vector produced by `get_genome` back into the network."""
        if genome.size != self.genome_size:
            raise ValueError(f"Genome size {genome.size} does not match {self.genome_size}")
        offset = 0
        for stack in self.stacks:
            offset = stack.load(genome, offset)


__all__ = ["NetworkConfig", "NeuralNetwork", "config_from_spec", "spec_from_config"]
