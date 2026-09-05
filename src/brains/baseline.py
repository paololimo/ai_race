"""The project's own brain, exposed through the registry as `baseline`.

A thin adapter: `NeuralNetwork` already satisfies the `Brain` protocol, it just
takes a `NetworkConfig` rather than a plain spec dict. Translating here keeps
`neural_network.py` free of any knowledge of the race harness.
"""

from typing import Any, Mapping

import numpy as np

from src.brains import register_brain
from src.config import NetworkConfig
from src.neural_network import NeuralNetwork


def spec_from_config(cfg: NetworkConfig) -> dict:
    """Serialisable description of a `NetworkConfig`, for the checkpoint."""
    return {
        "hidden_sizes": list(cfg.hidden_sizes),
        "output_size": cfg.output_size,
        "weight_init_scale": cfg.weight_init_scale,
        "symmetric": cfg.symmetric,
    }


def config_from_spec(spec: Mapping[str, Any]) -> NetworkConfig:
    """Rebuild a `NetworkConfig` from a spec, falling back to the defaults."""
    base = NetworkConfig()
    return NetworkConfig(
        hidden_sizes=tuple(int(n) for n in spec.get("hidden_sizes", base.hidden_sizes)),
        output_size=int(spec.get("output_size", base.output_size)),
        weight_init_scale=float(spec.get("weight_init_scale", base.weight_init_scale)),
        symmetric=bool(spec.get("symmetric", base.symmetric)),
    )


@register_brain("baseline")
class BaselineBrain(NeuralNetwork):
    """Dense `8 -> 14 -> 14 -> 2` with tanh, optionally left/right symmetric."""

    def __init__(
        self, input_size: int, spec: Mapping[str, Any], rng: np.random.Generator
    ) -> None:
        super().__init__(input_size, config_from_spec(spec), rng)


__all__ = ["BaselineBrain", "config_from_spec", "spec_from_config"]
