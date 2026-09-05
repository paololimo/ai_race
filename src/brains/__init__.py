"""Registry of drivable brains: the one part of the project a competitor writes.

Everything else — circuits, physics, sensors, fitness, the genetic algorithm,
the seed — is fixed by the harness, so a race between two brains compares the
architectures and nothing else.

A brain is any class satisfying `Brain`, registered under a name. Modules in
this package are imported on first use, so dropping a file in here is enough to
make its brain selectable from `train.py --brain <name>`.
"""

import importlib
import logging
import pkgutil
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Protocol, Type, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)


@runtime_checkable
class Brain(Protocol):
    """What a car needs from whatever is driving it.

    The constructor must be `(input_size: int, spec: Mapping[str, Any],
    rng: np.random.Generator)`. `spec` carries the brain's own hyperparameters
    and must be JSON-serialisable: it is written into the checkpoint, so a saved
    genome can be rebuilt without the code that trained it being on hand.
    """

    genome_size: int

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Map sensor readings to `[steering, throttle]`, both in [-1, 1]."""
        ...

    def get_genome(self) -> np.ndarray:
        """Flatten every learnable parameter into one 1-D float vector."""
        ...

    def set_genome(self, genome: np.ndarray) -> None:
        """Load a vector produced by `get_genome` back into the brain."""
        ...


@dataclass(frozen=True)
class BrainRef:
    """A brain identified by registry name plus its hyperparameters."""

    name: str = "baseline"
    spec: Mapping[str, Any] = field(default_factory=dict)


BRAIN_FACTORY: Dict[str, Type] = {}
_DISCOVERED = False


def register_brain(name: str):
    """Class decorator adding a brain to the registry."""

    def decorator(cls: Type) -> Type:
        if name in BRAIN_FACTORY and BRAIN_FACTORY[name] is not cls:
            raise ValueError(f"Brain '{name}' is already registered to {BRAIN_FACTORY[name]}")
        BRAIN_FACTORY[name] = cls
        return cls

    return decorator


def _discover() -> None:
    """Import every module in this package so its brains register themselves."""
    global _DISCOVERED
    if _DISCOVERED:
        return
    _DISCOVERED = True
    for module in pkgutil.iter_modules(__path__):
        if module.name.startswith("_"):
            continue
        try:
            importlib.import_module(f"{__name__}.{module.name}")
        except ImportError as exc:
            # One competitor's broken module must not sink the whole race.
            logger.error("Could not import brain module '%s': %s", module.name, exc)


def available_brains() -> list:
    _discover()
    return sorted(BRAIN_FACTORY)


def BrainFactory(name: str) -> Type:
    """Look up a registered brain class by name."""
    _discover()
    try:
        return BRAIN_FACTORY[name]
    except KeyError:
        raise SystemExit(
            f"Unknown brain '{name}'. Registered: {', '.join(available_brains()) or 'none'}"
        ) from None


def build_brain(ref: BrainRef, input_size: int, rng: np.random.Generator) -> Brain:
    """Instantiate the brain a reference names."""
    return BrainFactory(ref.name)(input_size, dict(ref.spec), rng)


__all__ = [
    "Brain",
    "BrainRef",
    "BrainFactory",
    "build_brain",
    "register_brain",
    "available_brains",
    "BRAIN_FACTORY",
]
