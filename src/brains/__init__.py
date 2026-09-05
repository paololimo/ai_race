"""The competitors. One file per entrant, and nothing else in this package.

Everything a comparison rests on — circuits, physics, sensors, fitness, the
genetic algorithm, the seed — is fixed by the harness, so what separates the
entrants is the architecture and the weights evolution finds for it.

An entrant is any class satisfying `Brain`, registered under a name. Every
module here is imported on startup, so **dropping a file in this directory is
the whole installation**: the next `python train.py` trains it alongside the
others and the next `python race.py` puts it on the grid, with no other change
anywhere.
"""

import importlib
import logging
import pkgutil
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Protocol, Tuple, Type, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)

Color = Tuple[int, int, int]

# Colours are set here, not by the entrants themselves, so that no competitor
# can pick one that clashes with another's. A name not listed gets the next
# unused colour from the palette below, which is why a new entrant needs no
# edit here either — this map only exists to honour specific requests.
COLORS: Dict[str, Color] = {
    "paololimo": (214, 62, 62),    # red
    "gemini": (70, 120, 235),      # blue
    "codex": (226, 228, 234),      # off-white
    "claude": (240, 150, 60),      # orange
}

PALETTE: Tuple[Color, ...] = (
    (140, 230, 130),  # green
    (232, 110, 190),  # magenta
    (250, 240, 120),  # yellow
    (120, 220, 220),  # teal
    (180, 140, 240),  # violet
)


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
    """An entrant: registry name plus the hyperparameters it was built with."""

    name: str
    spec: Mapping[str, Any] = field(default_factory=dict)


BRAIN_FACTORY: Dict[str, Type] = {}
_DISCOVERED = False


def register_brain(name: str):
    """Class decorator adding an entrant to the registry.

    One class is one entrant. Registering the same class under a second name
    would put it on the grid twice, under two colours, competing with itself —
    so the extra name is ignored rather than honoured.
    """

    def decorator(cls: Type) -> Type:
        if name in BRAIN_FACTORY and BRAIN_FACTORY[name] is not cls:
            raise ValueError(f"'{name}' is already registered to {BRAIN_FACTORY[name]}")
        already = next((n for n, c in BRAIN_FACTORY.items() if c is cls), None)
        if already is not None:
            logger.warning(
                "%s is already entered as '%s'; ignoring the second name '%s'",
                cls.__name__,
                already,
                name,
            )
            return cls
        BRAIN_FACTORY[name] = cls
        return cls

    return decorator


def _discover() -> None:
    """Import every module here so its entrant registers itself."""
    global _DISCOVERED
    if _DISCOVERED:
        return
    _DISCOVERED = True
    for module in pkgutil.iter_modules(__path__):
        if module.name.startswith("_"):
            continue
        try:
            importlib.import_module(f"{__name__}.{module.name}")
        except Exception as exc:  # noqa: BLE001 - see below
            # A competitor's file is code someone else wrote. If it raises on
            # import — a syntax error, a missing name, a bad expression at class
            # level — the race carries on without it rather than refusing to
            # start, and the log says who dropped out.
            logger.error("Entrant '%s' failed to load and is out: %s", module.name, exc)


def entrants() -> List[str]:
    """Every registered entrant, in a stable order."""
    _discover()
    return sorted(BRAIN_FACTORY)


def color_of(name: str) -> Color:
    """The colour this entrant's cars are drawn in, the same on every run."""
    if name in COLORS:
        return COLORS[name]
    unclaimed = [n for n in entrants() if n not in COLORS]
    return PALETTE[unclaimed.index(name) % len(PALETTE)] if name in unclaimed else PALETTE[0]


def BrainFactory(name: str) -> Type:
    """Look up a registered entrant by name."""
    _discover()
    try:
        return BRAIN_FACTORY[name]
    except KeyError:
        raise SystemExit(
            f"No entrant called '{name}'. Registered: {', '.join(entrants()) or 'none'}"
        ) from None


def build_brain(ref: BrainRef, input_size: int, rng: np.random.Generator) -> Brain:
    """Instantiate the entrant a reference names."""
    return BrainFactory(ref.name)(input_size, dict(ref.spec), rng)


__all__ = [
    "BRAIN_FACTORY",
    "Brain",
    "BrainFactory",
    "BrainRef",
    "Color",
    "build_brain",
    "color_of",
    "entrants",
    "register_brain",
]
