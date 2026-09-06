"""The palette both surfaces are drawn in.

The panel and the strip are two views of one run and have to look like it, so
the colours live in neither of them. They were in `dashboard.py`, which meant
the strip could not be imported without the panel and a palette change made
"in the dashboard" restyled a module its author was not editing.
"""

from src.brains import Color

BG = (18, 20, 32)
CARD = (28, 31, 48)
TEXT = (232, 234, 245)
MUTED = (140, 146, 170)
ACCENT = (255, 176, 60)
POSITIVE = (110, 220, 150)
NEGATIVE = (235, 100, 110)


def tint(color: Color, strength: float) -> Color:
    """`color` faded towards the card background by `strength` in [0, 1]."""
    return tuple(int(CARD[k] + (color[k] - CARD[k]) * strength) for k in range(3))


__all__ = ["ACCENT", "BG", "CARD", "MUTED", "NEGATIVE", "POSITIVE", "TEXT", "tint"]
