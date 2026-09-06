"""How every figure in the report is drawn, decided once.

The dashboard palette is built for a dark window and cannot be reused on paper:
codex is off-white, which on a white page is nothing at all. Each entrant keeps
its hue — so a reader moving between the video and the report recognises the
same car — and only its lightness is moved into a range that prints.

Everything here is set for vector output. Fonts are embedded as TrueType rather
than as Type 3, because Type 3 is what makes a PDF figure refuse to have its
text selected, searched, or re-rendered at print resolution; nothing is
rasterised, so the figures survive any zoom a reviewer applies.
"""

import colorsys
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")  # figures are written to files; no window is ever opened.

import matplotlib.pyplot as plt

from src.brains import COLORS, Color

# Ink, not decoration: one near-black for text, two greys for structure.
INK = "#16181f"
GRID = "#d9dce4"
FAINT = "#8b90a0"

# Lightness band that reads on white paper. Below it the entrants stop being
# distinguishable from the axis text; above it they vanish into the page.
_LIGHTNESS = (0.34, 0.52)


def _print_safe(color: Color) -> str:
    """`color` moved to a lightness that prints, with its hue untouched."""
    r, g, b = (channel / 255 for channel in color)
    hue, lightness, saturation = colorsys.rgb_to_hls(r, g, b)
    low, high = _LIGHTNESS
    lightness = min(high, max(low, lightness))
    # Near-greys (codex) would come back as a muddy tint; keep them neutral.
    saturation = max(saturation, 0.0) if saturation > 0.15 else 0.0
    return matplotlib.colors.to_hex(colorsys.hls_to_rgb(hue, lightness, saturation))


PRINT_COLORS: Dict[str, str] = {name: _print_safe(color) for name, color in COLORS.items()}

_FALLBACK: Tuple[str, ...] = ("#2f7d4f", "#a03080", "#8a7010", "#1f7f86", "#5a3fa8")


def color_for(name: str, ordinal: int = 0) -> str:
    """The entrant's colour on paper; an unlisted entrant gets a stable one."""
    return PRINT_COLORS.get(name, _FALLBACK[ordinal % len(_FALLBACK)])


def use_report_style() -> None:
    """Apply the report's look to every figure made afterwards."""
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,  # TrueType: selectable, searchable, re-renderable
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.dpi": 150,  # only affects the optional PNG copies
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "savefig.transparent": False,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 8,
            "axes.labelsize": 8.5,
            "axes.edgecolor": FAINT,
            "axes.linewidth": 0.6,
            "axes.labelcolor": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "text.color": INK,
            "xtick.color": FAINT,
            "ytick.color": FAINT,
            "xtick.labelcolor": INK,
            "ytick.labelcolor": INK,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "legend.handlelength": 1.6,
            "lines.solid_capstyle": "round",
        }
    )


__all__ = ["FAINT", "GRID", "INK", "PRINT_COLORS", "color_for", "use_report_style"]
