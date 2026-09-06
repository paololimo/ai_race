"""Turning a finished run into figures and tables a report can use directly.

A screenshot of the training window is a picture of a chart at whatever
resolution the screen happened to be, with the dashboard's dark palette and its
labels sized for a monitor. This package re-draws the same analyses from the
data the run left behind — as vector PDFs that stay sharp at any size, and as
tables whose numbers can be quoted rather than read off an axis.

    python experiments/make_report.py

`collect` reads what is on disk, `figures` draws, `tables` writes the numbers
out, and `style` decides how all of it looks. Nothing draws while reading and
nothing reads while drawing.
"""

from experiments.report import collect, figures, style, tables

__all__ = ["collect", "figures", "style", "tables"]
