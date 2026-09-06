"""The numbers as text, so no figure has to be read off with a ruler.

A chart shows a difference; a table is what someone quotes. Both formats come
out of one row builder, because a Markdown table and a LaTeX one that disagree
about a number are worse than either alone.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np

from experiments.report.collect import AblationGroup, Entrant, Standing

Row = Sequence[str]


def _markdown(header: Row, rows: Sequence[Row]) -> str:
    widths = [max(len(str(cell)) for cell in column) for column in zip(header, *rows, strict=True)]
    def line(cells: Row) -> str:
        return "| " + " | ".join(str(c).ljust(w) for c, w in zip(cells, widths, strict=True)) + " |"
    rule = "|-" + "-|-".join("-" * w for w in widths) + "-|"
    return "\n".join([line(header), rule, *map(line, rows)])


# The characters that come out of a log and mean something else in LaTeX. The
# em dash is here because pdflatex without unicode support does not fail on it
# loudly — it drops it, and the winner's row silently loses a word.
_ESCAPES = {
    "&": r"\&",
    "%": r"\%",
    "#": r"\#",
    "_": r"\_",
    "\u2014": "---",
    "\u2013": "--",
}


def _tex(cell: object) -> str:
    text = str(cell)
    for character, replacement in _ESCAPES.items():
        text = text.replace(character, replacement)
    return text


def _latex(header: Row, rows: Sequence[Row], caption: str) -> str:
    """A booktabs table: the one form that pastes into a paper unedited."""
    columns = "l" + "r" * (len(header) - 1)
    body = "\n".join(" & ".join(_tex(c) for c in row) + r" \\" for row in rows)
    return "\n".join(
        [
            r"\begin{table}[t]",
            r"  \centering",
            f"  \\caption{{{_tex(caption)}}}",
            f"  \\begin{{tabular}}{{{columns}}}",
            r"    \toprule",
            "    " + " & ".join(_tex(c) for c in header) + r" \\",
            r"    \midrule",
            *("    " + line for line in body.splitlines()),
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
        ]
    )


def standings_table(
    entrants: Sequence[Entrant], standings: Sequence[Standing]
) -> Dict[str, str]:
    """Who raced, what they cost, and where they finished — in one table.

    The architecture, the training score and the race result belong together:
    the entrant that trains best is not necessarily the one that finishes, and
    that gap is the project's central result. Split across two tables it takes
    a reader to notice it.
    """
    placed: Dict[str, Standing] = {s.name: s for s in standings}
    header = ("entrant", "parameters", "best training fitness", "laps raced", "result")
    ordered = sorted(entrants, key=lambda e: placed[e.name].rank if e.name in placed else 99)
    rows: List[Row] = []
    for entrant in ordered:
        standing = placed.get(entrant.name)
        rows.append(
            (
                entrant.name,
                str(entrant.parameters),
                f"{entrant.best_fitness:.2f}",
                f"{standing.laps:.2f}" if standing else "—",
                standing.note if standing else "did not race",
            )
        )
    caption = (
        "Training and race results. Fitness is the worst circuit plus 0.35 times the mean "
        "across the three training circuits; the race is five laps of a circuit no entrant "
        "trained on."
    )
    return {"markdown": _markdown(header, rows), "latex": _latex(header, rows, caption)}


def ablation_table(group: AblationGroup) -> Dict[str, str]:
    """One question's variants, with the spread kept next to the median.

    The range is not decoration. Two variants whose ranges overlap have not been
    told apart by this experiment, however far apart their medians sit, and a
    table that reports only the median hides exactly that.
    """
    header = ("variant", "parameters", "training (med)", "race (med)", "race range", "seeds")
    rows: List[Row] = [
        (
            row.variant,
            str(row.parameters),
            f"{row.training_median:.2f}",
            f"{row.race_median:.2f}",
            f"{min(row.race):.2f} to {max(row.race):.2f}",
            str(len(row.race)),
        )
        for row in group.rows
    ]
    caption = (
        f"{group.title}: median over seeds, with the full range. Where two ranges overlap, "
        f"the difference between those variants is not resolved by this experiment."
    )
    return {"markdown": _markdown(header, rows), "latex": _latex(header, rows, caption)}


def run_summary(entrants: Sequence[Entrant], seconds: Optional[np.ndarray]) -> str:
    """The facts a methods section has to state, in the order it states them."""
    if not entrants:
        return ""
    first = entrants[0]
    lines = [
        f"- entrants: {len(entrants)} ({', '.join(sorted(e.name for e in entrants))})",
        f"- generations: {first.generations}",
        f"- population per entrant: {first.population}",
        "- circuits per generation: 3 (each entrant scored under the same draw)",
        f"- evaluations per entrant: {first.evaluations:,}",
        f"- seed: {first.seed}",
    ]
    if seconds is not None and len(seconds):
        lines.append(
            f"- wall clock: {seconds.sum() / 3600:.1f} h, "
            f"median {np.median(seconds):.0f} s per generation"
        )
    return "\n".join(lines)


__all__ = ["ablation_table", "run_summary", "standings_table"]
