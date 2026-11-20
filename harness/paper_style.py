"""One paper style for every figure, matching the 12 Nov meeting figures.

    from harness import paper_style as ps
    ps.apply()
    fig, ax = plt.subplots(figsize=(ps.WIDTH_IN, 2.4))
    ...
    ps.save(fig, folder / "name")          # name.png (300 dpi) and name.pdf

- 7 in wide (a two-column figure*) or 3.4 in (one column), Liberation Serif to match a Times body,
  7 to 8 pt text at final size, white background.
- No headline inside the figure: the caption carries it. Panels are (a), (b), ...
- Colour has one meaning everywhere: blue = baseline / on time, orange =
  perturbed / late. Orange fills are hatched and orange lines are dashed, so
  every figure survives greyscale print. Grey is context (neighbours, other jobs).
- One y-axis per panel: two quantities get two panels, never a twin axis.
- Every figure that uses D or T says what they mean (footnote(fig, D_MEANS, T_MEANS)).
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID, SHADE = "#1a1a1a", "#5c5c5c", "#d9d9d9", "#f1f1f1"
CONTEXT = "#b3b3b3"  # neighbours and other jobs: present, but not the point
WIDTH_IN = 7.0  # two-column figure*
COLUMN_IN = 3.4  # one column
F_TITLE, F_TEXT, F_SMALL = 8.0, 7.0, 6.5
LW = 0.8
HATCH = "////"
DEADLINE_LS = (0, (2, 1.5))  # dotted black: a deadline
DASHES = [  # line patterns, in this order; then they repeat
    (0, (4, 2)),
    (0, (1.5, 1.5)),
    (0, (6, 2, 1.5, 2)),
    (0, (8, 2)),
    (0, (3, 1, 1, 1, 1, 1)),
]
TO, TS, TE = r"$t_{\mathrm{obs}}$", r"$t_{\mathrm{start}}$", r"$t_{\mathrm{end}}$"

# Say what the symbols mean in every figure that uses them.
D_MEANS = (
    r"$D$ = deadline: the most time a frame may take, from when it is observed ($t_{\mathrm{obs}}$)"
    r" to when its answer arrives ($t_{\mathrm{end}}$)"
)
T_MEANS = r"$T$ = frame period: the time between two frames"


def apply() -> None:
    plt.rcParams.update(
        {
            "font.family": "Liberation Serif",
            "mathtext.fontset": "stix",
            "font.size": F_TEXT,
            "axes.titlesize": F_TITLE,
            "axes.titlelocation": "left",
            "axes.titlepad": 4,
            "axes.labelsize": F_TEXT,
            "xtick.labelsize": F_TEXT,
            "ytick.labelsize": F_TEXT,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": GRID,
            "grid.linewidth": 0.4,
            "axes.axisbelow": True,
            "hatch.linewidth": 0.6,
            "legend.fontsize": F_TEXT,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_title(ax, letter: str, text: str) -> None:
    """'(a) Time-slicing', bold letter, left-aligned above the axes."""
    ax.set_title(f"({letter}) {text}", fontsize=F_TITLE, color=INK, loc="left", weight="bold")


def swatch(fc: str, hatch: str | None = None, ec: str = "white") -> Rectangle:
    """A legend handle for a filled bar, hatched if `hatch` (orange = late)."""
    return Rectangle((0, 0), 1, 1, fc=fc, ec=ec, lw=0 if ec == "white" else LW, hatch=hatch)


def visible_length(text: str) -> int:
    """Characters that print: mathtext markup like $t_{\\mathrm{obs}}$ counts as 'tobs'."""
    return len(re.sub(r"\\mathrm|[${}_\\]", "", text))


def wrap_visible(line: str, width: int) -> list[str]:
    """Greedy word wrap on printed width (never breaks inside a $...$ word)."""
    out, current = [], ""
    for word in line.split(" "):
        candidate = f"{current} {word}" if current else word
        if current and visible_length(candidate) > width:
            out.append(current)
            current = word
        else:
            current = candidate
    return out + ([current] if current else [])


def footnote(fig, *lines: str) -> int:
    """Small notes centred under everything else, e.g. what D and T mean.

    Each line is wrapped to the figure's width. Returns the number of printed
    lines, so the caller can leave room for them.
    """
    width_chars = int(fig.get_figwidth() * 17)  # about 17 characters of 6.5 pt serif per inch
    wrapped = [part for line in lines for part in wrap_visible(line, width_chars)]
    fig.text(
        0.5,
        0.012,
        "\n".join(wrapped),
        ha="center",
        va="bottom",
        fontsize=F_SMALL,
        color=MUTED,
        linespacing=1.3,
    )
    return len(wrapped)


def save(fig, path: Path) -> None:
    """Write path.png (300 dpi) and path.pdf (fonts embedded as TrueType)."""
    fig.savefig(path.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
