import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates  # noqa: F401  (re-exported for legacy callers)
from pathlib import Path

# ---------------------------------------------------------------------------
# Canonical TEP visual style.
#
# This is the single source of truth for every figure in the TEP-PPP pipeline.
# It reproduces the dominant shared style used across the companion papers
# TEP-GNSS (Paper 1, Jaipur) and TEP-GNSS-II (Paper 2, Cairo): a serif
# (Times New Roman) typeface, modest point sizes, thin axes, a dashed
# slate-blue grid, and the TEP manuscript colour palette.  Every step that
# renders a figure must call ``setup_publication_style()`` before plotting.
# ---------------------------------------------------------------------------

# Standard dimensions for consistency
STD_WIDTH = 12.0  # inches
STD_HEIGHT = 7.0  # inches (approx 16:9)
EXPORT_DPI = 300

# TEP manuscript palette -- mirrors TEPConfig.theme_colors in TEP-GNSS and the
# inline rcParams blocks used throughout TEP-GNSS / TEP-GNSS-II.
THEME_COLORS = {
    "primary":    "#2D0140",  # Dark purple (headers, primary series)
    "secondary":  "#495773",  # Slate blue-grey (secondary series, grid)
    "accent":     "#4A90C2",  # Medium blue (accent series)
    "highlight":  "#F39C12",  # Orange (highlight / contrast series)
    "text":       "#220126",  # Near-black text
    "background": "#F8F8FF",  # Off-white background
    "grid":       "#495773",  # Grid lines (matches the slate-blue grid)
    "border":     "#999999",  # Medium grey for frames / legend edges
}

# Multi-series colour cycle, ordered to match the cycle used in the
# TEP-GNSS visualisation steps (primary, secondary, highlight, accent, ...).
_COLOR_CYCLE = [
    THEME_COLORS["primary"],    # 0: dark purple
    THEME_COLORS["secondary"],  # 1: slate blue-grey
    THEME_COLORS["highlight"],  # 2: orange
    THEME_COLORS["accent"],     # 3: medium blue
    "#b43b4e",                  # 4: muted red (contrast / danger)
]


def setup_publication_style():
    """
    Configure Matplotlib for consistent, publication-quality TEP figures.

    Matches the dominant shared style of the TEP-GNSS and TEP-GNSS-II papers:
    serif (Times New Roman) typeface, base 11 pt, axis titles 14 pt, axis
    labels 12 pt, tick labels 10 pt, legend 9 pt, 1.5 pt lines, 1.0 pt axes,
    a dashed slate-blue (#495773) grid at 0.5 pt, near-black (#220126) text,
    and dark-purple (#2D0140) titles.
    """
    mpl.rcParams.update({
        # Fonts
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "mathtext.fontset": "dejavuserif",  # keep maths in the serif family
        "text.usetex": False,
        "font.size": 11,

        # Figure layout
        "figure.figsize": (STD_WIDTH, STD_HEIGHT),
        "figure.dpi": EXPORT_DPI,
        "figure.autolayout": False,
        "figure.facecolor": "white",

        "savefig.dpi": EXPORT_DPI,
        "savefig.bbox": "standard",
        "savefig.pad_inches": 0.0,

        # Axes
        "axes.facecolor": "white",
        "axes.edgecolor": THEME_COLORS["text"],
        "axes.linewidth": 1.0,
        "axes.grid": True,
        "axes.titlesize": 14,
        "axes.titleweight": "normal",
        "axes.titlecolor": THEME_COLORS["primary"],
        "axes.labelsize": 12,
        "axes.labelweight": "normal",
        "axes.labelcolor": THEME_COLORS["text"],
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.prop_cycle": mpl.cycler(color=_COLOR_CYCLE),

        # Grid
        "grid.color": THEME_COLORS["grid"],
        "grid.alpha": 0.5,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,

        # Ticks
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.color": THEME_COLORS["text"],
        "ytick.color": THEME_COLORS["text"],
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,

        # Legend
        "legend.fontsize": 9,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.fancybox": False,
        "legend.edgecolor": THEME_COLORS["border"],
        "legend.facecolor": "white",
        "legend.loc": "best",

        # Lines & markers
        "lines.linewidth": 1.5,
        "lines.markersize": 5,
        "lines.markeredgewidth": 0.8,

        # Text
        "text.color": THEME_COLORS["text"],

        # Patch
        "patch.edgecolor": THEME_COLORS["text"],
        "patch.force_edgecolor": True,
    })


def get_theme_colors():
    """Return a copy of the canonical TEP theme colour dictionary."""
    return dict(THEME_COLORS)


def get_figure_size(width=None, height=None, aspect_ratio=None):
    """
    Calculate figure dimensions, defaulting to the standard TEP canvas.

    Args:
        width: Override standard width (inches).
        height: Explicit height (inches).
        aspect_ratio: Width/Height ratio (e.g. 16/9). Used if height is None.

    Returns:
        (width, height) tuple.
    """
    w = width if width else STD_WIDTH

    if height:
        h = height
    elif aspect_ratio:
        h = w / aspect_ratio
    else:
        h = STD_HEIGHT

    return (w, h)


def save_publication_figure(fig, filename, output_dir=None):
    """
    Save a figure with the canonical TEP publication settings.

    Applies ``tight_layout`` (when possible) and writes a 300 dpi PNG on a
    white canvas, matching the save convention used in TEP-GNSS / TEP-GNSS-II.

    Args:
        fig: The matplotlib Figure object.
        filename: Filename (e.g. 'figure_1.png').
        output_dir: Output directory (optional).
    """
    if output_dir:
        out_path = Path(output_dir) / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_path = Path(filename)

    # Apply tight layout only when the figure does not already have a layout
    # engine (e.g. constrained_layout) configured, to avoid matplotlib's
    # "figure layout has changed to tight" warning.
    try:
        if fig.get_layout_engine() is None:
            fig.tight_layout()
    except Exception:
        pass

    fig.savefig(out_path, dpi=EXPORT_DPI, facecolor="white")
    print(f"Saved publication-quality figure to: {out_path}")


def get_color_palette(n_colors=5):
    """
    Return the canonical TEP colour cycle.

    Ordered to match the TEP-GNSS visualisation steps:
    0: dark purple (primary), 1: slate blue-grey (secondary),
    2: orange (highlight), 3: medium blue (accent), 4: muted red (contrast).
    """
    if n_colors <= len(_COLOR_CYCLE):
        return _COLOR_CYCLE[:n_colors]
    # Cycle if more colours are requested than the palette holds.
    return _COLOR_CYCLE * (n_colors // len(_COLOR_CYCLE) + 1)
