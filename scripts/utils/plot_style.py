"""
Legacy TEP style shim.

The canonical TEP visual style now lives in :mod:`scripts.utils.plotting`.
``apply_tep_style`` delegates to :func:`setup_publication_style` and returns a
colour dictionary keyed by the legacy semantic names, so any existing caller
keeps working while inheriting the unified TEP-GNSS / TEP-GNSS-II look.
"""
import matplotlib as mpl

from scripts.utils.plotting import setup_publication_style, THEME_COLORS


def apply_tep_style():
    """
    Apply the canonical TEP manuscript visual style to matplotlib plots.

    Delegates to :func:`scripts.utils.plotting.setup_publication_style` and
    returns a dictionary of the legacy semantic colour names mapped onto the
    canonical TEP palette.
    """
    setup_publication_style()

    return {
        "dark":        THEME_COLORS["text"],       # deep near-black
        "purple":      THEME_COLORS["primary"],    # dark purple
        "accent":      THEME_COLORS["highlight"],  # orange accent
        "red":         "#b43b4e",                  # muted red
        "blue":        THEME_COLORS["accent"],     # medium blue
        "light_blue":  THEME_COLORS["secondary"],  # slate blue-grey
        "green":       THEME_COLORS["secondary"],  # slate blue-grey
        "bg":          THEME_COLORS["background"], # off-white
    }
