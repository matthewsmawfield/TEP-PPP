#!/usr/bin/env python3
"""Render the Step 80 clock-field localisation result."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.utils.plotting import setup_publication_style, get_color_palette  # noqa: E402

SOURCE = ROOT / "results" / "outputs" / "step_80_clock_field_localisation.json"
OUT = ROOT / "results" / "figures" / "step_80_clock_field_localisation.png"


def wrapped_segment(lon1, lat1, lon2, lat2):
    if abs(lon2 - lon1) <= 180:
        return [([lon1, lon2], [lat1, lat2])]
    if lon1 < lon2:
        lon1 += 360
    else:
        lon2 += 360
    cross_lat = lat1 + (lat2 - lat1) * ((180 - min(lon1, lon2)) / abs(lon2 - lon1))
    return [([((lon1 + 180) % 360) - 180, 180], [lat1, cross_lat]),
            ([-180, ((lon2 + 180) % 360) - 180], [cross_lat, lat2])]


def main():
    data = json.loads(SOURCE.read_text())
    sites = data["sites"]
    names = ["MAT1-MATE", "THTI-THTG", "STR2-STR1"]
    labels = {"MAT1-MATE": "Matera", "THTI-THTG": "Tahiti", "STR2-STR1": "Stromlo"}
    _pal = get_color_palette()
    colors = {"MAT1-MATE": _pal[3], "THTI-THTG": _pal[2], "STR2-STR1": _pal[4]}

    setup_publication_style()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.4, 5.1),
                                 gridspec_kw={"width_ratios": [1.55, 1.0]})
    ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
    ax.set_xticks(np.arange(-180, 181, 60)); ax.set_yticks(np.arange(-60, 61, 30))
    ax.grid(color="#d9dee3", lw=0.7); ax.set_facecolor("#f7f9fb")
    ax.set_xlabel("Longitude (deg)"); ax.set_ylabel("Latitude (deg)")
    ax.set_title("A  Target-blind clock-field localisation")

    for name in names:
        scalar = sites[name]["fits"]["exponential"]
        multi = sites[name]["multimode_fits"]["exponential"]["all_modes"]
        c = colors[name]
        ax.scatter(scalar["true_lon_deg"], scalar["true_lat_deg"], marker="*", s=150,
                   color=c, edgecolor="black", linewidth=0.5, zorder=4)
        ax.scatter(scalar["estimated_lon_deg"], scalar["estimated_lat_deg"], marker="x",
                   s=75, color=c, linewidth=2, alpha=0.55, zorder=3)
        ax.scatter(multi["estimated_lon_deg"], multi["estimated_lat_deg"], marker="o",
                   s=70, facecolor="white", edgecolor=c, linewidth=2.2, zorder=5)
        for xs, ys in wrapped_segment(scalar["true_lon_deg"], scalar["true_lat_deg"],
                                      multi["estimated_lon_deg"], multi["estimated_lat_deg"]):
            ax.plot(xs, ys, color=c, lw=1.5, alpha=0.8)
        ax.annotate(labels[name], (scalar["true_lon_deg"], scalar["true_lat_deg"]),
                    xytext=(5, 7), textcoords="offset points", color=c, weight="bold")
    ax.scatter([], [], marker="*", s=120, color="0.4", label="true anchor")
    ax.scatter([], [], marker="x", s=60, color="0.4", label="scalar estimate")
    ax.scatter([], [], marker="o", s=60, facecolor="white", edgecolor="0.4",
               label="43-mode estimate")
    ax.legend(loc="lower left", frameon=True, ncol=3, fontsize=8)

    x = np.arange(len(names)); w = 0.34
    actual = [sites[n]["multimode_fits"]["exponential"]["all_modes"]["surface_error_km"]
              for n in names]
    null = [sites[n]["multimode_fits"]["exponential"]["all_modes"]["geometry_null"]
            ["median_localisation_error_km"] for n in names]
    bx.bar(x - w/2, actual, w, color=[colors[n] for n in names], label="43-mode error")
    bx.bar(x + w/2, null, w, color="#c8cdd2", label="geometry-null median")
    bx.set_yscale("log"); bx.set_ylim(200, 22000)
    bx.set_xticks(x, [labels[n] for n in names]); bx.set_ylabel("Surface error (km, log scale)")
    bx.set_title("B  Spatial prediction versus geometry null")
    bx.grid(axis="y", which="both", color="#d9dee3", lw=0.7); bx.set_axisbelow(True)
    for i, value in enumerate(actual):
        bx.text(i - w/2, value * 1.08, f"{value:,.0f}", ha="center", va="bottom",
                fontsize=9, weight="bold")
    bx.legend(loc="upper left", frameon=False)
    bx.text(0.02, 0.03, "Matera p=0.004   Tahiti p=0.010\nStromlo retained as adverse anchor",
            transform=bx.transAxes, fontsize=9)

    fig.suptitle("Phase modes turn the TEP distance structure into position information",
                 fontsize=14, weight="bold", y=1.01)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
