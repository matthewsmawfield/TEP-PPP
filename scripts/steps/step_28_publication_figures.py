#!/usr/bin/env python3
"""
STEP 28 -- Publication figures for the authoritative operational results.

Reads the JSON outputs of steps 25 / 26 / 27 (never recomputes) and renders
the paper's operational centerpiece:

  step_28_operational_case.png   2x2 panel:
    (a) held-out clock prediction error: no-network / bulk-mean / kernel,
        broadcast SPP vs precise products side by side
    (b) selected correlation length lambda* distribution vs the canonical
        4,200 km of Papers 1-2
    (c) position sigma vs visible-satellite count, three priors
    (d) constellation independence: short-baseline correlation across the
        four processing modes (D1)

Every number is drawn from the committed JSONs, so figure and text can never
disagree.
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.utils.plotting import setup_publication_style, save_publication_figure, get_color_palette  # noqa: E402

OUT = PROJECT_ROOT / "results" / "outputs"
FIG = PROJECT_ROOT / "results" / "figures"


def load(name):
    return json.loads((OUT / f"{name}.json").read_text())


def main():
    setup_publication_style()
    colors = get_color_palette()
    c_none, c_mean, c_kern = "#8a93a3", colors[1], colors[0]

    k = load("step_25_kernel_selection")
    g = load("step_26_operational_gain")
    d = load("step_27_discriminators")

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.6))
    (ax_a, ax_b), (ax_c, ax_d) = axes

    # ---------------------------------------------------------------- (a)
    b25, p25 = k["broadcast_spp"], k["precise_clk"]
    groups = [("Broadcast SPP\n(real-time regime)", b25["heldout_error_m"]),
              ("Precise products\n(post-processed)", p25["heldout_error_m"])]
    width = 0.26
    xs = np.arange(len(groups))
    for off, key, lab, col in ((-width, "raw", "No network", c_none),
                               (0.0, "net_mean", "Bulk network mean", c_mean),
                               (width, "kernel", "Distance kernel", c_kern)):
        vals = [gv[key] for _, gv in groups]
        bars = ax_a.bar(xs + off, vals, width * 0.92, label=lab, color=col)
        for r, v in zip(bars, vals):
            ax_a.text(r.get_x() + r.get_width() / 2, v * 1.03, f"{v:.2f}",
                      ha="center", va="bottom", fontsize=8.5)
    ax_a.set_yscale("log")
    ax_a.set_xticks(xs, [g0 for g0, _ in groups])
    ax_a.set_ylabel("Held-out clock prediction error (m)")
    ax_a.set_title("(a) Nested-CV, held-out stations: kernel skill "
                   f"{b25['skill_vs_raw_pct']['median']:+.0f}% (broadcast), "
                   f"{p25['skill_vs_raw_pct']['median']:+.1f}% (precise)", fontsize=10.5)
    ax_a.legend(fontsize=8.5, loc="upper right")

    # ---------------------------------------------------------------- (b)
    counts = b25["selected_lambda_km"]["counts"]
    lams = sorted(int(x) for x in counts)
    n = np.array([counts[str(x)] for x in lams], float)
    pos = np.arange(len(lams))
    ax_b.bar(pos, n / n.sum() * 100, 0.7, color=c_kern, alpha=0.85)
    ax_b.set_xticks(pos, [f"{x:,}" for x in lams], rotation=45, fontsize=8.5)
    if 4200 in lams:
        i = lams.index(4200)
        ax_b.axvline(i, color=colors[4], lw=2, ls="--")
        ax_b.text(i + 0.12, ax_b.get_ylim()[1] * 0.92,
                  "canonical $\\lambda_T$ = 4,200 km\n(Papers 1–2)",
                  color=colors[4], fontsize=9)
    ax_b.set_xlabel("Selected correlation length $\\lambda^{*}$ (km)")
    ax_b.set_ylabel("Fraction of held-out splits (%)")
    med = b25["selected_lambda_km"]["median"]; iqr = b25["selected_lambda_km"]["iqr"]
    ax_b.set_title(f"(b) $\\lambda^{{*}}$ chosen blind on held-out data: "
                   f"median {med:,.0f} km, IQR [{iqr[0]:,.0f}, {iqr[1]:,.0f}]", fontsize=10.5)

    # ---------------------------------------------------------------- (c)
    bysat = g["broadcast_spp"]["by_satellite_count"]
    ks = ["3", "4", "5", "6", "8", "99"]
    xlab = ["3", "4", "5", "6", "8", "full\nsky"]
    xs = np.arange(len(ks))
    for key, lab, col, mk in (("self", "No network (self prior)", c_none, "o"),
                              ("net_mean", "Bulk network mean", c_mean, "s"),
                              ("kernel", "Distance kernel", c_kern, "D")):
        ax_c.plot(xs, [bysat[kk]["pos_sigma_m"][key] for kk in ks],
                  marker=mk, color=col, lw=2, ms=5.5, label=lab)
    for i, kk in enumerate(("4", "5")):
        j = xs[ks.index(kk)]
        ax_c.annotate(f"−{bysat[kk]['kernel_vs_self_pct']:.0f}%",
                      (j, bysat[kk]["pos_sigma_m"]["kernel"]),
                      textcoords="offset points", xytext=(4, -14),
                      color=c_kern, fontsize=9, fontweight="bold")
    ax_c.set_yscale("log")
    ax_c.set_xticks(xs, xlab)
    ax_c.set_xlabel("Visible satellites")
    ax_c.set_ylabel("3D position $\\sigma$ (m)")
    ax_c.set_title("(c) Position gain concentrates where positioning is hard", fontsize=10.5)
    ax_c.legend(fontsize=8.5)

    # ---------------------------------------------------------------- (d)
    pm = d["D1_D2_constellation_and_distance"]["per_mode"]
    modes = [("gps_broadcast", "GPS\nbroadcast"), ("ionofree", "Iono-free\ndual-freq"),
             ("multignss_mgex", "Multi-GNSS\n(G+R+E+C)"), ("precise", "IGS precise\nsat clocks")]
    vals = [pm[m]["corr_lt400km"] for m, _ in modes]
    xs = np.arange(len(modes))
    ax_d.bar(xs, vals, 0.62, color=[c_kern, colors[2], colors[3], colors[1]], alpha=0.9)
    for x, v in zip(xs, vals):
        ax_d.text(x, v + 0.012, f"{v:+.3f}", ha="center", fontsize=9)
    mean_v = float(np.mean(vals))
    ax_d.axhline(mean_v, color="#555", lw=1, ls=":")
    ax_d.set_xticks(xs, [lab for _, lab in modes], fontsize=8.5)
    ax_d.set_ylabel("Pair correlation, baselines < 400 km")
    ax_d.set_ylim(0, max(vals) * 1.22)
    spread = d["D1_D2_constellation_and_distance"]["cross_mode_spread_frac"] * 100
    gps_spread = d["D1_D2_constellation_and_distance"].get(
        "satellite_product_axis_spread_frac", 0) * 100
    ax_d.set_title(f"(d) Constellation independence: GPS-only modes agree to {gps_spread:.0f}%\n"
                   f"(four-mode spread {spread:.0f}% driven by multi-GNSS offset; "
                   "correlation is receiver-side)", fontsize=10.5)

    fig.suptitle("TEP-PPP operational case: held-out skill, blind scale recovery, "
                 "degraded-geometry gain, constellation independence",
                 fontsize=12.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_publication_figure(fig, "step_28_operational_case.png", FIG)
    print(f"saved {FIG / 'step_28_operational_case.png'}")


if __name__ == "__main__":
    main()
