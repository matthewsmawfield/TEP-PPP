#!/usr/bin/env python3
"""
STEP 37c -- Live NTRIP Report and Comparison Figures
====================================================

Reads the step_37b live processing result and the step_33 archived
result, produces a comparison figure, and writes a summary JSON that
the manuscript Section 3.10 references.

Outputs
-------
- ``results/figures/step_37_live_comparison.png``: side-by-side
  comparison of live vs. archived position error under obstruction.
- ``results/outputs/step_37c_ntrip_report.json``: summary table with
  live and archived numbers, ready for manuscript inclusion.
"""
from __future__ import annotations

import sys
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.utils.logger import TEPLogger, set_step_logger, print_status  # noqa: E402
from scripts.utils.plotting import setup_publication_style, get_color_palette  # noqa: E402

OUT = PROJECT_ROOT / "results" / "outputs"
FIG = PROJECT_ROOT / "results" / "figures"


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _make_comparison_figure(live: dict, archived: dict | None):
    """Generate a side-by-side comparison figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    setup_publication_style()
    colors = get_color_palette()
    c_live_hold, c_live_kern = colors[4], colors[3]
    c_arch_hold, c_arch_kern = colors[2], colors[1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Left panel: position error vs gap length ---
    gap_labels = ["10 min", "30 min", "60 min", "120 min", "240 min"]
    gap_keys = ["600", "1800", "3600", "7200", "14400"]

    live_hold, live_kernel = [], []
    for k in gap_keys:
        g = live.get("by_gap_length_s", {}).get(k)
        if g:
            pe = g["pos_error_median_m"]
            live_hold.append(pe.get("hold", np.nan))
            live_kernel.append(pe.get("kernel", np.nan))
        else:
            live_hold.append(np.nan)
            live_kernel.append(np.nan)

    arch_hold, arch_kernel = [], []
    if archived:
        for k in gap_keys:
            g = archived.get("by_gap_length_s", {}).get(k)
            if g:
                pe = g["pos_error_median_m"]
                arch_hold.append(pe.get("hold", np.nan))
                arch_kernel.append(pe.get("kernel", np.nan))
            else:
                arch_hold.append(np.nan)
                arch_kernel.append(np.nan)

    x = np.arange(len(gap_labels))
    w = 0.18
    ax = axes[0]
    ax.bar(x - 1.5*w, live_hold, w, label="Live: persistence", color=c_live_hold, alpha=0.8)
    ax.bar(x - 0.5*w, live_kernel, w, label="Live: kernel", color=c_live_kern, alpha=0.8)
    if archived:
        ax.bar(x + 0.5*w, arch_hold, w, label="Archived: persistence", color=c_arch_hold, alpha=0.6, hatch="//")
        ax.bar(x + 1.5*w, arch_kernel, w, label="Archived: kernel", color=c_arch_kern, alpha=0.6, hatch="//")
    ax.set_xticks(x)
    ax.set_xticklabels(gap_labels)
    ax.set_ylabel("Median 3-D position error (m)")
    ax.set_title("Position error under 3-satellite obstruction")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # --- Right panel: kernel vs persistence improvement (%) ---
    live_imp = []
    for k in gap_keys:
        g = live.get("by_gap_length_s", {}).get(k)
        if g and g.get("kernel_vs_hold_pct") is not None:
            live_imp.append(g["kernel_vs_hold_pct"])
        else:
            live_imp.append(np.nan)

    arch_imp = []
    if archived:
        for k in gap_keys:
            g = archived.get("by_gap_length_s", {}).get(k)
            if g and g.get("kernel_vs_hold_pct_median") is not None:
                arch_imp.append(g["kernel_vs_hold_pct_median"])
            else:
                arch_imp.append(np.nan)

    ax2 = axes[1]
    ax2.plot(x, live_imp, "o-", color=c_live_kern, label="Live (NTRIP)", linewidth=1.5, markersize=5)
    if archived:
        ax2.plot(x, arch_imp, "s--", color=c_arch_kern, label="Archived (CDDIS)", linewidth=1.5, markersize=5)
    ax2.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(gap_labels)
    ax2.set_ylabel("Kernel vs persistence improvement (%)")
    ax2.set_title("Kernel improvement over persistence")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    fig.suptitle("TEP-PPP Live NTRIP vs Archived CDDIS: Position Error Under Obstruction",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig_path = FIG / "step_37_live_comparison.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def main():
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = TEPLogger("step_37c", log_file_path=log_dir / "step_37c.log")
    set_step_logger(logger)

    print_status("STEP 37c: Live NTRIP Report and Comparison", "TITLE")

    live = _load_json(OUT / "step_37b_ntrip_process.json")
    if not live:
        print_status("No live processing result found.", "ERROR")
        return
    if live.get("status") != "ok":
        print_status(f"Live processing status: {live.get('status')}", "WARN")
        return

    archived = _load_json(OUT / "step_33_position_demo.json")

    # Build comparison table from the actual gap keys in the live result.
    # For short live captures the keys can be 30/60/120/180/240 s; for long
    # captures they are the pre-registered 600/1800/3600/7200/14400 s windows.
    live_by_gap = live.get("by_gap_length_s", {})
    live_keys = [k for k in live_by_gap if live_by_gap[k]]
    archived_by_gap = archived.get("by_gap_length_s", {}) if archived else {}
    comparison = []
    for k in live_keys:
        lg = live_by_gap[k]
        ag = archived_by_gap.get(k)
        # infer a human label from the key in seconds
        sec = int(k)
        if sec >= 60:
            label = f"{sec//60} min" if sec % 60 == 0 else f"{sec} s"
        else:
            label = f"{sec} s"
        row = {"gap": label, "gap_s": sec}
        if lg:
            row["live_hold_m"] = lg["pos_error_median_m"].get("hold")
            row["live_kernel_m"] = lg["pos_error_median_m"].get("kernel")
            row["live_kernel_vs_hold_pct"] = lg.get("kernel_vs_hold_pct")
        if ag:
            row["archived_hold_m"] = ag["pos_error_median_m"].get("hold")
            row["archived_kernel_m"] = ag["pos_error_median_m"].get("kernel")
            row["archived_kernel_vs_hold_pct"] = ag.get("kernel_vs_hold_pct_median")
        comparison.append(row)

    # Generate figure
    fig_path = None
    try:
        fig_path = _make_comparison_figure(live, archived)
        print_status(f"Figure saved: {fig_path}", "SUCCESS")
    except Exception as e:
        print_status(f"Figure generation failed: {e}", "WARN")

    # Real-time latency (if available)
    latency = live.get("processing_latency_ms")

    report = {
        "test": "ntrip_live_report",
        "status": "ok",
        "blind_protocol_frozen": True,
        "live_result": {
            "n_stations": live.get("n_stations"),
            "recording_start_utc": live.get("recording_start_utc"),
            "recording_duration_s": live.get("recording_duration_s"),
            "caster": live.get("caster"),
            "kernel_prior": live.get("kernel_prior"),
        },
        "archived_result": {
            "n_station_days": archived.get("n_station_days") if archived else None,
            "n_stations": archived.get("n_stations") if archived else None,
        } if archived else None,
        "comparison_table": comparison,
        "figure": str(fig_path.relative_to(PROJECT_ROOT)) if fig_path else None,
        "processing_latency_ms": latency,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (OUT / "step_37c_ntrip_report.json").write_text(json.dumps(report, indent=2))
    print_status(f"Report saved: {OUT / 'step_37c_ntrip_report.json'}", "SUCCESS")

    # Print comparison table
    print_status("\nComparison: Live (NTRIP) vs Archived (CDDIS)", "TITLE")
    print(f"{'Gap':>10s}  {'Live hold':>10s}  {'Live kern':>10s}  "
          f"{'Arch hold':>10s}  {'Arch kern':>10s}")

    def _fmt(row, key):
        v = row.get(key)
        if isinstance(v, (int, float, np.floating)) and np.isfinite(v):
            return f"{v:.1f}"
        return "--"

    for row in comparison:
        lh, lk = _fmt(row, 'live_hold_m'), _fmt(row, 'live_kernel_m')
        ah, ak = _fmt(row, 'archived_hold_m'), _fmt(row, 'archived_kernel_m')
        print(f"{row['gap']:>10s}  {lh:>10s}  {lk:>10s}  {ah:>10s}  {ak:>10s}")


if __name__ == "__main__":
    main()
