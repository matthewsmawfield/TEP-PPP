#!/usr/bin/env python3
"""
STEP 62  --  PPP cold-start convergence: does a TEP clock prior speed it up?

Step 23 showed a network clock prior is a no-op for a *converged* 24 h
multi-GNSS phase PPP solution (the clock is already observation-limited to
~1 cm).  The open question is the transient: from a cold start, does
constraining the receiver clock to the TEP network prediction
(sigma ~ 0.6 m, Step 58) shorten the time-to-converge?

Mechanism: PPP models the receiver clock as white noise (re-estimated freely
each epoch).  Early on, the clock, the float ambiguities and the position are
strongly correlated and the filter cannot separate them.  Pinning the clock to
an external prediction frees one degree of freedom, so the ambiguities and
position should condition sooner.

This driver reuses the Step 23 PRIDE-PPP-AR machinery:
  arm A  free clock (TEP_CLK_PRIOR unset)
  arm B  distance-kernel TEP clock prior (step_22 --arm C), sigma from Step 58
  arm C  persistence / AR1 clock prior at the same sigma (deployable control)

It parses the per-epoch kinematic solution and reports, against the rover's
known ITRF coordinate:
  * 3-D error at 2 / 5 / 10 / 20 / 40 / 60 min from the first epoch
  * time-to-converge  (first epoch with 3-D error < 0.10 m that stays
    below 0.15 m; the 1.5x tolerance prevents a single noisy epoch from
    resetting the convergence clock)

Usage:
  python scripts/steps/step_62_ppp_convergence.py \
      --rundir <staged PRIDE 2024/100 dir> --rover ZIMM \
      --rover-xyz "X,Y,Z" --date 2024100

Output: results/outputs/step_62_ppp_convergence.json
"""
from __future__ import annotations

import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
OUT_DIR = PROJECT_ROOT / "results" / "outputs"
OUT = OUT_DIR / "step_62_ppp_convergence.json"

# Step 58 dense-network predictable fraction -> clock-prior sigma
PREDICTABLE_FRACTION = 0.64
CLOCK_RMS_M = 1.16
PRIOR_SIGMA_M = float(np.sqrt(max(1.0 - PREDICTABLE_FRACTION, 1e-3)) * CLOCK_RMS_M)  # ~0.70 m

CONV_THRESH_M = 0.10
import os as _os
# Focus the sampling where the effect lives (the shortest cold starts) and take
# many independent start epochs so the gain gets a real confidence interval.
WINDOWS_MIN = [int(x) for x in _os.environ.get("TEP_PPP_WINDOWS", "2,3,5,10").split(",")]
_N_STARTS = int(_os.environ.get("TEP_PPP_NSTARTS", "24"))
# snap to the 300 s grid: RINEX epochs are on 30 s, and a session that starts
# off-grid loses epochs a 2-minute window cannot spare (most such runs return
# no solution at all).
START_SODS = sorted({int(round(x / 300.0) * 300) for x in
                     __import__("numpy").linspace(1800, 84300, _N_STARTS)})


def _kin_xyz(tokens):
    """Parse ECEF from a PRIDE kin epoch.  Windowed kinematic lines insert a
    quality flag (`*`) after the seconds-of-day token; full-day files do not."""
    if not (len(tokens) >= 5 and tokens[0].isdigit() and len(tokens[0]) == 5):
        return None
    rest = tokens[2:]
    if rest and rest[0] in ("*", "X", "P"):
        rest = rest[1:]
    if len(rest) < 3:
        return None
    try:
        return np.array([float(rest[0]), float(rest[1]), float(rest[2])])
    except ValueError:
        return None


def _epoch_series(kin_path, truth_xyz):
    """(t_sec_from_start, err3d_m) per epoch from a PRIDE kin_ file."""
    rows = []
    for line in Path(kin_path).read_text().splitlines():
        p = line.split()
        xyz = _kin_xyz(p)
        if xyz is None:
            continue
        try:
            mjd = float(p[0]) + float(p[1]) / 86400.0
        except ValueError:
            continue
        rows.append((mjd, float(np.linalg.norm(xyz - truth_xyz))))
    if not rows:
        return np.array([]), np.array([])
    rows.sort()
    t0 = rows[0][0]
    t = np.array([(m - t0) * 86400.0 for m, _ in rows])
    e = np.array([x for _, x in rows])
    return t, e


def _converge_time(t, e):
    for i in range(len(e)):
        if e[i] < CONV_THRESH_M and np.all(e[i:] < CONV_THRESH_M * 1.5):
            return float(t[i])
    return None


def _at(t, e, minutes):
    if len(t) == 0:
        return None
    j = int(np.searchsorted(t, minutes * 60.0))
    j = min(max(j, 0), len(e) - 1)
    return float(e[j])


def _arm_curve(res_dir, kin_name, truth):
    kin = res_dir / kin_name
    if not kin.exists():
        return None
    t, e = _epoch_series(kin, truth)
    if len(t) < 10:
        return None
    return {
        "n_epochs": len(t),
        "converge_time_s": _converge_time(t, e),
        "err_at_min": {str(m): _at(t, e, m) for m in CHECK_MIN},
        "final_err_m": float(np.median(e[len(e) * 2 // 3:])),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rundir")
    ap.add_argument("--rover", default="ZIMM")
    ap.add_argument("--rover-xyz", help="comma-separated ITRF X,Y,Z (m)")
    ap.add_argument("--date", default="2024100")
    ap.add_argument("--product", default="clk_cod")
    ap.add_argument("--min-ref-km", type=float, default=50.0,
                    help="minimum rover-reference baseline for the clock prior. "
                         "COD0OPSFIN contains ZIM2/ZIM3 at 0.0 km and WAB1/WAB2 at "
                         "5.2 km from ZIMM; a co-located receiver shares the site and "
                         "often the frequency standard, so including one would test "
                         "'the receiver next door' rather than the distance-structured "
                         "field.  50 km excludes the immediate neighbourhood.")
    ap.add_argument("--gps-only", action="store_true",
                    help="restrict PRIDE to GPS L1/L2 (urban-canyon / few-sat regime)")
    ap.add_argument("--elev-cutoff", type=float, default=None,
                    help="observation-mask elevation (deg).  Site line cutoff, not AR-only.")
    ap.add_argument("--k", type=int, default=None,
                    help="keep the k highest-elevation GPS satellites per epoch in the RINEX")
    ap.add_argument("--phase-v2-only", action="store_true",
                    help="run free/persistence/PN_v2 only; requires a product containing the rover's D-1 clock history")
    ap.add_argument("--out", default=None,
                    help="output JSON path; default is step_62_ppp_degraded.json when degraded")
    a = ap.parse_args()
    degraded = bool(a.gps_only or a.elev_cutoff is not None or a.k)
    out_path = Path(a.out) if a.out else (
        OUT_DIR / "step_62_ppp_degraded.json" if degraded else OUT)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "step": 62,
        "title": ("PPP cold-start convergence with a TEP clock prior"
                  + (" (degraded geometry)" if degraded else "")),
        "geometry": {
            "gps_only": bool(a.gps_only),
            "elev_cutoff_deg": a.elev_cutoff,
            "k_highest_elev": a.k,
            "note": ("clean full-sky phase PPP is observation-limited on the clock; "
                     "the operational niche is GPS-only / high-elevation / k-thinned"),
        },
        "prior_arm": "PN_v2" if a.phase_v2_only else "PN",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prior_sigma_m": round(PRIOR_SIGMA_M, 3),
        "min_ref_km": None,   # filled from args below
        "prior_sigma_note": (f"TEP arm: per-epoch sigma from step_22 (base ~{PRIOR_SIGMA_M:.3f} m, "
                             "modulated ±30% by TEP astronomical phases via tep_process_noise_scale). "
                             f"Persistence arm: flat sigma = {PRIOR_SIGMA_M:.3f} m. "
                             f"Base from sqrt(1 - {PREDICTABLE_FRACTION}) * {CLOCK_RMS_M} m, Step 58."),
        "trend_mode": "causal",
        "trend_note": ("For each cold-start window starting at t0, the prior's absolute "
                       "trend is fit only to target rck data with sod <= t0, removing the "
                       "full-day look-ahead bias.  A persistence control (target-only "
                       "causal linear extrapolation, no network) is run at the same sigma."),
    }

    if not a.rundir or not Path(a.rundir).exists():
        out["status"] = "needs_inputs"
        out["required"] = {
            "rundir": ("a staged PRIDE-PPP-AR run directory (as in Step 23): RINEX + "
                       "SP3/CLK/BIA/ERP/ATX for one 24 h static station"),
            "rover_xyz": "the station's known ITRF X,Y,Z (m), for the error reference",
            "engine": "archive/PRIDE-PPPAR/bin/lsq with the lsq_add_clkcon patch (present)",
            "clock_prior": ("built causally by step_22 --arm C from a dense CORS network "
                            "(step_61) or the ppp_dense product"),
        }
        out["protocol"] = [
            "run arm A (free clock) and arm B (TEP clock prior at sigma ~0.70 m)",
            "optionally arm C (persistence/AR1 prior at the same sigma) as the control",
            "parse the per-epoch kin_ solution; error vs the known ITRF coordinate",
            "report 3-D error at 2/5/10/20/40/60 min and time-to-converge (<0.10 m)",
            "repeat from several restart epochs through the day; day-cluster the A-B and C-B deltas",
        ]
        out_path.write_text(json.dumps(out, indent=2))
        print(json.dumps(out, indent=2))
        return 0

    # --- live run: genuine windowed COLD STARTS ---
    # PRIDE's lsq is a BATCH least-squares over the whole session, so a single
    # 24 h run has no convergence transient at all -- every epoch already sees
    # the full day.  To measure convergence the session itself must be short.
    # The config's "Session time" line sets that, so each window is a separate
    # cold-start solution.  Truth is the SINEX ITRF coordinate: the RINEX header
    # APPROX POSITION is only metre-accurate and would swamp the whole effect.
    import shutil, subprocess, tempfile
    from scripts.lib import tep_sinex as sx
    from scripts.steps.step_23_pride_endtoend import sdk_env, PRIDE, ATX

    rundir = Path(a.rundir).resolve()
    year, doy = int(a.date[:4]), int(a.date[4:])
    truth = (np.array([float(x) for x in a.rover_xyz.split(",")])
             if a.rover_xyz else np.asarray(sx.station_xyz(a.rover, year, doy), float))
    obs_files = sorted(rundir.parent.parent.glob("*.rnx"))
    if not obs_files:
        out["status"] = "no_rinex"; out_path.write_text(json.dumps(out, indent=2)); return 1
    obs_full = str(obs_files[0])
    work = Path(tempfile.mkdtemp(prefix="step62_"))
    gen = PROJECT_ROOT / "scripts" / "steps" / "step_22_make_clock_prior.py"
    obs_eval = obs_full
    if a.k:
        from scripts.lib.rinex_thin import (load_sp3, sat_grid, crx_to_text,
                                            thin_rinex_stream)
        sp3_dir = PROJECT_ROOT / "data" / "rinex" / "sp3"
        sp3 = load_sp3(a.date, sp3_dir, systems=("G",))
        if sp3 is None:
            out["status"] = "no_sp3_for_thin"
            out_path.write_text(json.dumps(out, indent=2))
            return 1
        grid = np.arange(0.0, 86400.0, 30.0)
        thinned = work / f"thinned_k{a.k}.rnx"
        text = crx_to_text(Path(obs_full))
        with open(thinned, "w") as fout:
            thin_rinex_stream(iter(text.splitlines(keepends=True)), fout,
                              sat_grid(sp3, grid), k=a.k, strategy="highest",
                              elev_min_deg=10.0, station_ecef=truth,
                              system="G")
        obs_eval = str(thinned)
        print(f"thinned RINEX to k={a.k} GPS -> {obs_eval}", flush=True)

    def run_window(tag, t0, dur, prior, degrade=True):
        d = work / tag
        if d.exists():
            shutil.rmtree(d)
        shutil.copytree(rundir, d, symlinks=True,
                        ignore=shutil.ignore_patterns('*.SP3', '*.CLK', '*.BIA', '*.ERP'))
        for big in rundir.glob('*'):
            if big.suffix.upper() in ('.SP3', '.CLK', '.BIA', '.ERP') and not (d / big.name).exists():
                (d / big.name).symlink_to(big.resolve())
        junk = [f"rck_{a.date}", f"res_{a.date}", f"ztd_{a.date}", "abs_igs.atx"]
        if degrade:
            junk += ["tmp_obs", "tmp_rem", "tmp_cid"]
        for name in junk:
            (d / name).unlink(missing_ok=True)
        (d / "abs_igs.atx").symlink_to(ATX)
        cfg = next(d.glob("config.*"))
        txt = cfg.read_text().splitlines()
        hh, mm, ss = t0 // 3600, (t0 % 3600) // 60, t0 % 60
        dt = datetime.strptime(a.date, "%Y%j")
        for k, ln in enumerate(txt):
            s = ln.strip()
            if s.startswith("Session time"):
                txt[k] = (f"Session time            = {dt.year} {dt.month:02d} {dt.day:02d} "
                          f"{hh:02d} {mm:02d} {ss:.2f} {float(dur):.2f}")
            apply_mask = degrade and (a.gps_only or a.elev_cutoff is not None)
            if apply_mask and a.gps_only and s.startswith("Frequency combination"):
                txt[k] = "Frequency combination  = G12"
            if apply_mask and a.elev_cutoff is not None and s.startswith("Cutoff elevation"):
                txt[k] = f"Cutoff elevation        = {int(a.elev_cutoff)}"
            # Observation mask is SITE%cutoff on the station line (3rd numeric
            # after GMF), not the AR-only Cutoff elevation keyword.  name/skd/map
            # are column-sensitive (`K  GMF`); do not collapse that spacing.
            if apply_mask and a.elev_cutoff is not None and ln.startswith(" ") and "  GMF " in ln:
                head, rest = ln.split("  GMF ", 1)
                toks = rest.split()
                if len(toks) >= 3:
                    toks[2] = str(int(a.elev_cutoff))
                    txt[k] = head + "  GMF " + " ".join(toks)
            if apply_mask and a.gps_only and len(ln) > 4 and ln[0] == " " and ln[1] in "RECJ" and ln[2:4].isdigit():
                txt[k] = "#" + ln[1:]
        cfg.write_text("\n".join(txt) + "\n")
        env = sdk_env()
        if prior:
            env["TEP_CLK_PRIOR"] = str(prior)
        else:
            env.pop("TEP_CLK_PRIOR", None)
        obs_this = obs_eval if degrade else obs_full
        local_obs = d / "obs.rnx"
        if Path(obs_this).resolve() != local_obs.resolve():
            try:
                local_obs.symlink_to(Path(obs_this).resolve())
            except OSError:
                shutil.copy2(obs_this, local_obs)
        with open(d / "lsq.log", "w") as log:
            subprocess.run([str(PRIDE / "bin" / "lsq"), cfg.name, local_obs.name],
                           cwd=d, env=env, stdout=log, stderr=subprocess.STDOUT)
        kin = d / f"kin_{a.date}_{a.rover.lower()}"
        applied = "lsq_add_clkcon" in (d / "lsq.log").read_text(errors="replace")
        rows = []
        if kin.exists():
            for ln in kin.read_text().splitlines():
                xyz = _kin_xyz(ln.split())
                if xyz is not None:
                    rows.append(xyz.tolist())
        keep_rck = (tag == "passA")
        if keep_rck:
            for f in list(d.iterdir()):
                if f.is_file() and not f.name.startswith("rck_"):
                    try:
                        f.unlink()
                    except OSError:
                        pass
        else:
            # Each window copies the whole staged PRIDE directory (products
            # included).  Without this the 190-odd solves fill the disk.
            if len(rows) >= 3:
                shutil.rmtree(d, ignore_errors=True)
        if len(rows) < 3:
            logp = d / "lsq.log"
            tail = ""
            if logp.exists():
                lines = logp.read_text(errors="replace").splitlines()
                tail = " | ".join(lines[-8:])
            print(f"    {tag} kin_rows={len(rows)} applied={applied} {tail}",
                  flush=True)
            return None, applied
        arr = np.array(rows)
        tail = arr[max(len(arr) * 2 // 3, 1):]
        # Per-window steady-state bias: 3-D norm of the mean position over
        # the last 1/3 of the window minus truth.  This measures the
        # converged bias, not per-epoch RMS; the aggregate across windows
        # (by_window.*_err_m_rms) takes the RMS of these bias values.
        return float(np.linalg.norm(tail.mean(0) - truth)), applied

    # pass 1 (full day, free clock) supplies the absolute clock trend the
    # prior generator needs.  Do not apply the degraded-sky mask here: the
    # trend must be a clean PRIDE clock, and a 50 deg GPS-only 24 h run can
    # fail to write rck.  Evaluation windows below carry the mask.
    # 86370 s = last 30 s epoch in a 24 h file (0, 30, ..., 86370).
    print("passA: full-day free-clock trend (clean sky)", flush=True)
    full_free, _ = run_window("passA", 0, 86370, None, degrade=False)
    rck = work / "passA" / f"rck_{a.date}"
    if not rck.exists():
        out["status"] = "passA_no_rck"
        out["full_day_free_clock_err_m"] = full_free
        out_path.write_text(json.dumps(out, indent=2))
        print(f"passA did not write {rck}", flush=True)
        return 1
    print(f"passA rck={rck} full_day_err={full_free}", flush=True)

    # ---- Causal prior generation -------------------------------------------
    # The previous implementation generated one prior using a 4th-order
    # polynomial fit to the *full-day* rck file and reused it for every
    # cold-start window.  That polynomial sees target clock data from
    # after the window start, which is a look-ahead bias.
    #
    # Fix: for each start time t0, regenerate the prior with
    # --trend-end-sod t0 so the trend polynomial is fit only to target
    # clock data strictly before the window.  The network variation
    # (pred_var) is target-free and uses reference-station data at each
    # epoch, which is the real-time deployment model.  A persistence
    # control arm (target-only causal linear extrapolation, no network)
    # is generated at the same sigma so the TEP-specific gain can be
    # separated from the generic "any clock prior helps" effect.
    prior_cache: dict[int, dict[str, Path]] = {}   # t0 -> {"tep": path, "pers": path}

    def _make_causal_priors(t0: int) -> dict[str, Path]:
        if t0 in prior_cache:
            return prior_cache[t0]
        tep_path = work / f"prior_C_t{t0}.cpr"
        pers_path = work / f"prior_pers_t{t0}.cpr"
        # TEP arm: causal trend + network kriging variation
        subprocess.run([sys.executable, str(gen), "--rover", a.rover, "--date", a.date,
                        "--product", a.product, "--arm", "C", "--rover-xyz",
                        ",".join(f"{v:.4f}" for v in truth), "--from-rck", str(rck),
                        "--trend-end-sod", str(t0), "--min-ref-km", str(a.min_ref_km),
                        "-o", str(tep_path)], check=True, stdout=subprocess.DEVNULL)
        # Preserve the per-epoch sigma from step_22: it now carries TEP
        # astronomical phase modulation (sigma varies ±30% per epoch).
               # Only the persistence arm uses the flat PRIOR_SIGMA_M.
        # Persistence arm: target-only causal linear extrapolation, same sigma
        subprocess.run([sys.executable, str(gen), "--rover", a.rover, "--date", a.date,
                        "--product", a.product, "--arm", "persistence", "--rover-xyz",
                        ",".join(f"{v:.4f}" for v in truth), "--from-rck", str(rck),
                        "--trend-end-sod", str(t0), "--sigma-m", str(PRIOR_SIGMA_M),
                        "--min-ref-km", str(a.min_ref_km),
                        "-o", str(pers_path)], check=True, stdout=subprocess.DEVNULL)
        # MATCHED-SIGMA TEP ARM.  The TEP and persistence arms above differ in
        # THREE ways, not one: the network variation term (the quantity under
        # test), the base sigma (TEP uses a data-driven leave-one-out RMSE while
        # persistence uses the flat PRIOR_SIGMA_M), and the astronomical per-epoch
        # modulation (TEP only).  A filter weights a prior by its sigma, so two of
        # those differences change the answer for reasons unrelated to whether the
        # network prediction carries information.  This arm is arm C run at the
        # SAME flat sigma as persistence, so tep_flat vs persistence isolates the
        # network term alone -- both have the identical causal trend and the
        # identical prior precision.
        tepflat_path = work / f"prior_C_flat_t{t0}.cpr"
        subprocess.run([sys.executable, str(gen), "--rover", a.rover, "--date", a.date,
                        "--product", a.product, "--arm", "C", "--rover-xyz",
                        ",".join(f"{v:.4f}" for v in truth), "--from-rck", str(rck),
                        "--trend-end-sod", str(t0), "--sigma-m", str(PRIOR_SIGMA_M),
                        "--min-ref-km", str(a.min_ref_km),
                        "-o", str(tepflat_path)], check=True, stdout=subprocess.DEVNULL)
        # NESTED ARM.  tep_flat still differs from persistence in its TREND: arm C
        # fits a polynomial over all causal history while the persistence arm fits
        # a line to the last 30 minutes, and a clock is locally linear but globally
        # curved, so the short fit is the better short-horizon predictor on its own.
        # Arm PN takes the persistence trend and ADDS the network term, so
        # PN vs persistence isolates the network contribution with trend and sigma
        # both held fixed.  This is the comparison the operational question asks.
        pn_path = work / f"prior_PN_t{t0}.cpr"
        subprocess.run([sys.executable, str(gen), "--rover", a.rover, "--date", a.date,
                        "--product", a.product, "--arm", "PN", "--rover-xyz",
                        ",".join(f"{v:.4f}" for v in truth), "--from-rck", str(rck),
                        "--trend-end-sod", str(t0), "--sigma-m", str(PRIOR_SIGMA_M),
                        "--min-ref-km", str(a.min_ref_km),
                        "-o", str(pn_path)], check=True, stdout=subprocess.DEVNULL)
        pnv2_path = work / f"prior_PN_v2_t{t0}.cpr"
        if a.phase_v2_only:
            subprocess.run([sys.executable, str(gen), "--rover", a.rover, "--date", a.date,
                            "--product", a.product, "--arm", "PN_v2", "--rover-xyz",
                            ",".join(f"{v:.4f}" for v in truth), "--from-rck", str(rck),
                            "--trend-end-sod", str(t0), "--sigma-m", str(PRIOR_SIGMA_M),
                            "--min-ref-km", str(a.min_ref_km),
                            "-o", str(pnv2_path)], check=True, stdout=subprocess.DEVNULL)
        paths = {"tep": tep_path, "pers": pers_path, "tep_flat": tepflat_path,
                 "pn": pn_path, "pn_v2": pnv2_path}
        prior_cache[t0] = paths
        return paths

    rows = []
    for wmin in WINDOWS_MIN:
        for t0 in START_SODS:
            dur = wmin * 60
            priors = _make_causal_priors(t0)
            ef, apf = run_window(f"f_{wmin}_{t0}", t0, dur, None)
            et, apt = (run_window(f"t_{wmin}_{t0}", t0, dur, priors["tep"])
                       if not a.phase_v2_only else (None, None))
            ep, app = run_window(f"p_{wmin}_{t0}", t0, dur, priors["pers"])
            # matched-sigma TEP: identical causal trend and identical flat sigma
            # to the persistence arm, so the ONLY difference is the network term
            etf, aptf = (run_window(f"tf_{wmin}_{t0}", t0, dur, priors["tep_flat"])
                         if not a.phase_v2_only else (None, None))
            # persistence trend + network: the nested isolation of the network term
            epn, appn = (run_window(f"pn_{wmin}_{t0}", t0, dur, priors["pn"])
                         if not a.phase_v2_only else (None, None))
            epnv2, appnv2 = (run_window(f"pnv2_{wmin}_{t0}", t0, dur, priors["pn_v2"])
                             if a.phase_v2_only else (None, None))
            primary = epnv2 if a.phase_v2_only else (epn if epn is not None else et)
            primary_applied = appnv2 if a.phase_v2_only else (appn if epn is not None else apt)
            if ef is None or primary is None:
                print(f"  skip {wmin} min start {t0}s  free={ef} tep={et} "
                      f"pers={ep} applied={primary_applied}", flush=True)
                continue
            row = {"window_min": wmin, "start_sod": t0,
                   "free_clock_err_m": ef, "tep_prior_err_m": primary,
                   "improvement_pct": (100.0 * (1.0 - primary / ef)
                                       if ef > 0 else None),
                   "prior_applied": bool(primary_applied)}
            # Keep arm C (polynomial trend + network) as a diagnostic
            if et is not None:
                row["tep_armC_err_m"] = et
                row["tep_armC_improvement_pct"] = (100.0 * (1.0 - et / ef)
                                                    if ef > 0 else None)
            if ep is not None:
                row["persistence_err_m"] = ep
                row["persistence_improvement_pct"] = 100.0 * (1.0 - ep / ef) if ef > 0 else None
                # TEP vs persistence uses PN (the fair isolation: same trend,
                # same sigma, network added) as the primary TEP arm
                tep_primary = primary
                row["tep_vs_persistence_pct"] = (100.0 * (1.0 - tep_primary / ep)
                                                 if ep > 0 else None)
                row["persistence_applied"] = bool(app)
                if etf is not None:
                    row["tep_flat_err_m"] = etf
                    row["tep_flat_improvement_pct"] = (100.0 * (1.0 - etf / ef)
                                                       if ef > 0 else None)
                    # the clean isolation of the network contribution
                    row["tep_flat_vs_persistence_pct"] = (100.0 * (1.0 - etf / ep)
                                                          if ep > 0 else None)
                    row["tep_flat_applied"] = bool(aptf)
            if epn is not None:
                    row["pn_err_m"] = epn
                    row["pn_improvement_pct"] = 100.0 * (1.0 - epn / ef) if ef > 0 else None
                    # THE isolation: same trend, same sigma, network added
                    row["pn_vs_persistence_pct"] = 100.0 * (1.0 - epn / ep) if ep > 0 else None
                    row["pn_applied"] = bool(appn)
            if epnv2 is not None:
                row["pn_v2_err_m"] = epnv2
                row["pn_v2_improvement_pct"] = 100.0 * (1.0 - epnv2 / ef) if ef > 0 else None
                row["pn_v2_vs_persistence_pct"] = 100.0 * (1.0 - epnv2 / ep) if ep > 0 else None
                row["pn_v2_applied"] = bool(appnv2)
            rows.append(row)
            display_tep = primary if a.phase_v2_only else et
            pers_str = (f"  pers {ep:8.4f}  TEPvP {100*(1-display_tep/ep):+6.1f}%"
                        + (f"  TEPflat {etf:8.4f}  flatVp {100*(1-etf/ep):+6.1f}%"
                           if etf is not None else "")
                        + (f"  PN {epn:8.4f}  PNvP {100*(1-epn/ep):+6.1f}%"
                           if epn is not None else "")) if ep is not None else ""
            print(f"  {wmin:3d} min  start {t0:6d}s   free {ef:8.4f} m   "
                  f"TEP {display_tep:8.4f} m   {100*(1-display_tep/ef):+6.1f}%   "
                  f"prior_applied={primary_applied}{pers_str}", flush=True)

    out["min_ref_km"] = a.min_ref_km
    out["prior_product"] = a.product
    out["status"] = "ok"
    out["truth_source"] = "SINEX ITRF" if not a.rover_xyz else "user"
    out["full_day_free_clock_err_m"] = full_free
    out["windows"] = rows
    if rows:
        valid = [r for r in rows if r["improvement_pct"] is not None]
        by = {}
        for w in WINDOWS_MIN:
            sel = [r for r in valid if r["window_min"] == w]
            if sel:
                # RMS-based improvement is the headline statistic the manuscript
                # cites: 100*(1 - RMS_tep/RMS_free).  RMS weights larger errors
                # more than the arithmetic mean, matching the "3-D position RMS"
                # language in the abstract and Section 3.6i.
                rms_free = float(np.sqrt(np.mean([r["free_clock_err_m"]**2 for r in sel])))
                rms_tep = float(np.sqrt(np.mean([r["tep_prior_err_m"]**2 for r in sel])))
                mean_free = float(np.mean([r["free_clock_err_m"] for r in sel]))
                mean_tep = float(np.mean([r["tep_prior_err_m"] for r in sel]))
                by[str(w)] = {
                    "n_starts": len(sel),
                    "free_clock_err_m_rms": rms_free,
                    "tep_prior_err_m_rms": rms_tep,
                    "improvement_pct_rms": float(100.0 * (1.0 - rms_tep / rms_free)) if rms_free > 0 else None,
                    "free_clock_err_m_median": float(np.median([r["free_clock_err_m"] for r in sel])),
                    "tep_prior_err_m_median": float(np.median([r["tep_prior_err_m"] for r in sel])),
                    "improvement_pct_median": float(np.median([r["improvement_pct"] for r in sel])),
                    "free_clock_err_m_mean": mean_free,
                    "tep_prior_err_m_mean": mean_tep,
                    "improvement_pct_from_mean": float(100.0 * (1.0 - mean_tep / mean_free)) if mean_free > 0 else None,
                    "n_improved": sum(1 for r in sel if r["improvement_pct"] > 0),
                    "n_degraded": sum(1 for r in sel if r["improvement_pct"] < 0),
                }
                # Persistence comparison for this window size
                pers_sel = [r for r in sel if r.get("persistence_err_m") is not None]
                if pers_sel:
                    rms_pers = float(np.sqrt(np.mean([r["persistence_err_m"]**2 for r in pers_sel])))
                    by[str(w)]["persistence_err_m_rms"] = rms_pers
                    by[str(w)]["persistence_improvement_pct_rms"] = (
                        float(100.0 * (1.0 - rms_pers / rms_free)) if rms_free > 0 else None)
                    by[str(w)]["tep_vs_persistence_pct_rms"] = (
                        float(100.0 * (1.0 - rms_tep / rms_pers)) if rms_pers > 0 else None)
        out["by_window"] = by
        # Overall statistics (the numbers the manuscript cites)
        n_pos = sum(1 for r in valid if r["improvement_pct"] > 0)
        n_neg = sum(1 for r in valid if r["improvement_pct"] < 0)
        diffs = [r["free_clock_err_m"] - r["tep_prior_err_m"] for r in valid]
        mean_diff = float(np.mean(diffs)) if diffs else 0.0
        # Bootstrap CI for the paired reduction
        rng = np.random.default_rng(42)
        boot = [float(np.mean(rng.choice(diffs, len(diffs), replace=True)))
                for _ in range(10000)] if diffs else []
        ci95 = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))] if boot else [0.0, 0.0]
        # Bootstrap CI for the median per-start improvement
        imps = [r["improvement_pct"] for r in valid]
        boot_med = [float(np.median(rng.choice(imps, len(imps), replace=True)))
                    for _ in range(10000)] if imps else []
        med_ci95 = [float(np.percentile(boot_med, 2.5)), float(np.percentile(boot_med, 97.5))] if boot_med else [0.0, 0.0]
        # Binomial sign test (one-sided, greater)
        from scipy.stats import binom
        p_sign = float(binom.cdf(n_neg, n_pos + n_neg, 0.5)) if (n_pos + n_neg) > 0 else 1.0
        p_sign_two = float(2.0 * min(p_sign, 1.0 - p_sign)) if (n_pos + n_neg) > 0 else 1.0
        # Overall RMS improvement (the manuscript headline: "5.2% over 60 cold starts")
        all_rms_free = float(np.sqrt(np.mean([r["free_clock_err_m"]**2 for r in valid])))
        all_rms_tep = float(np.sqrt(np.mean([r["tep_prior_err_m"]**2 for r in valid])))
        overall_imp_rms = float(100.0 * (1.0 - all_rms_tep / all_rms_free)) if all_rms_free > 0 else None
        rms_imps = [v.get("improvement_pct_rms") for v in by.values()
                    if v.get("improvement_pct_rms") is not None]
        out["headline"] = {
            "by_window_improvement_pct": {k: v.get("improvement_pct_rms") for k, v in by.items()},
            "by_window_improvement_pct_median": {k: v.get("improvement_pct_median") for k, v in by.items()},
            "by_window_improvement_pct_from_mean": {k: v.get("improvement_pct_from_mean") for k, v in by.items()},
            "overall_improvement_pct_rms": overall_imp_rms,
            "best_improvement_pct": max(rms_imps) if rms_imps else None,
            "short_window_improvement_pct": by.get("2", {}).get("improvement_pct_rms"),
            "n_starts": len(valid),
            "n_improved": n_pos,
            "n_degraded": n_neg,
            "sign_test_p_one_sided": p_sign,
            "sign_test_p_two_sided": p_sign_two,
            "paired_reduction_m": mean_diff,
            "paired_reduction_ci95": ci95,
            "median_improvement_pct": float(np.median(imps)) if imps else None,
            "median_improvement_ci95": med_ci95,
            # This is deliberately only the value-of-*a-clock-prior* result.
            # It is not evidence that the network/TEP component adds value over
            # an equally precise causal persistence prior; that nested question
            # is decided below once the matched comparison is available.
            "any_clock_prior_convergence_gain_detected": bool(
                overall_imp_rms is not None and overall_imp_rms > 5.0),
        }
        # Overall persistence comparison
        pers_valid = [r for r in valid if r.get("persistence_err_m") is not None]
        if pers_valid:
            all_rms_pers = float(np.sqrt(np.mean([r["persistence_err_m"]**2 for r in pers_valid])))
            out["headline"]["persistence_err_m_rms"] = all_rms_pers
            out["headline"]["persistence_improvement_pct_rms"] = (
                float(100.0 * (1.0 - all_rms_pers / all_rms_free)) if all_rms_free > 0 else None)
            out["headline"]["tep_vs_persistence_pct_rms"] = (
                float(100.0 * (1.0 - all_rms_tep / all_rms_pers)) if all_rms_pers > 0 else None)
            # Paired TEP vs persistence
            tp_diffs = [r["tep_prior_err_m"] - r["persistence_err_m"] for r in pers_valid]
            tp_mean = float(np.mean(tp_diffs)) if tp_diffs else 0.0
            tp_boot = [float(np.mean(rng.choice(tp_diffs, len(tp_diffs), replace=True)))
                       for _ in range(10000)] if tp_diffs else []
            tp_ci95 = [float(np.percentile(tp_boot, 2.5)),
                       float(np.percentile(tp_boot, 97.5))] if tp_boot else [0.0, 0.0]
            out["headline"]["tep_vs_persistence_paired_m"] = tp_mean
            out["headline"]["tep_vs_persistence_ci95"] = tp_ci95
            tp_pos = sum(1 for r in pers_valid if r.get("tep_vs_persistence_pct", 0) > 0)
            tp_neg = sum(1 for r in pers_valid if r.get("tep_vs_persistence_pct", 0) < 0)
            out["headline"]["tep_vs_persistence_n_tep_wins"] = tp_pos
            out["headline"]["tep_vs_persistence_n_pers_wins"] = tp_neg
            out["headline"]["tep_vs_persistence_sign_p"] = float(
                binom.cdf(tp_neg, tp_pos + tp_neg, 0.5)) if (tp_pos + tp_neg) > 0 else 1.0
            # P3 requires a positive paired *reduction* (persistence error -
            # TEP error), not simply an improvement over an unconstrained free
            # clock.  Treating the latter as a TEP pass lets a generic prior
            # receive credit for the network term.
            out["headline"]["tep_specific_position_gain_detected"] = bool(
                tp_mean < 0.0 and tp_ci95[1] < 0.0)
            out["headline"]["p3_gate"] = {
                "status": ("pass" if out["headline"]["tep_specific_position_gain_detected"]
                           else "fail"),
                "criterion": ("matched persistence minus network-plus-persistence "
                              "paired 95% CI is wholly positive"),
                "paired_reduction_m": -tp_mean,
                "paired_reduction_ci95": [-tp_ci95[1], -tp_ci95[0]],
            }
    out_path.write_text(json.dumps(out, indent=2))
    print("\n" + json.dumps(out.get("headline", {}), indent=1))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
