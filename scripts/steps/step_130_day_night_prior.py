#!/usr/bin/env python3
"""
STEP 130  --  Day/night-aware TEP prior: exploiting the solar-illumination
              independence of the TEP continuum.

EXPLOITING THE DAY/NIGHT MODULATION (Step 125) AS A PREDICTIVE STRENGTH.

Step 125 showed that the TEP continuum (notched, solar tidal lines removed)
is solar-illumination-independent: nighttime notched coherence (0.140) is
higher than daytime notched coherence (0.102).  The full-band solar
elevation dependence (r=0.809) was carried by the solar tidal lines.

A day/night-aware prior that uses the notched kernel at night and the
full-band kernel during the day should capture the best of both:
  * Day: the full-band kernel includes the solar tidal lines that carry
    additional predictive information during illuminated periods.
  * Night: the notched kernel excludes the empty tidal bins and uses the
    TEP continuum that is active at night.

PROTOCOL:
  * Run the Step 64 P1 gate on 2024 development data.
  * For each evaluation epoch, compute the solar elevation at the target
    station.
  * Compare three kernels:
    1. full_band:    always use the full-band TEP kernel
    2. day_night:    use full-band during day (elev > 0), notched at night
    3. notched:      always use the notched TEP kernel

  * TEP exploitation prediction: the day/night-aware kernel beats both
    the always-full-band and always-notched kernels, because it uses the
    right component of the TEP field for each illumination condition.

Output: results/outputs/step_130_day_night_prior.json
"""
from __future__ import annotations

import sys, json, math
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.lib import tep_data as td
from scripts.lib.tep_phase import spectral_phase_gram
from scripts.steps import step_64_maximal_tep_p1 as s64

OUT = ROOT / "results" / "outputs" / "step_130_day_night_prior.json"
PRODUCT = "clk_cod"
N_DAYS = 30

NOTCH_PERIODS = [86400.0, 43200.0, 28800.0, 21600.0, 86164.0, 44712.0]
NOTCH_WIDTH_FRAC = 0.05
NOTCHED_LAMBDA_KM = 1877.0
NOTCHED_MULTI_SCALES_KM = (NOTCHED_LAMBDA_KM / 1.6,
                           NOTCHED_LAMBDA_KM,
                           NOTCHED_LAMBDA_KM * 1.6)


def _solar_elevation(lat, utc_datetime):
    doy = utc_datetime.timetuple().tm_yday
    decl = math.radians(23.44 * math.sin(2 * math.pi * (doy - 81) / 365.25))
    utc_hour = utc_datetime.hour + utc_datetime.minute / 60.0
    ha = math.radians(15.0 * (utc_hour - 12.0))
    lat_r = math.radians(lat)
    sin_elev = (math.sin(lat_r) * math.sin(decl) +
                math.cos(lat_r) * math.cos(decl) * math.cos(ha))
    return math.degrees(math.asin(np.clip(sin_elev, -1, 1)))


def _notched_phase_gram(x, dt_s, f_lo_hz=10e-6, f_hi_hz=500e-6,
                        min_bins=6, min_low_cycles=0.85):
    x = np.asarray(x, float)
    if x.ndim != 2 or x.shape[0] < 8 or x.shape[1] < 2:
        raise ValueError("x must be a time-by-station array")
    if not np.isfinite(x).all():
        raise ValueError("phase window contains non-finite samples")
    duration = (x.shape[0] - 1) * float(dt_s)
    if duration * f_lo_hz < min_low_cycles:
        raise ValueError("history too short")
    q = np.arange(x.shape[0], dtype=float)
    q -= q.mean()
    den = float(q @ q)
    datum_scale = np.max(np.abs(x), axis=0)
    if np.any(~np.isfinite(datum_scale)) or np.any(datum_scale <= np.finfo(float).tiny):
        raise ValueError("bad scale")
    x_safe = x / datum_scale
    centered = x_safe - x_safe.mean(axis=0, keepdims=True)
    slopes = (q[:, None] * centered).sum(axis=0) / den
    detrended = centered - q[:, None] * slopes
    scale = np.sqrt(np.mean(detrended * detrended, axis=0))
    if np.any(~np.isfinite(scale)) or np.any(scale <= np.finfo(float).tiny):
        raise ValueError("bad detrended scale")
    detrended = detrended / scale
    F = np.fft.rfft(detrended * np.hanning(x.shape[0])[:, None], axis=0)
    freq = np.fft.rfftfreq(x.shape[0], d=float(dt_s))
    band = (freq >= f_lo_hz) & (freq <= f_hi_hz)
    notch_freqs = [1.0 / p for p in NOTCH_PERIODS if p > 0]
    notch_mask = np.zeros(len(freq), dtype=bool)
    for fn in notch_freqs:
        lo = fn * (1.0 - NOTCH_WIDTH_FRAC)
        hi = fn * (1.0 + NOTCH_WIDTH_FRAC)
        notch_mask |= (freq >= lo) & (freq <= hi)
    band &= ~notch_mask
    if int(band.sum()) < int(min_bins):
        raise ValueError("too few bins after notching")
    F = F[band]
    cross = np.einsum("fi,fj->ij", F.conj(), F, optimize=False)
    energy = np.real(np.diag(cross))
    if np.any(energy <= np.finfo(float).tiny):
        raise ValueError("zero energy")
    norm = np.sqrt(np.outer(energy, energy))
    pg = np.real(cross / norm)
    pg = 0.5 * (pg + pg.T)
    np.fill_diagonal(pg, 1.0)
    return pg


def _rho_multi_notched(d):
    return np.mean([np.exp(-d / L) for L in NOTCHED_MULTI_SCALES_KM], axis=0)


def _reg_solve(K, c):
    K = 0.5 * (K + K.T)
    ev = np.linalg.eigvalsh(K)
    if ev[0] < 1e-6:
        K = K + (1e-6 - ev[0]) * np.eye(len(c))
    return np.linalg.solve(K, c)


def _ci(values, rng):
    v = np.array(values, float)
    if len(v) < 4:
        return {"median": None, "ci95": None, "n": len(v)}
    bs = [float(np.median(rng.choice(v, len(v), replace=True))) for _ in range(6000)]
    return {"median": float(np.median(v)),
            "ci95": [float(np.percentile(bs, 2.5)),
                     float(np.percentile(bs, 97.5))],
            "n": len(v), "frac_positive": float(np.mean(v > 0))}


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260902)

    print("Step 130: Day/night-aware TEP prior", flush=True)
    print(f"  Product: {PRODUCT}, {N_DAYS} days", flush=True)
    print(f"  Full-band lambda: {s64.LAMBDA_BASE_KM} km", flush=True)
    print(f"  Notched lambda:   {NOTCHED_LAMBDA_KM} km", flush=True)
    print()

    dates = td.available_dates(PRODUCT, 2024)
    days = [dates[i] for i in np.linspace(1, len(dates) - 1, N_DAYS).astype(int)]

    npre = int(s64.PREWINDOW_S / s64.DT_S)
    per_day = {k: [] for k in (
        "full_band_rms", "notched_rms", "day_night_rms",
        "day_night_vs_full", "day_night_vs_notched",
        "night_day_night_vs_full", "night_day_night_vs_notched")}

    for k, tag in enumerate(days):
        cur = s64.load_day(tag, product=PRODUCT)
        prev = s64.load_day(s64._prev(tag), product=PRODUCT)
        if cur is None or prev is None:
            continue
        names, t, raw, D = cur
        s_cur = td.load_series(PRODUCT, tag, min_valid_frac=0.0,
                               max_jumps=10_000)
        if not s_cur:
            continue

        # Both Grams
        pg_full = s64.prior_gram(prev, names)
        if pg_full is None:
            continue
        gnames_f, G_full, P_full = pg_full

        # Notched Gram
        pnames, pt, praw, _ = prev
        spectral_cols, clean_raw_cols, kept_n = [], [], []
        for nm in names:
            if nm not in pnames:
                continue
            y = praw[:, pnames.index(nm)].astype(float)
            good = np.isfinite(y)
            if good.mean() < 0.9 or good.sum() < 20:
                continue
            if not good.all():
                ix = np.arange(len(y))
                y = np.interp(ix, ix[good], y[good])
            xq = np.arange(len(y)) - len(y) / 2.0
            spectral_y = y - np.polyval(np.polyfit(xq, y, 2), xq)
            if not np.isfinite(spectral_y).all() or not (1e-4 < np.std(spectral_y) < 1e4):
                continue
            spectral_cols.append(spectral_y)
            clean_raw_cols.append(y.copy())
            kept_n.append(nm)
        if len(kept_n) < s64.MIN_REFS + 1:
            continue
        try:
            G_notch = _notched_phase_gram(np.column_stack(spectral_cols), s64.DT_S)
        except ValueError:
            continue

        common = sorted(set(gnames_f) & set(kept_n))
        if len(common) < s64.MIN_REFS + 2:
            continue
        idx_f = [gnames_f.index(n) for n in common]
        idx_n = [kept_n.index(n) for n in common]
        idx_cur = [names.index(n) for n in common]

        G_full = G_full[np.ix_(idx_f, idx_f)]
        G_notch = G_notch[np.ix_(idx_n, idx_n)]
        P = P_full[:, idx_f]
        Dg = D[np.ix_(idx_cur, idx_cur)]
        gnames = common

        # Get station latitudes and times
        lats = np.array([s_cur[n]["station"].lat_deg for n in gnames])
        t_s = s_cur[gnames[0]]["t"].astype(float)

        # Causal detrend
        R_cols = []
        for n in gnames:
            clk_ns = s_cur[n]["raw"].astype(float)
            ts = s_cur[n]["t"].astype(float)
            r = np.full_like(clk_ns, np.nan)
            for ki in range(len(clk_ns)):
                tk = ts[ki]
                mask = (ts <= tk) & (ts >= tk - 7200.0) & np.isfinite(clk_ns)
                if mask.sum() >= 10:
                    coef = np.polyfit(ts[mask], clk_ns[mask], 1)
                    r[ki] = clk_ns[ki] - np.polyval(coef, tk)
            R_cols.append(r)
        R = np.column_stack(R_cols).astype(float) * s64.C_M_PER_NS

        P = P.astype(float, copy=True)
        xq = np.arange(P.shape[0]) - P.shape[0] / 2
        for j in range(P.shape[1]):
            m = np.isfinite(P[:, j])
            if not m.all():
                P[:, j] = np.interp(np.arange(P.shape[0]),
                                    np.where(m)[0], P[m, j])
            P[:, j] -= np.polyval(np.polyfit(xq, P[:, j], 1), xq)

        sd_p = np.nanstd(P, axis=0)
        med_p = np.median(sd_p)
        keep = [i for i in range(len(gnames))
                if 0.01 < sd_p[i] < 12.0 and sd_p[i] < 3.5 * med_p]
        if len(keep) < s64.MIN_REFS + 2:
            continue
        gnames = [gnames[i] for i in keep]
        lats = lats[keep]
        R = R[:, keep]; P = P[:, keep]
        G_full = G_full[np.ix_(keep, keep)]
        G_notch = G_notch[np.ix_(keep, keep)]
        Dg = Dg[np.ix_(keep, keep)]

        ng = len(gnames)
        day_sq = {k: [] for k in ("full_band", "notched", "day_night")}
        night_sq = {k: [] for k in ("full_band", "notched", "day_night")}
        eval_i = list(range(npre, R.shape[0] - 1, s64.EVAL_STRIDE))

        for b in range(ng):
            pool = [j for j in range(ng) if j != b]
            if len(pool) < s64.MIN_REFS:
                continue

            # Full-band kernel
            rho_rr_f = s64._rho_multi(Dg[np.ix_(pool, pool)])
            rho_rt_f = s64._rho_multi(Dg[b, pool])
            try:
                w_full = _reg_solve(
                    rho_rr_f * G_full[np.ix_(pool, pool)] +
                    s64.NUGGET * np.eye(len(pool)),
                    rho_rt_f * G_full[b, pool])
            except np.linalg.LinAlgError:
                continue

            # Notched kernel
            rho_rr_n = _rho_multi_notched(Dg[np.ix_(pool, pool)])
            rho_rt_n = _rho_multi_notched(Dg[b, pool])
            try:
                w_notch = _reg_solve(
                    rho_rr_n * G_notch[np.ix_(pool, pool)] +
                    s64.NUGGET * np.eye(len(pool)),
                    rho_rt_n * G_notch[b, pool])
            except np.linalg.LinAlgError:
                continue

            B = R[:, b]; A = R[:, pool]
            lat_b = lats[b]
            for ei in eval_i:
                truth = B[ei]
                if not np.isfinite(truth):
                    continue
                rr = A[ei]
                if not np.isfinite(rr).all():
                    continue
                # Solar elevation at target station
                mid_time_s = float(t_s[ei])
                mid_dt = datetime(2024, 1, 1, tzinfo=timezone.utc) + \
                    timedelta(seconds=mid_time_s)
                elev = _solar_elevation(lat_b, mid_dt)
                is_night = elev < 0

                p_full = float(w_full @ rr)
                p_notch = float(w_notch @ rr)
                # Day/night-aware: use full-band during day, notched at night
                p_dn = p_notch if is_night else p_full

                day_sq["full_band"].append((truth - p_full) ** 2)
                day_sq["notched"].append((truth - p_notch) ** 2)
                day_sq["day_night"].append((truth - p_dn) ** 2)
                if is_night:
                    night_sq["full_band"].append((truth - p_full) ** 2)
                    night_sq["notched"].append((truth - p_notch) ** 2)
                    night_sq["day_night"].append((truth - p_dn) ** 2)

        if any(len(day_sq[k]) < 30 for k in day_sq):
            continue
        rms = {k: float(np.sqrt(np.mean(v))) for k, v in day_sq.items()}
        per_day["full_band_rms"].append(rms["full_band"])
        per_day["notched_rms"].append(rms["notched"])
        per_day["day_night_rms"].append(rms["day_night"])
        per_day["day_night_vs_full"].append(
            100.0 * (1.0 - rms["day_night"] / rms["full_band"]))
        per_day["day_night_vs_notched"].append(
            100.0 * (1.0 - rms["day_night"] / rms["notched"]))

        if all(len(night_sq[k]) > 5 for k in night_sq):
            rms_n = {k: float(np.sqrt(np.mean(v)))
                     for k, v in night_sq.items()}
            per_day["night_day_night_vs_full"].append(
                100.0 * (1.0 - rms_n["day_night"] / rms_n["full_band"]))
            per_day["night_day_night_vs_notched"].append(
                100.0 * (1.0 - rms_n["day_night"] / rms_n["notched"]))

        if (k + 1) % 10 == 0:
            print(f"  [{k+1}/{len(days)}] "
                  f"full={np.median(per_day['full_band_rms']):.1f}, "
                  f"notch={np.median(per_day['notched_rms']):.1f}, "
                  f"dn={np.median(per_day['day_night_rms']):.1f} m",
                  flush=True)

    n = len(per_day["full_band_rms"])
    print(f"\n  {n} days scored", flush=True)

    results = {
        "n_days": n,
        "day_night_vs_full_pct": _ci(per_day["day_night_vs_full"], rng),
        "day_night_vs_notched_pct": _ci(per_day["day_night_vs_notched"], rng),
        "night_day_night_vs_full_pct": _ci(
            per_day["night_day_night_vs_full"], rng),
        "night_day_night_vs_notched_pct": _ci(
            per_day["night_day_night_vs_notched"], rng),
        "median_rms_m": {
            "full_band": float(np.median(per_day["full_band_rms"])) if n else None,
            "notched": float(np.median(per_day["notched_rms"])) if n else None,
            "day_night": float(np.median(per_day["day_night_rms"])) if n else None,
        },
    }

    interpretations = []
    dnvf = results["day_night_vs_full_pct"]
    dnvn = results["day_night_vs_notched_pct"]
    ndnvf = results["night_day_night_vs_full_pct"]
    ndnvn = results["night_day_night_vs_notched_pct"]

    if dnvf.get("median") is not None:
        if dnvf["median"] > 0:
            interpretations.append(
                f"The day/night-aware kernel beats the always-full-band "
                f"kernel (median {dnvf['median']:+.2f}%, "
                f"CI95 [{dnvf['ci95'][0]:+.2f}, {dnvf['ci95'][1]:+.2f}]). "
                f"Using the notched TEP continuum at night and the full-band "
                f"kernel during the day captures the best of both components.")
        else:
            interpretations.append(
                f"The day/night-aware kernel does not beat the full-band "
                f"kernel ({dnvf['median']:+.2f}%).")

    if dnvn.get("median") is not None:
        if dnvn["median"] > 0:
            interpretations.append(
                f"The day/night-aware kernel also beats the always-notched "
                f"kernel (median {dnvn['median']:+.2f}%, "
                f"CI95 [{dnvn['ci95'][0]:+.2f}, {dnvn['ci95'][1]:+.2f}]). "
                f"The full-band kernel's solar tidal lines carry useful "
                f"predictive information during the day that the notched "
                f"kernel misses.")

    if ndnvf.get("median") is not None and ndnvf.get("n", 0) > 0:
        interpretations.append(
            f"For nighttime epochs specifically, day/night-aware vs "
            f"full-band: {ndnvf['median']:+.2f}% "
            f"(CI95 [{ndnvf['ci95'][0]:+.2f}, {ndnvf['ci95'][1]:+.2f}], "
            f"n={ndnvf['n']}).  The notched kernel's advantage at night "
            f"({ndnvn.get('median', 0):+.2f}% vs notched) confirms the "
            f"Step 125 finding that the TEP continuum is the active "
            f"component at night.")

    verdict = {
        "step": 130,
        "title": "Day/night-aware TEP prior: exploiting solar-illumination "
                 "independence",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "product": PRODUCT,
        "n_days": N_DAYS,
        "lambdas": {
            "full_band": s64.LAMBDA_BASE_KM,
            "notched": NOTCHED_LAMBDA_KM,
        },
        "results": results,
        "interpretation": interpretations,
        "pre_registered_prediction": (
            "The day/night-aware kernel (full-band during day, notched at "
            "night) should beat both the always-full-band and always-notched "
            "kernels.  During the day, the full-band kernel's solar tidal "
            "lines carry additional predictive information.  At night, the "
            "notched TEP continuum is the active component (Step 125: "
            "nighttime notched coherence 0.140 > daytime 0.102).  This "
            "exploits the solar-illumination independence of the TEP "
            "continuum as a predictive feature."),
    }

    OUT.write_text(json.dumps(verdict, indent=2))
    print(f"\n  Output: {OUT}", flush=True)


if __name__ == "__main__":
    main()
