"""Step 177 -- Distance-blind geography recovery from clock correlations alone.

Question: does the pairwise phase-alignment structure encode physical
geography?  If the field correlation is genuinely spatial, the station x
station correlation matrix alone -- with coordinates withheld -- must let a
blind dimensionality reduction (classical MDS) reconstruct the network map.
A processing systematic tied to anything but distance cannot do this.

Protocol (identical estimator to Step 53, Papers 1-2 phase alignment):
  1. Load N_SPAN days of the CODE precise-clock product (clk_cod), evenly
     spread across 2024 -- the same product and band as Step 53.
  2. Per station per day, compute the 10-500 uHz complex spectrum
     (scripts/steps/step_53_coherent_lambda.py::_spectra, imported so the
     estimator is literally the same code path).
  3. Accumulate the pair phasor S_ij = sum_days sum_f conj(F_i) F_j across
     all days; the pairwise similarity is C_ij = cos(arg S_ij).
  4. Restrict to the largest set of stations with complete pairwise
     coverage (greedy drop of the station with most missing pairs).
  5. Classical MDS on dissimilarity D = 1 - C -> 2-D embedding.
  6. Score: Pearson r and Spearman rho between MDS pairwise distances and
     true great-circle distances (Mantel-style statistic); Procrustes
     alignment to the true map -> median position error in km.
  7. Nulls:
       a. per-station Fourier phase randomisation -> identical pipeline
          (n_null realisations),
       b. station-label permutation of the measured C matrix -> destroys
          the correlation-to-distance correspondence while preserving the
          marginal distribution of C (n_null realisations).

Outputs: results/outputs/step_177_distance_blind_geography.json
         results/figures/step_177_distance_blind_geography.png
"""

import itertools
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "steps"))

import scripts.lib.tep_data as td  # noqa: E402
from step_53_coherent_lambda import _spectra  # noqa: E402

PRODUCT = "clk_cod"
YEAR = 2024
N_DAYS = 30
MIN_SHARED_DAYS_FRAC = 0.8   # a pair must be observed on >= this fraction of days
SUBNET_DIAMETER_KM = 5500.0  # below the correlation-saturation range
N_NULL = 40
OUT_JSON = PROJECT_ROOT / "results" / "outputs" / "step_177_distance_blind_geography.json"
OUT_FIG = PROJECT_ROOT / "results" / "figures" / "step_177_distance_blind_geography.png"


def _accumulate_phasors(days_RF):
    """pair -> (complex phasor sum, n_days observed)."""
    acc = {}
    for RF in days_RF:
        nm = sorted(RF)
        for i, j in itertools.combinations(nm, 2):
            Fi, Fj = RF[i], RF[j]
            if Fi.shape != Fj.shape:
                continue
            S = np.sum(np.conj(Fi) * Fj)
            if not np.isfinite(S):
                continue
            key = (i, j)
            if key not in acc:
                acc[key] = [0j, 0]
            acc[key][0] += S
            acc[key][1] += 1
    return acc


def _similarity_matrix(acc, n_days):
    """Station list + complete C matrix over the largest fully-covered set."""
    stations = sorted({s for pair in acc for s in pair})
    idx = {s: k for k, s in enumerate(stations)}
    n = len(stations)
    Csum = np.zeros((n, n))
    Ccnt = np.zeros((n, n), int)
    for (i, j), (S, cnt) in acc.items():
        Csum[idx[i], idx[j]] = Csum[idx[j], idx[i]] = np.cos(np.angle(S))
        Ccnt[idx[i], idx[j]] = Ccnt[idx[j], idx[i]] = cnt

    min_days = max(1, int(round(MIN_SHARED_DAYS_FRAC * n_days)))
    keep = list(range(n))
    while True:
        miss = {k: int(np.sum((Ccnt[k, keep] < min_days))) for k in keep}
        worst = max(miss, key=miss.get)
        if miss[worst] <= 1:  # diagonal counts 0; allow only the self-pair
            break
        keep.remove(worst)
    keep = np.array(sorted(keep))
    C = Csum[np.ix_(keep, keep)]
    np.fill_diagonal(C, 1.0)
    return [stations[k] for k in keep], C


def _classical_mds(D, n_dim=2):
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    w, V = np.linalg.eigh(B)
    order = np.argsort(w)[::-1]
    w = np.clip(w[order][:n_dim], 0, None)
    return V[:, order][:, :n_dim] * np.sqrt(w)


def _pairwise_dists(X):
    n = X.shape[0]
    iu = np.triu_indices(n, 1)
    return np.linalg.norm(X[iu[0]] - X[iu[1]], axis=1), iu


def _regional_subnet(names, coords_true, max_diameter_km):
    """Largest station subset with all pairwise distances < max_diameter_km.

    Greedy: repeatedly drop the station that participates in the most
    over-threshold pairs until the subgraph's diameter is below threshold.
    The correlation profile saturates at zero beyond the field scale
    (~6,000-7,000 km), so a blind embedding can only be informative inside
    the regime where C(d) still varies with distance.
    """
    keep = list(names)
    while len(keep) > 2:
        bad = {s: 0 for s in keep}
        for i, j in itertools.combinations(keep, 2):
            if td.great_circle_km(coords_true[i], coords_true[j]) >= max_diameter_km:
                bad[i] += 1
                bad[j] += 1
        worst = max(bad, key=bad.get)
        if bad[worst] == 0:
            break
        keep.remove(worst)
    return keep


def _procrustes(X, Y):
    """Similarity alignment X -> Y. Returns aligned X and per-point error."""
    Xc = X - X.mean(0)
    Yc = Y - Y.mean(0)
    M = Xc.T @ Yc
    U, s, Vt = np.linalg.svd(M)
    R = U @ Vt
    scale = s.sum() / (Xc ** 2).sum()
    Xa = scale * (Xc @ R) + Y.mean(0)
    return Xa, np.linalg.norm(Xa - Y, axis=1), scale


def _score(C, coords_true, station_names):
    """MDS the similarity matrix, score vs true geography."""
    D = 1.0 - C
    np.fill_diagonal(D, 0.0)
    X = _classical_mds(D)
    d_mds, iu = _pairwise_dists(X)
    iu_arr = np.array(iu)
    la1 = np.array([coords_true[s].lat_deg for s in station_names])
    lo1 = np.array([coords_true[s].lon_deg for s in station_names])
    d_true = np.array([
        td.great_circle_km(coords_true[station_names[i]],
                           coords_true[station_names[j]])
        for i, j in zip(*iu)])
    r = float(np.corrcoef(d_mds, d_true)[0, 1])
    rho = float(spearmanr(d_mds, d_true).statistic)

    # Procrustes position error (km) on a local tangent-plane projection.
    lat0 = np.radians(la1.mean())
    Y = np.column_stack([
        np.radians(lo1) * np.cos(lat0) * td.EARTH_RADIUS_KM,
        np.radians(la1) * td.EARTH_RADIUS_KM])
    Xa, errs, scale = _procrustes(X, Y)
    return {"pearson_r": r, "spearman_rho": rho,
            "median_position_error_km": float(np.median(errs)),
            "X_mds": X, "Y_true": Y, "X_aligned": Xa,
            "d_mds": d_mds, "d_true": d_true}


def _fig(score, station_names, null_phase_r, null_perm_r):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scripts.utils.plotting import setup_publication_style, THEME_COLORS
    setup_publication_style()
    C = THEME_COLORS

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6))

    ax = axes[0]
    ax.scatter(score["Y_true"][:, 0], score["Y_true"][:, 1],
               s=26, color=C["secondary"], edgecolors="none")
    ax.set_title("(a) True station geography (withheld)")
    ax.set_xlabel("east (km)")
    ax.set_ylabel("north (km)")

    ax = axes[1]
    ax.scatter(score["X_aligned"][:, 0], score["X_aligned"][:, 1],
               s=26, color=C["primary"], edgecolors="none")
    ax.set_title(f"(b) Map recovered blind from clock correlations "
                 f"(median error {score['median_position_error_km']:.0f} km)")
    ax.set_xlabel("east (km)")
    ax.set_ylabel("north (km)")

    ax = axes[2]
    # MDS distances live in arbitrary embedding units; rescale to the true
    # distance scale (least-squares slope through the origin) so the
    # scatter is readable in km.
    slope = float(score["d_mds"] @ score["d_true"] / (score["d_mds"] ** 2).sum())
    d_mds_km = score["d_mds"] * slope
    lo = 0.0
    hi = max(score["d_true"].max(), d_mds_km.max())
    ax.plot([lo, hi], [lo, hi], color=C["border"], lw=1.0, ls="--")
    ax.scatter(score["d_true"], d_mds_km, s=8,
               color=C["accent"], alpha=0.35, edgecolors="none")
    ax.set_title(f"(c) Pairwise distance: recovered vs true "
                 f"(r = {score['pearson_r']:.3f})")
    ax.set_xlabel("true great-circle distance (km)")
    ax.set_ylabel("MDS-recovered distance")

    for ax in axes:
        ax.set_aspect("equal" if ax is not axes[2] else "auto")

    fig.suptitle("Distance-blind geography recovery: the correlation "
                 "matrix encodes the map", fontsize=13,
                 color=THEME_COLORS["primary"])
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=160)
    plt.close(fig)


def main():
    dates = td.available_dates(PRODUCT, YEAR)
    idx = np.linspace(0, len(dates) - 1, N_DAYS).round().astype(int)
    tags = [dates[i] for i in idx]

    days = [s for _, s in td.load_multi_day(PRODUCT, tags) if len(s) >= 8]
    days_RF, days_STA = [], {}
    for s in days:
        RF, STA = _spectra(s)
        if len(RF) >= 8:
            days_RF.append(RF)
            days_STA.update(STA)
    n_days = len(days_RF)

    acc = _accumulate_phasors(days_RF)
    names, C = _similarity_matrix(acc, n_days)
    score = _score(C, days_STA, names)

    # Regional arm: the correlation profile saturates at zero beyond the
    # field scale, so a blind embedding is only informative where C(d)
    # still varies.  Restrict to the largest sub-network with diameter
    # below the saturation range.
    sub_names = _regional_subnet(names, days_STA, SUBNET_DIAMETER_KM)
    sub_idx = [names.index(s) for s in sub_names]
    C_sub = C[np.ix_(sub_idx, sub_idx)]
    score_sub = _score(C_sub, days_STA, sub_names)

    # Null a: per-station Fourier phase randomisation, identical pipeline.
    # The same subnet selection is re-derived inside each null realisation
    # so the comparison arm matches the real one.
    null_phase_r, null_phase_rho = [], []
    null_phase_r_sub, null_phase_rho_sub = [], []
    for k in range(N_NULL):
        rng = np.random.default_rng(190_000 + k)
        RF_null = []
        for s in days:
            RFn, _ = _spectra(s, phase_rng=rng)
            if len(RFn) >= 8:
                RF_null.append(RFn)
        accn = _accumulate_phasors(RF_null)
        names_n, Cn = _similarity_matrix(accn, len(RF_null))
        sc = _score(Cn, days_STA, names_n)
        null_phase_r.append(sc["pearson_r"])
        null_phase_rho.append(sc["spearman_rho"])
        sub_n = _regional_subnet(names_n, days_STA, SUBNET_DIAMETER_KM)
        if len(sub_n) >= 8:
            si = [names_n.index(s) for s in sub_n]
            sc_sub = _score(Cn[np.ix_(si, si)], days_STA, sub_n)
            null_phase_r_sub.append(sc_sub["pearson_r"])
            null_phase_rho_sub.append(sc_sub["spearman_rho"])

    # Null b: station-label permutation -- the measured C matrix is kept
    # fixed and the geographic labels are shuffled, so each row of C is
    # scored against the wrong station's true distances.  This preserves
    # the marginal distribution of C while destroying the correlation-to-
    # distance correspondence.  (Permuting rows and columns of C together
    # with the name list would be an identity operation.)
    rng = np.random.default_rng(200_000)
    null_perm_r, null_perm_r_sub = [], []
    for _ in range(N_NULL):
        p = rng.permutation(len(names))
        sc = _score(C, days_STA, [names[q] for q in p])
        null_perm_r.append(sc["pearson_r"])
        ps = rng.permutation(len(sub_names))
        scs = _score(C_sub, days_STA, [sub_names[q] for q in ps])
        null_perm_r_sub.append(scs["pearson_r"])

    def _sig(real, nulls):
        a = np.asarray(nulls, float)
        sd = float(a.std())
        return {"null_mean": float(a.mean()), "null_std": sd,
                "n_null": int(len(a)),
                "n_exceeding": int(np.sum(a >= real)),
                "p_value": float((np.sum(a >= real) + 1) / (len(a) + 1)),
                "sigma": (float((real - a.mean()) / sd) if sd > 1e-9 else None)}

    res = {
        "test": "distance_blind_geography_recovery",
        "question": ("does the pairwise phase-alignment matrix alone "
                     "reconstruct station geography with coordinates "
                     "withheld?"),
        "statistic": ("classical MDS on D = 1 - cos(arg S_ij); Mantel "
                      "correlation of recovered vs true pairwise distances; "
                      "Procrustes median position error"),
        "estimator": ("identical to Step 53: cos(arg(sum_f S_ij(f))), "
                      "10-500 uHz, linear detrend only"),
        "product": PRODUCT, "year": YEAR, "n_days": n_days,
        "date_tags": tags, "n_stations": len(names),
        "stations": names,
        "timestamp": datetime.now().isoformat(),
        "pearson_r": score["pearson_r"],
        "spearman_rho": score["spearman_rho"],
        "median_position_error_km": score["median_position_error_km"],
        "global_arm_note": ("global embedding is uninformative by "
                            "construction: C(d) saturates at zero beyond "
                            "the field scale, so all far pairs are "
                            "equidistant in correlation space; the "
                            "informative arm is the regional subnet"),
        "subnet": {
            "max_diameter_km": SUBNET_DIAMETER_KM,
            "n_stations": len(sub_names),
            "stations": sub_names,
            "pearson_r": score_sub["pearson_r"],
            "spearman_rho": score_sub["spearman_rho"],
            "median_position_error_km": score_sub["median_position_error_km"],
            "fig_data": {
                "d_true_km": score_sub["d_true"].tolist(),
                "d_mds": score_sub["d_mds"].tolist(),
                "Y_true_km": score_sub["Y_true"].tolist(),
                "X_aligned": score_sub["X_aligned"].tolist(),
            },
            "null_phase_randomise": _sig(score_sub["pearson_r"],
                                         null_phase_r_sub),
            "null_phase_randomise_spearman": _sig(
                score_sub["spearman_rho"], null_phase_rho_sub),
            "null_label_permutation": _sig(score_sub["pearson_r"],
                                           null_perm_r_sub),
        },
        "null_phase_randomise": _sig(score["pearson_r"], null_phase_r),
        "null_phase_randomise_spearman": _sig(score["spearman_rho"],
                                              null_phase_rho),
        "null_label_permutation": _sig(score["pearson_r"], null_perm_r),
        "figure": "results/figures/step_177_distance_blind_geography.png",
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(__import__("json").dumps(res, indent=2))
    _fig(score_sub, sub_names, null_phase_r_sub, null_perm_r_sub)

    print(f"n_days={n_days} n_stations={len(names)}")
    print(f"global: pearson r = {score['pearson_r']:.3f}")
    print(f"subnet ({len(sub_names)} stations): "
          f"pearson r = {score_sub['pearson_r']:.3f}  "
          f"spearman = {score_sub['spearman_rho']:.3f}  "
          f"median error = {score_sub['median_position_error_km']:.0f} km")
    print("subnet phase null:",
          _sig(score_sub["pearson_r"], null_phase_r_sub))
    print(f"-> {OUT_JSON}\n-> {OUT_FIG}")


if __name__ == "__main__":
    main()
