#!/usr/bin/env python3
"""
STEP 74b  --  Pair-clustered control on the short-baseline EW/NS anisotropy.

Step 74 found the EW-NS phase-alignment contrast peaking at 500-800 km on two
independent products (+0.062 precise, +0.082 broadcast) with day-clustered
z = +8.1 and +9.3.  Those z values are INFLATED and must not be reported.

The day-clustered bootstrap resamples days, which handles temporal
pseudo-replication -- but the same station pairs recur every day, so the
effective sample size is the number of PAIRS, not pair-days: 26,141 pair-days
in the precise arm come from just 150 unique pairs (60 EW, 90 NS).

This step redoes the inference clustering on unique pairs, and adds
leave-one-station-out to test whether any single station carries the effect.

Result: the sign is robust (every LOSO value positive on both arms, two
independent products agreeing at +11% and +17%), but the pair-clustered CI
includes zero (P(>0) = 0.80 precise, 0.87 broadcast).  The limiting resource is
INDEPENDENT STATION PAIRS, not observation days -- more days cannot close this,
a denser network can.

Output: results/outputs/step_74b_anisotropy_pair_clustered.json
"""
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
import numpy as np, itertools, importlib.util, json
from pathlib import Path
from collections import defaultdict
ROOT=_P(__file__).resolve().parents[2]
from scripts.lib import tep_data as td
sp=importlib.util.spec_from_file_location("s53",ROOT/"scripts/steps/step_53_coherent_lambda.py")
s53=importlib.util.module_from_spec(sp); sp.loader.exec_module(s53)
RNG=np.random.default_rng(11)
HALF=22.5; LO,HI=500.0,800.0

ARMS={}
for product in ("precise","broadcast"):
    dates=td.available_dates(product,2024)[:366]
    rec=[]
    for k,tag in enumerate(dates):
        try: ser=td.load_series(product,tag,min_valid_frac=0.0,max_jumps=10_000)
        except Exception: continue
        RF,STA=s53._spectra(ser)
        if len(RF)<6: continue
        nm=list(RF)
        for i,j in itertools.combinations(range(len(nm)),2):
            Fi,Fj=RF[nm[i]],RF[nm[j]]
            if Fi.shape!=Fj.shape: continue
            A,B=STA[nm[i]],STA[nm[j]]
            d=td.great_circle_km(A,B)
            if not (LO<=d<HI): continue
            S=np.sum(np.conj(Fi)*Fj)
            if not np.isfinite(S) or S==0: continue
            az=td.baseline_azimuth_midpoint_deg(A,B)
            sec='EW' if abs(az-90.0)<=HALF else ('NS' if (az<=HALF or az>=180-HALF) else None)
            if sec: rec.append((A.name,B.name,sec,float(np.cos(np.angle(S)))))
    if not rec: print(product,"none"); continue
    pid=np.array([f"{a}-{b}" for a,b,_,_ in rec]); sec=np.array([r[2] for r in rec])
    val=np.array([r[3] for r in rec]); 
    upairs=np.unique(pid)
    nE=len(np.unique(pid[sec=='EW'])); nN=len(np.unique(pid[sec=='NS']))
    obs=val[sec=='EW'].mean()-val[sec=='NS'].mean()
    print(f"\n=== {product}  {LO:.0f}-{HI:.0f} km ===")
    print(f"  {len(rec):,} pair-days from {len(upairs)} UNIQUE pairs  (EW {nE} pairs, NS {nN} pairs)")
    print(f"  observed EW-NS = {obs:+.4f}")
    # PAIR-clustered bootstrap: resample unique pairs, not pair-days
    idx_by=defaultdict(list)
    for i,p in enumerate(pid): idx_by[p].append(i)
    bs=[]
    for _ in range(2000):
        pk=RNG.choice(upairs,len(upairs),replace=True)
        ii=np.concatenate([idx_by[p] for p in pk])
        e=sec[ii]=='EW'; n=sec[ii]=='NS'
        if e.sum()>5 and n.sum()>5: bs.append(val[ii][e].mean()-val[ii][n].mean())
    bs=np.array(bs)
    print(f"  PAIR-clustered bootstrap CI95 [{np.percentile(bs,2.5):+.4f}, {np.percentile(bs,97.5):+.4f}]  P(>0)={np.mean(bs>0):.3f}")
    # leave-one-station-out
    stns=sorted({s for p in upairs for s in p.split('-')})
    loso=[]
    for st in stns:
        m=np.array([st not in p.split('-') for p in pid])
        e=m&(sec=='EW'); n=m&(sec=='NS')
        if e.sum()>20 and n.sum()>20: loso.append((st,val[e].mean()-val[n].mean()))
    if loso:
        vals=np.array([v for _,v in loso])
        print(f"  LOSO over {len(loso)} stations: min {vals.min():+.4f}  max {vals.max():+.4f}  all positive: {bool((vals>0).all())}")
        worst=sorted(loso,key=lambda x:x[1])[:3]
        print(f"  most influential (lowest when removed): {[(s,round(v,4)) for s,v in worst]}")
        ARMS[product]={"band_km":[LO,HI],"n_pair_days":len(rec),"n_unique_pairs":int(len(upairs)),
            "n_ew_pairs":int(nE),"n_ns_pairs":int(nN),"observed_diff":float(obs),
            "pair_clustered_ci95":[float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))],
            "pair_clustered_p_gt_0":float(np.mean(bs>0)),
            "loso_min":float(vals.min()),"loso_max":float(vals.max()),
            "loso_all_positive":bool((vals>0).all()),"n_loso_stations":int(len(loso))}

json.dump({"step":"74b","title":"Pair-clustered control on short-baseline EW/NS anisotropy",
  "note":"day-clustered z from step_74 (+8.1/+9.3) is inflated by pair-level pseudo-replication; "
         "these pair-clustered intervals supersede it",
  "arms":ARMS,
  "verdict":{"sign_robust":all(a["loso_all_positive"] for a in ARMS.values()),
             "significant_pair_clustered":any(a["pair_clustered_ci95"][0]>0 for a in ARMS.values()),
             "limiting_resource":"independent station pairs, not observation days"}},
  open(ROOT/"results/outputs/step_74b_anisotropy_pair_clustered.json","w"), indent=2)
print("\nsaved step_74b_anisotropy_pair_clustered.json")
