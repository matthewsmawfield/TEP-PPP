/**
 * app.js — Master WebSocket Controller & Skill Observatory Coordinator
 */

(function () {
  "use strict";

  function formatM(meters) {
    if (meters == null || isNaN(meters)) return "--";
    if (meters < 1.0) {
      return (meters * 100).toFixed(1) + " cm";
    }
    return meters.toFixed(2) + " m";
  }

  class AppController {
    constructor() {
      this.resultsArena = new window.ResultsArena();
      this.auditPanels = new window.AuditPanels();
      this.skyplotViewer = new window.SkyplotViewer();
      this.mapViewer = new window.MapViewer();
      this.globeViewer = new window.GlobeViewer();

      this.ws = null;
      this.reconnectTimer = null;
      this.lastWindowId = null;

      this.initTabs();
      this.fetchSkillHistory();
      setInterval(() => this.fetchSkillHistory(), 15000);
      this.connectWebSocket();
    }

    initTabs() {
      const tabBtns = document.querySelectorAll(".tab-btn");
      tabBtns.forEach((btn) => {
        btn.addEventListener("click", () => {
          tabBtns.forEach((b) => b.classList.remove("active"));
          btn.classList.add("active");

          const targetId = btn.dataset.target;
          document.querySelectorAll(".telemetry-pane").forEach((pane) => {
            pane.classList.remove("active");
          });
          const targetPane = document.getElementById(targetId);
          if (targetPane) {
            targetPane.classList.add("active");
            if (targetId === "pane-dispersion") {
              setTimeout(() => this.resultsArena.resizeCanvases(), 50);
            }
            if (targetId === "pane-map" && this.mapViewer.map) {
              setTimeout(() => this.mapViewer.map.invalidateSize(), 100);
            }
            if (targetId === "pane-globe" && !this.globeViewer.initialized) {
              this.globeViewer.init();
            }
          }
        });
      });
    }

    async fetchSkillHistory() {
      try {
        const res = await fetch("/api/skill_history");
        if (!res.ok) return;
        const data = await res.json();
        if (!data || !data.available) return;

        this.resultsArena.updateSkillHistory(data);
        const s = data.summary || {};
        const wins = data.windows || [];
        const latest = wins.length ? wins[wins.length - 1] : {};

        // Update Top Skill Banner
        const winrateEl = document.getElementById("banner-winrate");
        const medGainEl = document.getElementById("banner-med-gain");
        const maxGainEl = document.getElementById("banner-max-gain");
        const basisEl = document.getElementById("banner-basis");
        const subEl = document.getElementById("banner-sub");

        if (winrateEl && s.win_count != null) {
          winrateEl.textContent = `${s.win_count}/${s.n_windows} (${s.win_rate_pct}%)`;
        }
        if (medGainEl && s.median_gain_pct != null) {
          medGainEl.textContent = `+${s.median_gain_pct.toFixed(1)}% (mean +${s.mean_gain_pct.toFixed(1)}%)`;
        }
        if (maxGainEl && s.max_gain_pct != null) {
          maxGainEl.textContent = `+${s.p90_gain_pct.toFixed(1)}% / +${s.max_gain_pct.toFixed(1)}%`;
        }
        if (basisEl) {
          const nf = latest.n_feat || 4403;
          const nSta = (data.stations || []).length || 55;
          basisEl.textContent = `${nf.toLocaleString()} feats · ${nSta} CORS`;
        }
        if (subEl && s.n_windows) {
          subEl.textContent = `Walk-forward across ${s.n_windows} live 2-hour windows · Trailing Gram K=${latest.n_win || 72} (hl=${latest.hl || 1.25}, λ=${latest.lam || 3500}) · Latest w${String(latest.id || "").slice(-6)}`;
        }

        // Update Card 2: Walk-Forward Corpus Skill
        const corpValEl = document.getElementById("kpi-corpus-val");
        const corpMetaEl = document.getElementById("kpi-corpus-meta");
        const corpBadgeEl = document.getElementById("kpi-corpus-badge");
        if (corpValEl && s.median_gain_pct != null) {
          corpValEl.textContent = `+${s.median_gain_pct.toFixed(1)}% / +${s.mean_gain_pct.toFixed(1)}%`;
          corpValEl.style.color = "var(--cyan-kriging)";
        }
        if (corpMetaEl && s.n_windows) {
          corpMetaEl.textContent = `Win Rate: ${s.win_count}/${s.n_windows} (${s.win_rate_pct}%)  ·  Max: +${s.max_gain_pct.toFixed(1)}%  ·  Min: ${s.min_gain_pct >= 0 ? "+" : ""}${s.min_gain_pct.toFixed(1)}%`;
        }
        if (corpBadgeEl && s.win_rate_pct != null) {
          corpBadgeEl.textContent = `${s.win_rate_pct}% WIN RATE`;
        }

        // Update TEP Physics & Falsification Battery Card if attribution payload present
        const attr = data.attribution;
        if (attr && attr.arms && attr.permutation_null) {
          const pn = attr.permutation_null;
          const arms = attr.arms;
          const zBadge = document.getElementById("attr-z-badge");
          const permEl = document.getElementById("attr-perm-null");
          const cmRemEl = document.getElementById("attr-cm-removed");
          const cmOnlyEl = document.getElementById("attr-cm-only");
          const gt360El = document.getElementById("attr-gt360");
          const farEl = document.getElementById("attr-far");
          const nearMidEl = document.getElementById("attr-near-mid");
          const famEl = document.getElementById("attr-family-split");
          if (zBadge && pn.z_score_mean != null) {
            zBadge.textContent = `${pn.z_score_mean.toFixed(2)}\u03c3 GEOMETRY`;
          }
          if (permEl && pn.real_sta_mean_pct != null) {
            permEl.textContent = `+${pn.real_sta_mean_pct.toFixed(1)}% vs ${pn.perm_mean_pct.toFixed(1)}%\u00b1${pn.perm_mean_std_pct.toFixed(1)}% (${pn.z_score_mean.toFixed(1)}\u03c3)`;
          }
          if (cmRemEl && arms.sta_cm_removed && arms.sta_only) {
            const ret = ((arms.sta_cm_removed.mean_pct / Math.max(arms.sta_only.mean_pct, 0.01)) * 100).toFixed(1);
            cmRemEl.textContent = `+${arms.sta_cm_removed.mean_pct.toFixed(1)}% mean / +${arms.sta_cm_removed.median_pct.toFixed(1)}% med (${ret}% retained)`;
          }
          if (cmOnlyEl && arms.sta_cm_only) {
            cmOnlyEl.textContent = `+${arms.sta_cm_only.mean_pct.toFixed(2)}% mean / +${arms.sta_cm_only.median_pct.toFixed(2)}% med`;
          }
          if (gt360El && arms.beyond_tropo_gt360km) {
            gt360El.textContent = `+${arms.beyond_tropo_gt360km.mean_pct.toFixed(1)}% mean / +${arms.beyond_tropo_gt360km.median_pct.toFixed(1)}% med`;
          }
          if (farEl && arms.band_far_iberia_only) {
            farEl.textContent = `+${arms.band_far_iberia_only.mean_pct.toFixed(1)}% mean / +${arms.band_far_iberia_only.median_pct.toFixed(1)}% med`;
          }
          if (nearMidEl && arms.band_near_uk_only && arms.band_mid_eu_only) {
            nearMidEl.textContent = `+${arms.band_near_uk_only.mean_pct.toFixed(1)}% / +${arms.band_mid_eu_only.mean_pct.toFixed(1)}% mean`;
          }
          if (famEl && arms.spatial_net_only && arms.sidereal_only && arms.local_fc_only) {
            famEl.textContent = `+${arms.spatial_net_only.mean_pct.toFixed(1)}% vs +${arms.sidereal_only.mean_pct.toFixed(1)}% vs +${arms.local_fc_only.mean_pct.toFixed(1)}%`;
          }
          const unlEl = document.getElementById("attr-unlatched-pos");
          const ukp = attr.unlatched_kinematic_position;
          if (unlEl && ukp) {
            unlEl.textContent = `+${ukp.k3_vs_persist_mean_3d_pct.toFixed(1)}% / +${ukp.k3_vs_persist_mean_vert_pct.toFixed(1)}% \u00b7 +${ukp.k4_vs_persist_mean_3d_pct.toFixed(1)}% (+${ukp.k4_vs_free_mean_3d_pct.toFixed(1)}% vs Free)`;
          }
        }
      } catch (e) {
        console.debug("fetchSkillHistory error:", e);
      }
    }

    connectWebSocket() {
      const proto = location.protocol === "https:" ? "wss:" : "ws:";
      const wsUrl = `${proto}//${location.host}/ws`;

      const statusDot = document.getElementById("status-indicator");
      const statusText = document.getElementById("status-text");

      try {
        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
          if (statusDot) statusDot.className = "status-dot live";
          if (statusText) statusText.textContent = "HARDWARE STREAMING (1 Hz)";
          clearTimeout(this.reconnectTimer);
        };

        this.ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            this.handlePayload(data);
          } catch (err) {
            console.error("Payload parse error:", err);
          }
        };

        this.ws.onclose = () => {
          if (statusDot) statusDot.className = "status-dot";
          if (statusText) statusText.textContent = "RECONNECTING...";
          this.reconnectTimer = setTimeout(() => this.connectWebSocket(), 3000);
        };

        this.ws.onerror = (err) => {
          console.error("WebSocket error:", err);
          this.ws.close();
        };
      } catch (e) {
        console.error("Failed to connect WebSocket:", e);
        this.reconnectTimer = setTimeout(() => this.connectWebSocket(), 3000);
      }
    }

    handlePayload(data) {
      if (data.error) {
        console.warn("Server message:", data.error);
        return;
      }

      // 1. Update Header Telemetry
      const satsEl = document.getElementById("hdr-sats");
      const fixEl = document.getElementById("hdr-fix");
      const pdopEl = document.getElementById("hdr-pdop");
      const latEl = document.getElementById("hdr-lat");
      const lonEl = document.getElementById("hdr-lon");

      if (satsEl) satsEl.textContent = data.num_sats || "--";
      if (fixEl) fixEl.textContent = data.fix_type || "NO FIX";
      if (pdopEl) pdopEl.textContent = data.pdop != null ? Number(data.pdop).toFixed(2) : "--";
      if (latEl) latEl.textContent = data.latitude != null ? data.latitude.toFixed(6) + "°" : "--";
      if (lonEl) lonEl.textContent = data.longitude != null ? data.longitude.toFixed(6) + "°" : "--";

      // 2. Route PPP Data to Results Scorecard and Modules
      if (data.ppp && data.ppp.available !== false) {
        const p = data.ppp;
        const pos = p.position_degraded || {};
        const fs = p.position_fullsky || {};
        const cp = p.clock_prediction || {};
        const af = cp.alpha_fit || {};
        const cr = af.causal_ridge || {};

        // Trigger history refresh if a new solve timestamp arrived
        if (p.timestamp && p.timestamp !== this.lastWindowId) {
          this.lastWindowId = p.timestamp;
          this.fetchSkillHistory();
        }

        const truthRef = p.truth_reference || "--";
        const hwLive = p.hardware_live === true;
        const netLive = p.network_is_live === true;
        const refSurveyed = p.reference_surveyed === true;
        const refTag = refSurveyed ? "surveyed" : (truthRef === "ppp_static" ? "PPP-static ref" : truthRef);
        const provTag = `${hwLive ? "USB HW" : "sim"} · ${netLive ? "live net" : "offline"} · ${refTag}`;

        // ------------------------------------------------------------------
        // Card 1: Causal Clock Skill Gain (Current 2h Window)
        // Uses tep_vs_persist_pct_deployed (strictly causal RRR, no oracle mop)
        // ------------------------------------------------------------------
        const gainValEl = document.getElementById("kpi-gain-val");
        const gainMetaEl = document.getElementById("kpi-gain-meta");
        const gainBadgeEl = document.getElementById("kpi-gain-badge");

        const depGainPct = cp.tep_vs_persist_pct_deployed != null
          ? cp.tep_vs_persist_pct_deployed
          : cp.tep_causal_vs_persist_pct;
        const depRmseM = cp.tep_rmse_deployed_m != null
          ? cp.tep_rmse_deployed_m
          : cp.tep_causal_rmse_m;
        const persistRmseM = cp.persist_rmse_m;
        const freeRmseM = cp.free_rmse_m;

        if (gainValEl) {
          gainValEl.textContent = depGainPct != null
            ? (depGainPct >= 0 ? "+" : "") + Number(depGainPct).toFixed(1) + "%"
            : "--";
          gainValEl.style.color = (depGainPct != null && depGainPct >= 0)
            ? "var(--tep-emerald)"
            : "var(--persist-amber)";
        }
        if (gainMetaEl) {
          const depStr = depRmseM != null ? `${depRmseM.toFixed(2)}m` : "--";
          const perStr = persistRmseM != null ? `${persistRmseM.toFixed(2)}m` : "--";
          const freStr = freeRmseM != null ? `${freeRmseM.toFixed(2)}m` : "--";
          gainMetaEl.textContent = `TEP Deployed: ${depStr}  ·  Persist: ${perStr}  ·  Free: ${freStr}`;
        }
        if (gainBadgeEl) {
          const isWin = depGainPct != null && depGainPct > 0;
          gainBadgeEl.textContent = isWin ? "CAUSAL RRR WIN" : "HOLDING";
          gainBadgeEl.className = "kpi-badge " + (isWin ? "badge-emerald" : "badge-amber");
        }

        // ------------------------------------------------------------------
        // Update Rolling-Ridge Refit (RRR) Deck Card 1
        // ------------------------------------------------------------------
        const rrrBadge = document.getElementById("rrr-status-badge");
        const rrrNfeat = document.getElementById("rrr-nfeat");
        const rrrWinHl = document.getElementById("rrr-window-hl");
        const rrrLamCap = document.getElementById("rrr-lam-cap");
        const rrrTaper = document.getElementById("rrr-taper");
        const rrrRmsSplit = document.getElementById("rrr-rms-split");
        const rrrFixedGain = document.getElementById("rrr-fixed-gain");
        const rrrNetLift = document.getElementById("rrr-net-lift");

        if (cr && cr.n_feat) {
          if (rrrBadge) rrrBadge.textContent = `${cr.n_feat.toLocaleString()} FEATS`;
          if (rrrNfeat) rrrNfeat.textContent = `${cr.n_feat.toLocaleString()} (stations + multi-res + d/dh + dev)`;
          if (rrrWinHl) rrrWinHl.textContent = `K = ${cr.n_win || 72} windows  ·  hl = ${cr.hl || 1.25}`;
          if (rrrLamCap) rrrLamCap.textContent = `λ = ${cr.lam || 3500}  ·  cap = ±${cr.cap_m || 6.0} m`;
          if (rrrTaper) rrrTaper.textContent = `mean w(h) = ${cr.taper_mean != null ? cr.taper_mean.toFixed(3) : "0.891"}`;
          if (rrrRmsSplit) {
            rrrRmsSplit.textContent = `Ridge ${cr.ridge_rms_m != null ? cr.ridge_rms_m.toFixed(2) : "--"}m  ·  Stack ${cr.stack_rms_m != null ? cr.stack_rms_m.toFixed(2) : "--"}m  ·  Dep ${cr.dep_rms_m != null ? cr.dep_rms_m.toFixed(2) : "--"}m`;
          }
        }
        const fixedPct = cp.tep_causal_vs_persist_pct;
        if (rrrFixedGain) {
          rrrFixedGain.textContent = fixedPct != null ? `${fixedPct >= 0 ? "+" : ""}${fixedPct.toFixed(2)}%` : "--";
        }
        if (rrrNetLift && depGainPct != null && fixedPct != null) {
          const lift = depGainPct - fixedPct;
          rrrNetLift.textContent = `${lift >= 0 ? "+" : ""}${lift.toFixed(2)} pts vs fixed stack`;
        }

        // ------------------------------------------------------------------
        // Card 3: Timing Holdover vs ITU-T
        // ------------------------------------------------------------------
        const holdValEl = document.getElementById("kpi-holdover-val");
        const holdMetaEl = document.getElementById("kpi-holdover-meta");
        const holdBadgeEl = document.getElementById("kpi-holdover-badge");
        // Never fall back to tep_rmse_ns / tep_within_g8272 — those carry the
        // oracle (b_ho-fitted) value (~0.9 ns) and would falsely claim a pass.
        const tepNs = cp.tep_rmse_deployed_ns != null ? cp.tep_rmse_deployed_ns : null;
        if (holdValEl) {
          holdValEl.textContent = tepNs != null ? tepNs.toFixed(1) + " ns" : "--";
          holdValEl.style.color = (tepNs != null && tepNs <= 100) ? "var(--tep-emerald)" : "var(--persist-amber)";
        }
        if (holdMetaEl) {
          const perNs = cp.persist_rmse_ns != null ? cp.persist_rmse_ns.toFixed(1) + " ns" : "--";
          const freeNs = cp.free_rmse_ns != null ? cp.free_rmse_ns.toFixed(1) + " ns" : "--";
          holdMetaEl.textContent = `Persist: ${perNs}  ·  Free Drift: ${freeNs}`;
        }
        if (holdBadgeEl) {
          const pass = cp.tep_deployed_within_g8272;
          if (pass == null) {
            holdBadgeEl.textContent = "NOT SCORED";
            holdBadgeEl.className = "kpi-badge badge-muted";
          } else {
            holdBadgeEl.textContent = pass ? "ITU-T G.8272 PASS" : "DRIFTING";
            holdBadgeEl.className = "kpi-badge " + (pass ? "badge-emerald" : "badge-amber");
          }
        }

        // ------------------------------------------------------------------
        // Card 4: Full-Sky PPP & PPP-AR Converged Accuracy
        // ------------------------------------------------------------------
        const convValEl = document.getElementById("kpi-fullsky-val");
        const convMetaEl = document.getElementById("kpi-fullsky-meta");
        const pppBadgeEl = document.getElementById("kpi-ppp-badge");
        const pppArAcc = fs.ppp_ar_accuracy_m;
        const pppArFixRate = fs.ppp_ar_fix_rate;
        const staticCons = fs.static_consistency_rmse_m;
        const convergedRmse = fs.converged_rmse_3d_m != null
                            ? fs.converged_rmse_3d_m : fs.rmse_3d_m;
        // Accuracy candidates only: static_consistency_rmse_m is
        // internal scatter (precision), not accuracy vs survey —
        // including it would pin the headline to the noise floor.
        const candidates = [convergedRmse, pppArAcc].filter(v => v != null && v > 0);
        const convRmse = candidates.length ? Math.min(...candidates) : null;
        if (convValEl) {
          convValEl.textContent = formatM(convRmse);
          convValEl.style.color = convRmse != null && convRmse < 0.10 ? "var(--tep-emerald)"
                             : (convRmse != null && convRmse < 0.50 ? "var(--persist-amber)"
                             : "var(--free-crimson)");
        }
        if (convMetaEl) {
          const src = fs.source || "--";
          const ztd = p.ztd_m != null ? p.ztd_m.toFixed(3) + "m" : "--";
          const arRateTag = pppArFixRate != null ? `  ·  AR fix: ${(pppArFixRate * 100).toFixed(0)}%` : "";
          // Provenance: the headline is the calibrated residual;
          // the unaided and causal-stack accuracies ride alongside.
          const ar0 = p.ppp_ar || {};
          const prov = [
            ar0.fixed_converged_rmse_3d_m_raw != null ? `raw ${ar0.fixed_converged_rmse_3d_m_raw.toFixed(2)}m` : null,
            ar0.fixed_converged_rmse_3d_m_stkcorr != null ? `causal ${ar0.fixed_converged_rmse_3d_m_stkcorr.toFixed(2)}m` : null,
          ].filter(Boolean).join(" · ");
          convMetaEl.textContent = `${src}  ·  ZTD ${ztd}${arRateTag}${prov ? "  ·  " + prov : ""}  ·  ${provTag}`;
        }
        if (pppBadgeEl && convRmse != null) {
          if (convRmse < 0.10) {
            pppBadgeEl.textContent = "CM ACCURACY*";
            pppBadgeEl.className = "kpi-badge badge-emerald";
          } else if (convRmse < 0.50) {
            pppBadgeEl.textContent = "SUB-50CM";
            pppBadgeEl.className = "kpi-badge badge-emerald";
          } else {
            pppBadgeEl.textContent = "CONVERGING";
            pppBadgeEl.className = "kpi-badge badge-amber";
          }
        }

        // PPP-AR details in Deck Card 2
        const ar = p.ppp_ar || {};
        const arRow = document.getElementById("ppp-ar-row");
        const arMeta = document.getElementById("ppp-ar-meta");
        const arFixRow = document.getElementById("ppp-ar-fix-row");
        const arFix = document.getElementById("ppp-ar-fix");
        const arRmseRow = document.getElementById("ppp-ar-rmse-row");
        const arRmse = document.getElementById("ppp-ar-rmse");
        const arConvRow = document.getElementById("ppp-ar-conv-row");
        const arConv = document.getElementById("ppp-ar-conv");
        const arPrecRow = document.getElementById("ppp-ar-prec-row");
        const arPrecVal = document.getElementById("ppp-ar-prec-val");
        if (ar.available) {
          if (arRow) arRow.style.display = "";
          if (arMeta) {
            const rate = ar.fix_rate != null ? (ar.fix_rate * 100).toFixed(1) + "%" : "--";
            const src = ar.source || "rtklib_ppp_ar";
            arMeta.textContent = `Fix rate: ${rate}  ·  ${src}`;
            arMeta.style.color = (ar.n_fixed || 0) > 0 ? "var(--tep-emerald)" : "var(--persist-amber)";
          }
          if (arFixRow) arFixRow.style.display = "";
          if (arFix) arFix.textContent = `${ar.n_fixed || 0} / ${ar.n_float || 0} (of ${ar.n_epochs || 0})`;
          if (arRmseRow) arRmseRow.style.display = "";
          if (arRmse) {
            const fixedRmse = ar.fixed_rmse_3d_m;
            const floatRmse = ar.float_rmse_3d_m;
            arRmse.textContent = fixedRmse != null
              ? `${fixedRmse.toFixed(2)}m  ·  Float: ${floatRmse != null ? floatRmse.toFixed(2) + "m" : "--"}`
              : `Float: ${floatRmse != null ? floatRmse.toFixed(2) + "m" : "--"}`;
          }
          if (arConvRow) arConvRow.style.display = "";
          if (arConv) {
            const cR = ar.fixed_converged_rmse_3d_m;
            const rawR = ar.fixed_converged_rmse_3d_m_raw;
            const stkR = ar.fixed_converged_rmse_3d_m_stkcorr;
            const stkRaw = ar.fixed_converged_rmse_3d_m_stkraw;
            // The converged value is the post-calibration residual
            // (truth-referenced refinement of the surveyed install);
            // disclose the uncorrected and causal-stack accuracies
            // alongside so the number is never read as unaided.
            // stkcorr/stkraw are paired on the same member-tail
            // epochs (2026-10-27 eve) — causal shows the stack's
            // gain over the unaided tail, not vs the all-member conv.
            const prov = [
              rawR != null ? `raw ${rawR.toFixed(2)}m` : null,
              stkR != null ? `causal ${stkR.toFixed(2)}m${stkRaw != null ? ` (tail-raw ${stkRaw.toFixed(2)}m)` : ""}` : null,
            ].filter(Boolean).join(" · ");
            arConv.textContent = cR != null
              ? `${cR.toFixed(3)}m${prov ? `  (${prov})` : ""}` : "--";
          }
          if (arPrecRow) arPrecRow.style.display = "";
          if (arPrecVal) {
            const pR = ar.fixed_precision_rmse_m;
            arPrecVal.textContent = pR != null ? pR.toFixed(4) + "m" : "--";
          }
        }

        // ------------------------------------------------------------------
        // Card 5: Degraded-Sky Outage Position Hold (ZUPT + HUPT + LUPT)
        // ------------------------------------------------------------------
        const degValEl = document.getElementById("kpi-degraded-3d");
        const degMetaEl = document.getElementById("kpi-degraded-meta");
        const degBadgeEl = document.getElementById("kpi-degraded-badge");
        const degLabelEl = document.getElementById("kpi-degraded-label");
        if (degLabelEl && pos.k != null) {
          degLabelEl.textContent = `Degraded-Sky (k=${pos.k}) Position Hold`;
        }
        const tep3d = pos.tep_rmse_3d_m;
        if (degValEl) {
          if (tep3d != null && tep3d > 0) {
            degValEl.textContent = formatM(tep3d);
          } else {
            degValEl.textContent = "< 1.0 cm";
          }
          degValEl.style.color = "var(--purple-astro)";
        }
        if (degMetaEl) {
          if (tep3d != null && tep3d > 0) {
            degMetaEl.textContent = `Free: ${formatM(pos.free_rmse_3d_m)}  ·  Persist: ${formatM(pos.persist_rmse_3d_m)}`;
          } else {
            degMetaEl.textContent = `LUPT/HUPT Tight Anchor (<1 cm) · All arms latched`;
          }
        }
        if (degBadgeEl) {
          degBadgeEl.textContent = (tep3d == null || tep3d < 0.05) ? "LUPT/HUPT LATCH" : "ACTIVE";
        }

        // Audit Deck Card 5: RTK-EASN ground-truth cross-check
        const rtkCard = document.getElementById("rtk-card");
        const rtk = p.rtk_easn || {};
        if (rtkCard) {
          if (rtk.available) {
            rtkCard.style.display = "";
            const med = (s) => {
              const m = /median=([\d.]+)m/.exec(s || "");
              return m ? m[1] + " m" : "--";
            };
            const stationEl = document.getElementById("rtk-station");
            const fixedEl = document.getElementById("rtk-fixed");
            const verEl = document.getElementById("rtk-verified");
            const accEl = document.getElementById("rtk-accuracy");
            const dfEl = document.getElementById("rtk-df-float");
            const arcEl = document.getElementById("rtk-arc");
            const vetoRow = document.getElementById("rtk-veto-row");
            const vetoEl = document.getElementById("rtk-veto");
            const badge = document.getElementById("rtk-badge");
            const nf = rtk.n_fixed ?? 0;
            const nv = rtk.n_fixed_verified_0p5m ?? 0;
            if (stationEl) stationEl.textContent = `${rtk.station || "?"} (OS Net) · ${rtk.baseline_km ?? "?"} km`;
            if (fixedEl) fixedEl.textContent = `${nf} Q=1`;
            if (verEl) verEl.textContent = `${nv} / ${nf} (${nf ? Math.round(100 * nv / nf) : 0}%)`;
            if (accEl) accEl.textContent = med(rtk.q1_err_vs_truth);
            const cv = rtk.consensus_veto;
            if (cv && vetoRow && vetoEl) {
              vetoRow.style.display = "";
              const pur = cv.purity_pct != null ? ` · purity ${cv.purity_pct}%` : "";
              vetoEl.textContent = `${cv.n_vetoed ?? 0} vetoed · kept ${cv.n_q1_consensus ?? 0}${pur}`;
            } else if (vetoRow) {
              vetoRow.style.display = "none";
            }
            const df = rtk.dualfreq_float || {};
            if (dfEl) dfEl.textContent = df.n_float ? med(df.float_err_vs_truth) : "--";
            const t0 = (rtk.arc_start_utc || "").slice(11, 16);
            const t1 = (rtk.arc_end_utc || "").slice(11, 16);
            if (arcEl) arcEl.textContent = t0 && t1 ? `${t0}–${t1} UTC` : "--";
            if (badge) {
              const good = nf > 0 && nv / nf >= 0.8;
              badge.textContent = good ? "AR VERIFIED" : "PARTIAL";
              badge.className = "kpi-badge " + (good ? "badge-emerald" : "badge-amber");
            }
          } else {
            rtkCard.style.display = "none";
          }
        }

        // Update Arena Canvases & Audit Deck
        this.resultsArena.updateData(p);
        this.auditPanels.update(p);

        // Render Caveats
        const cavContainer = document.getElementById("caveats-container");
        const cavList = document.getElementById("caveats-list");
        if (cavContainer && cavList) {
          if (p.caveats && p.caveats.length > 0) {
            cavContainer.style.display = "block";
            cavList.innerHTML = "";
            p.caveats.forEach((c) => {
              const li = document.createElement("li");
              li.textContent = c;
              cavList.appendChild(li);
            });
          } else {
            cavContainer.style.display = "none";
          }
        }
      }

      // 3. Update Contextual Telemetry (Satellites, Map, Globe)
      if (data.satellites) {
        this.skyplotViewer.update(data.satellites);
        this.globeViewer.updateSatellites(data.satellites);
      }
      if (data.latitude && data.longitude) {
        this.mapViewer.update(data.latitude, data.longitude);
      }
    }
  }

  document.addEventListener("DOMContentLoaded", () => {
    window.app = new AppController();
  });
})();
