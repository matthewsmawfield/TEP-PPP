/**
 * audit_panels.js — Metrological Audit & Scientific Verification Panels
 * Handles:
 *   - Full-Sky PPP cm-level Convergence Engine & Curve
 *   - Network Kriging Covariance & Spatial Correlation \rho(d)
 *   - Astronomical Phase Modulation (Chandler, Nodal, Semiannual)
 *   - Provenance Ledger & ITU-T Holdover Verification
 */

(function () {
  "use strict";

  class AuditPanels {
    constructor() {
      this.convCanvas = document.getElementById("convergence-canvas");
      this.convCtx = this.convCanvas ? this.convCanvas.getContext("2d") : null;
      if (this.convCanvas) {
        this.resizeConvergenceCanvas();
        window.addEventListener("resize", () => this.resizeConvergenceCanvas());
      }
    }

    resizeConvergenceCanvas() {
      if (!this.convCanvas) return;
      const dpr = window.devicePixelRatio || 1;
      const rect = this.convCanvas.parentElement.getBoundingClientRect();
      const w = rect.width;
      const h = Math.max(160, rect.height || 160);
      this.convCanvas.width = w * dpr;
      this.convCanvas.height = h * dpr;
      this.convCanvas.style.width = w + "px";
      this.convCanvas.style.height = h + "px";
      this.convCtx.resetTransform();
      this.convCtx.scale(dpr, dpr);
    }

    update(pppData) {
      if (!pppData) return;
      this.updateFullSky(pppData.position_fullsky, pppData.arms);
      this.updateKriging(pppData.network_geometry, pppData.clock_prediction, pppData.prior_source);
      this.updateAstronomical(pppData.tep_astronomical, pppData.arms);
      this.updateProvenance(pppData);
      this.updateMultiband(pppData.multiband);
    }

    // ------------------------------------------------------------------------
    // Full-Sky Convergence & Precision
    // ------------------------------------------------------------------------
    updateFullSky(fs, arms) {
      if (!fs) return;
      // Priority: PPP-AR accuracy (vs truth) > static consistency
      // (precision) > converged RMSE.  The PPP-AR accuracy is the
      // cm-level headline: accuracy against a surveyed truth reference.
      const pppArAcc = fs.ppp_ar_accuracy_m;
      const staticCons = fs.static_consistency_rmse_m;
      const convRmse = pppArAcc != null ? pppArAcc
                     : (staticCons != null ? staticCons
                     : (fs.converged_rmse_3d_m != null ? fs.converged_rmse_3d_m : fs.rmse_3d_m));
      const fsEl = document.getElementById("fullsky-converged-val");
      const fsSub = document.getElementById("fullsky-meta");

      if (fsEl) {
        if (convRmse != null) {
          fsEl.textContent = convRmse < 1.0 ? (convRmse * 100).toFixed(1) + " cm" : convRmse.toFixed(2) + " m";
          fsEl.style.color = convRmse < 0.10 ? "var(--tep-emerald)" : (convRmse < 0.50 ? "var(--persist-amber)" : "var(--free-crimson)");
        } else {
          fsEl.textContent = "--";
        }
      }

      if (fsSub) {
        const src = fs.source || "--";
        const kinRmse = fs.kinematic_rmse_3d_m;
        const kinTag = kinRmse != null ? `  ·  Kin cross-check: ${kinRmse.toFixed(1)}m` : "";
        const arTag = pppArAcc != null && fs.ppp_ar_fix_rate != null
          ? `  ·  AR fix: ${(fs.ppp_ar_fix_rate * 100).toFixed(0)}%`
          : "";
        fsSub.textContent = `${src}  ·  Epochs: ${fs.n_epochs || 0}  ·  Phase Obs: ${fs.ambiguity_phase_obs || 0}  ·  Slips: ${fs.ambiguity_slips || 0}${kinTag}${arTag}`;
      }

      // Draw convergence curve if series is present
      if (fs.convergence_series && fs.convergence_series.length > 1) {
        this.renderConvergenceCurve(fs.convergence_series, convRmse);
      }
    }

    renderConvergenceCurve(series, convergedTarget) {
      const ctx = this.convCtx;
      if (!ctx) return;
      const cvs = this.convCanvas;
      const w = parseFloat(cvs.style.width);
      const h = parseFloat(cvs.style.height);

      ctx.clearRect(0, 0, w, h);

      const padL = 45, padR = 15, padT = 15, padB = 25;
      const plotW = w - padL - padR;
      const plotH = h - padT - padB;

      let maxErr = 1.0;
      series.forEach((pt) => {
        if (pt.err_m > maxErr && pt.err_m < 50) maxErr = pt.err_m;
      });
      maxErr *= 1.1;

      const n = series.length;
      const xToPx = (idx) => padL + (idx / (n - 1)) * plotW;
      const yToPx = (err) => padT + plotH - (Math.min(maxErr, Math.max(0, err)) / maxErr) * plotH;

      // Horizontal target reference (e.g. 5 cm / 10 cm target)
      if (convergedTarget != null && convergedTarget < maxErr) {
        const tgtY = yToPx(convergedTarget);
        ctx.strokeStyle = "rgba(16, 185, 129, 0.4)";
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(padL, tgtY);
        ctx.lineTo(w - padR, tgtY);
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.fillStyle = "var(--tep-emerald)";
        ctx.font = "9px monospace";
        ctx.textAlign = "right";
        ctx.fillText(convergedTarget < 1.0 ? (convergedTarget * 100).toFixed(0) + "cm" : convergedTarget.toFixed(2) + "m", padL - 4, tgtY + 3);
      }

      // Plot series trace
      ctx.strokeStyle = "var(--cyan-kriging)";
      ctx.lineWidth = 1.75;
      ctx.beginPath();
      series.forEach((pt, idx) => {
        const px = xToPx(idx);
        const py = yToPx(pt.err_m);
        if (idx === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      });
      ctx.stroke();

      // Shaded fill to baseline
      ctx.lineTo(xToPx(n - 1), padT + plotH);
      ctx.lineTo(xToPx(0), padT + plotH);
      ctx.closePath();
      ctx.fillStyle = "rgba(6, 182, 212, 0.08)";
      ctx.fill();

      // Labels
      ctx.fillStyle = "var(--text-faint)";
      ctx.font = "9px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Full-Sky Convergence Time Series", padL + plotW / 2, h - 6);
    }

    // ------------------------------------------------------------------------
    // Kriging Field Covariance & Geometry
    // ------------------------------------------------------------------------
    updateKriging(geo, cp, priorSource) {
      geo = geo || {};
      cp = cp || {};

      const distEl = document.getElementById("kriging-dist");
      const corrEl = document.getElementById("kriging-corr");
      const varEl = document.getElementById("kriging-var");
      const cmEl = document.getElementById("kriging-common-mode");
      const residEl = document.getElementById("kriging-resid");
      const targetEl = document.getElementById("kriging-target");

      if (distEl) {
        const d = geo.distance_km != null ? geo.distance_km : geo.separation_km;
        distEl.textContent = d != null ? Math.round(d) + " km" : "--";
      }
      if (corrEl) {
        const rho = geo.expected_field_correlation != null ? geo.expected_field_correlation : 0.92;
        corrEl.textContent = (rho * 100).toFixed(1) + "%";
      }
      if (varEl) {
        const v = cp.network_variance_m2 != null ? cp.network_variance_m2 : 1.25;
        varEl.textContent = Number(v).toFixed(2) + " m²";
      }
      if (cmEl) {
        const cm = cp.network_common_mode_m;
        cmEl.textContent = cm != null ? cm.toFixed(2) + " m" : "--";
      }
      if (residEl) {
        const r = cp.network_residual_m;
        residEl.textContent = r != null ? r.toFixed(3) + " m" : "--";
      }
      if (targetEl) {
        targetEl.textContent = cp.network_target || geo.target_station || geo.target || priorSource || "--";
      }
    }

    // ------------------------------------------------------------------------
    // Astronomical Phase Modulation
    // ------------------------------------------------------------------------
    updateAstronomical(astro, arms) {
      astro = astro || {};
      const ta = (arms && arms.tep) || {};

      const qScaleEl = document.getElementById("astro-q-scale");
      const qVal = ta.astro_q_scale != null ? ta.astro_q_scale : 1.0;
      if (qScaleEl) qScaleEl.textContent = "×" + Number(qVal).toFixed(3);

      const chandlerBar = document.getElementById("phase-bar-chandler");
      const nodalBar = document.getElementById("phase-bar-nodal");
      const semiBar = document.getElementById("phase-bar-semiannual");

      // Approximate orbital phases from current date
      const now = new Date();
      const dayOfYear = Math.floor((now - new Date(now.getFullYear(), 0, 0)) / (1000 * 60 * 60 * 24));
      
      const semiPct = ((dayOfYear % 182.5) / 182.5) * 100;
      const chandlerPct = ((dayOfYear % 433) / 433) * 100;
      const nodalPct = 68.4; // 18.6-year slow precession

      if (semiBar) semiBar.style.width = semiPct.toFixed(1) + "%";
      if (chandlerBar) chandlerBar.style.width = chandlerPct.toFixed(1) + "%";
      if (nodalBar) nodalBar.style.width = nodalPct.toFixed(1) + "%";
    }

    // ------------------------------------------------------------------------
    // Provenance & Audit Ledger
    // ------------------------------------------------------------------------
    updateProvenance(pppData) {
      const truthRefEl = document.getElementById("prov-truth-ref");
      const freqModeEl = document.getElementById("prov-freq-mode");
      const pppSourceEl = document.getElementById("prov-ppp-source");
      const kinRmseEl = document.getElementById("prov-kinematic-rmse");
      const ztdEl = document.getElementById("prov-ztd");
      const bandNoteEl = document.getElementById("tep-band-status");
      const netSrcEl = document.getElementById("prov-network-source");
      const nearestEl = document.getElementById("prov-nearest-station");

      if (truthRefEl) {
        truthRefEl.textContent = pppData.truth_reference === "surveyed" ? "Surveyed ECEF (Ground Truth)" :
                                (pppData.truth_reference === "ppp_static" ? "PPP-Static Converged (L1+L5 IF + Precise)" : "Session Running Mean");
      }

      if (freqModeEl) {
        const fm = pppData.freq_mode || "dual_l1l5";
        freqModeEl.textContent = fm === "dual_l1l5" ? "Dual-Band (L1+L5 Ionosphere-Free)" :
                                 (fm === "dual_l1l2" ? "Dual-Band (L1+L2 IF)" : "Single L1 (GRAPHIC)");
      }

      const fs = pppData.position_fullsky || {};
      if (pppSourceEl) {
        pppSourceEl.textContent = fs.source || "--";
      }
      if (kinRmseEl) {
        const kinRmse = fs.kinematic_rmse_3d_m;
        const kinConv = fs.kinematic_converged_rmse_3d_m;
        if (kinRmse != null) {
          kinRmseEl.textContent = `${kinRmse.toFixed(2)}m (overall)${kinConv != null ? "  ·  " + kinConv.toFixed(2) + "m (conv.)" : ""}`;
          kinRmseEl.style.color = "var(--text-faint)";
        } else {
          kinRmseEl.textContent = "--";
        }
      }

      if (ztdEl) {
        ztdEl.textContent = pppData.ztd_m != null ? Number(pppData.ztd_m).toFixed(3) + " m" : "--";
      }

      if (netSrcEl) {
        if (pppData.network_is_live) {
          const mode = pppData.network_source_mode || "live";
          netSrcEl.textContent = `Live CORS (${mode})`;
          netSrcEl.style.color = "var(--tep-emerald)";
        } else if (pppData.network_source_mode === "simulation") {
          netSrcEl.textContent = "Replayed Archive (Simulation)";
          netSrcEl.style.color = "var(--persist-amber)";
        } else {
          netSrcEl.textContent = "Autonomous (Local Receiver)";
          netSrcEl.style.color = "var(--text-faint)";
        }
      }

      if (nearestEl) {
        const ng = pppData.network_geometry || {};
        const dist = ng.distance_km;
        const target = ng.target_station || ng.target || "--";
        if (dist != null) {
          const tag = dist > 100 ? "DISTANT" : (dist <= 10 ? "CO-LOCATED" : "nearby");
          nearestEl.textContent = `${target} (${dist.toFixed(0)} km, ${tag})`;
          nearestEl.style.color = dist > 100 ? "var(--persist-amber)" : "var(--tep-emerald)";
        } else {
          nearestEl.textContent = "--";
        }
      }

      const cp = pppData.clock_prediction || {};
      if (bandNoteEl && cp.tep_band) {
        const b = cp.tep_band;
        if (b.in_band) {
          bandNoteEl.textContent = `${b.verdict || "OK"} (${Math.round(b.capture_span_s)}s) — Full astronomical & spatial prior active`;
          bandNoteEl.style.color = "var(--tep-emerald)";
        } else {
          bandNoteEl.textContent = `Rolling Window: ${Math.round(b.capture_span_s || 0)}s / ${Math.round(b.min_useful_s || 2000)}s — Kriging prior active; phase modulation scaling with window.`;
          bandNoteEl.style.color = "var(--persist-amber)";
        }
      }
    }

    // ------------------------------------------------------------------------
    // Multi-band TEP kriging (Paper 1 frequency-dependent lambdas)
    // ------------------------------------------------------------------------
    updateMultiband(mb) {
      mb = mb || {};
      const container = document.getElementById("multiband-container");
      if (!container) return;

      const enabled = mb.enabled === true;
      const mbErr = mb.error_m;
      const sbErr = mb.singleband_error_m;
      const mbPred = mb.predicted_m;
      const sbPred = mb.singleband_predicted_m;
      const bands = mb.bands || [];

      if (!enabled || mbPred == null) {
        container.style.display = "none";
        return;
      }
      container.style.display = "block";

      const errEl = document.getElementById("multiband-err");
      const cmpEl = document.getElementById("multiband-comparison");
      const bandsEl = document.getElementById("multiband-bands");

      if (errEl) {
        errEl.textContent = mbErr != null ? mbErr.toFixed(3) + " m" : "--";
      }

      if (cmpEl && sbErr != null && mbErr != null) {
        const pct = ((sbErr - mbErr) / sbErr * 100);
        const sign = pct >= 0 ? "+" : "";
        cmpEl.textContent = `Single-band: ${sbErr.toFixed(3)}m  ·  Multi-band: ${mbErr.toFixed(3)}m  ·  ${sign}${pct.toFixed(1)}%`;
        cmpEl.style.color = pct >= 0 ? "var(--tep-emerald)" : "var(--persist-amber)";
      } else if (cmpEl) {
        cmpEl.textContent = `Single-band: ${sbErr != null ? sbErr.toFixed(3) + "m" : "--"}  ·  Multi-band: ${mbErr != null ? mbErr.toFixed(3) + "m" : "--"}`;
      }

      if (bandsEl) {
        bandsEl.innerHTML = bands.map(b => {
          const lam = b.lambda_km != null ? Number(b.lambda_km).toFixed(0) + " km" : "--";
          const fLo = b.f_lo_uhz != null ? Number(b.f_lo_uhz).toFixed(1) : "--";
          const fHi = b.f_hi_uhz != null ? Number(b.f_hi_uhz).toFixed(1) : "--";
          return `<span style="margin-right:12px;font-size:0.85em;color:var(--text-faint);">${b.name}: ${fLo}–${fHi} µHz  ·  λ=${lam}</span>`;
        }).join("");
      }
    }
  }

  window.AuditPanels = AuditPanels;
})();
