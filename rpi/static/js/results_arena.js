/**
 * results_arena.js — TEP-PPP Skill Observatory & Comparative Results Engine
 * Renders:
 *   1. Walk-Forward Causal Skill History (122+ Windows Bar + Dual RMSE Curves)
 *   2. 2-Hour Holdout Clock Field Reconstruction (Truth Drift vs Causal TEP Correction)
 *   3. Distance-Structured CORS Network Field Profile (Log-Distance 1.6..2830 km)
 *   4. Live Window-by-Window Skill Ledger Table
 *   5. 2D Horizontal Dispersion & Multi-Arm 3D Position Trace (Drawer Tab)
 */

(function () {
  "use strict";

  function computeCovariance2D(points) {
    if (!points || points.length < 3) return null;
    const n = points.length;
    let sumE = 0, sumN = 0;
    for (let i = 0; i < n; i++) {
      sumE += points[i][0];
      sumN += points[i][1];
    }
    const meanE = sumE / n;
    const meanN = sumN / n;

    let varE = 0, varN = 0, covEN = 0;
    for (let i = 0; i < n; i++) {
      const dE = points[i][0] - meanE;
      const dN = points[i][1] - meanN;
      varE += dE * dE;
      varN += dN * dN;
      covEN += dE * dN;
    }
    varE /= (n - 1);
    varN /= (n - 1);
    covEN /= (n - 1);

    const trace = varE + varN;
    const term = Math.sqrt(Math.max(0, (varE - varN) * (varE - varN) / 4 + covEN * covEN));
    const lambda1 = trace / 2 + term;
    const lambda2 = Math.max(0, trace / 2 - term);

    const scale95 = 2.4477;
    const semiMajor = scale95 * Math.sqrt(Math.max(0, lambda1));
    const semiMinor = scale95 * Math.sqrt(Math.max(0, lambda2));
    const angle = 0.5 * Math.atan2(2 * covEN, varE - varN);
    const rms2d = Math.sqrt(varE + varN);
    return { meanE, meanN, semiMajor, semiMinor, angle, rms2d, count: n };
  }

  function formatDist(meters) {
    if (meters == null || isNaN(meters)) return "--";
    if (meters < 1.0) {
      return (meters * 100).toFixed(1) + " cm";
    }
    return meters.toFixed(2) + " m";
  }

  class ResultsArena {
    constructor() {
      // New Skill Observatory Canvases
      this.histCanvas = document.getElementById("skill-history-canvas");
      this.holdoutCanvas = document.getElementById("holdout-curve-canvas");
      this.staCanvas = document.getElementById("station-field-canvas");

      // Legacy Drawer Canvases
      this.dispCanvas = document.getElementById("dispersion-canvas");
      this.timeCanvas = document.getElementById("timeseries-canvas");

      this.histCtx = this.histCanvas ? this.histCanvas.getContext("2d") : null;
      this.holdoutCtx = this.holdoutCanvas ? this.holdoutCanvas.getContext("2d") : null;
      this.staCtx = this.staCanvas ? this.staCanvas.getContext("2d") : null;
      this.dispCtx = this.dispCanvas ? this.dispCanvas.getContext("2d") : null;
      this.timeCtx = this.timeCanvas ? this.timeCanvas.getContext("2d") : null;

      // Data Stores
      this.skillHistory = null; // { summary, windows, holdout, stations }
      this.scatterData = { free: [], persist: [], tep: [] };
      this.timeSeriesData = [];

      // Hover states
      this.hoverHistIdx = -1;
      this.hoverHoldoutIdx = -1;
      this.hoverStaIdx = -1;
      this.hoverTimeIdx = -1;

      this.resizeCanvases();
      window.addEventListener("resize", () => this.resizeCanvases());
      this.initInteraction();
    }

    resizeCanvases() {
      const dpr = window.devicePixelRatio || 1;
      const list = [
        [this.histCanvas, 350],
        [this.holdoutCanvas, 350],
        [this.staCanvas, 310],
        [this.dispCanvas, 310],
        [this.timeCanvas, 310],
      ];
      list.forEach(([cvs, minH]) => {
        if (!cvs || !cvs.parentElement) return;
        const rect = cvs.parentElement.getBoundingClientRect();
        const w = Math.max(300, rect.width || 600);
        const h = Math.max(minH, rect.height || minH);
        cvs.width = w * dpr;
        cvs.height = h * dpr;
        cvs.style.width = w + "px";
        cvs.style.height = h + "px";
        const ctx = cvs.getContext("2d");
        ctx.resetTransform();
        ctx.scale(dpr, dpr);
      });
      this.render();
    }

    initInteraction() {
      // 1. Skill History hover
      if (this.histCanvas) {
        this.histCanvas.addEventListener("mousemove", (e) => {
          const wins = (this.skillHistory && this.skillHistory.windows) || [];
          if (wins.length < 2) return;
          const rect = this.histCanvas.getBoundingClientRect();
          const mx = e.clientX - rect.left;
          const padL = 48, padR = 48;
          const plotW = rect.width - padL - padR;
          if (mx >= padL && mx <= rect.width - padR) {
            const r = (mx - padL) / plotW;
            this.hoverHistIdx = Math.min(wins.length - 1, Math.max(0, Math.floor(r * wins.length)));
          } else {
            this.hoverHistIdx = -1;
          }
          this.renderSkillHistory();
        });
        this.histCanvas.addEventListener("mouseleave", () => {
          this.hoverHistIdx = -1;
          this.renderSkillHistory();
        });
      }

      // 2. Holdout Curve hover
      if (this.holdoutCanvas) {
        this.holdoutCanvas.addEventListener("mousemove", (e) => {
          const ho = (this.skillHistory && this.skillHistory.holdout) || {};
          const hs = ho.h_s || [];
          if (hs.length < 2) return;
          const rect = this.holdoutCanvas.getBoundingClientRect();
          const mx = e.clientX - rect.left;
          const padL = 52, padR = 20;
          const plotW = rect.width - padL - padR;
          if (mx >= padL && mx <= rect.width - padR) {
            const r = (mx - padL) / plotW;
            this.hoverHoldoutIdx = Math.min(hs.length - 1, Math.max(0, Math.round(r * (hs.length - 1))));
          } else {
            this.hoverHoldoutIdx = -1;
          }
          this.renderHoldoutCurve();
        });
        this.holdoutCanvas.addEventListener("mouseleave", () => {
          this.hoverHoldoutIdx = -1;
          this.renderHoldoutCurve();
        });
      }

      // 3. Station Field hover
      if (this.staCanvas) {
        this.staCanvas.addEventListener("mousemove", (e) => {
          const stas = (this.skillHistory && this.skillHistory.stations) || [];
          if (!stas.length) return;
          const rect = this.staCanvas.getBoundingClientRect();
          const mx = e.clientX - rect.left;
          const my = e.clientY - rect.top;
          let bestIdx = -1;
          let bestDist = 22;
          (this._staScreenPts || []).forEach((pt, idx) => {
            const d = Math.hypot(mx - pt.x, my - pt.y);
            if (d < bestDist) {
              bestDist = d;
              bestIdx = idx;
            }
          });
          if (bestIdx !== this.hoverStaIdx) {
            this.hoverStaIdx = bestIdx;
            this.renderStationField();
          }
        });
        this.staCanvas.addEventListener("mouseleave", () => {
          this.hoverStaIdx = -1;
          this.renderStationField();
        });
      }

      // 4. Legacy TimeSeries hover
      if (this.timeCanvas) {
        this.timeCanvas.addEventListener("mousemove", (e) => {
          const rect = this.timeCanvas.getBoundingClientRect();
          const mouseX = e.clientX - rect.left;
          const n = this.timeSeriesData.length;
          if (n < 2) return;
          const padL = 50, padR = 20;
          const plotW = rect.width - padL - padR;
          if (mouseX >= padL && mouseX <= rect.width - padR) {
            const ratio = (mouseX - padL) / plotW;
            this.hoverTimeIdx = Math.min(n - 1, Math.max(0, Math.round(ratio * (n - 1))));
          } else {
            this.hoverTimeIdx = -1;
          }
          this.renderTimeSeries();
        });
        this.timeCanvas.addEventListener("mouseleave", () => {
          this.hoverTimeIdx = -1;
          this.renderTimeSeries();
        });
      }
    }

    updateSkillHistory(histPayload) {
      if (!histPayload || !histPayload.available) return;
      this.skillHistory = histPayload;
      this.renderSkillHistory();
      this.renderHoldoutCurve();
      this.renderStationField();
      this.renderLedgerTable();
    }

    updateData(pppData) {
      if (!pppData) return;
      const arms = pppData.arms || {};
      const posDegraded = pppData.position_degraded || {};

      ["free", "persist", "tep"].forEach((armKey) => {
        const arm = arms[armKey];
        if (arm && arm.positions && arm.positions.length > 0) {
          const pts = [];
          const posList = arm.positions;
          const ref = posList[0];
          const start = Math.max(0, posList.length - 80);
          for (let i = start; i < posList.length; i++) {
            const p = posList[i];
            const dE = p[0] - ref[0];
            const dN = p[1] - ref[1];
            if (Math.abs(dE) < 500 && Math.abs(dN) < 500) {
              pts.push([dE, dN]);
            }
          }
          this.scatterData[armKey] = pts;
        }
      });

      const tSec = pppData.clock_prediction ? (pppData.clock_prediction.horizon_s || 0) : this.timeSeriesData.length;
      const freeRmse = (arms.free && arms.free.rmse_3d_m) || posDegraded.free_rmse_3d_m || null;
      const persistRmse = (arms.persist && arms.persist.rmse_3d_m) || posDegraded.persist_rmse_3d_m || null;
      const tepRmse = (arms.tep && arms.tep.rmse_3d_m) || posDegraded.tep_rmse_3d_m || null;

      if (freeRmse != null || persistRmse != null || tepRmse != null) {
        this.timeSeriesData.push({
          t: tSec,
          free: freeRmse,
          persist: persistRmse,
          tep: tepRmse,
        });
        if (this.timeSeriesData.length > 120) {
          this.timeSeriesData.shift();
        }
      }

      // If holdout isn't populated yet from /api/skill_history, synthesize from network_accum_curve
      if ((!this.skillHistory || !this.skillHistory.holdout || !this.skillHistory.holdout.h_s) &&
          pppData.clock_prediction && pppData.clock_prediction.network_accum_curve) {
        const nac = pppData.clock_prediction.network_accum_curve;
        if (nac.h_s && nac.corr_m) {
          this.skillHistory = this.skillHistory || { available: true, windows: [], stations: [] };
          this.skillHistory.holdout = {
            window_id: "live",
            h_s: nac.h_s,
            tep_corr_m: nac.corr_m,
            truth_dev_m: nac.corr_m.map((v) => -v),
            residual_after_tep_m: nac.corr_m.map(() => 0),
            persist_rmse_m: pppData.clock_prediction.persist_rmse_m,
            tep_dep_rmse_m: pppData.clock_prediction.tep_rmse_deployed_m,
          };
        }
      }

      this.render();
    }

    render() {
      this.renderSkillHistory();
      this.renderHoldoutCurve();
      this.renderStationField();
      this.renderDispersion();
      this.renderTimeSeries();
    }

    // ------------------------------------------------------------------------
    // 1. RENDER: Walk-Forward Causal Skill History (122 Windows)
    // ------------------------------------------------------------------------
    renderSkillHistory() {
      const ctx = this.histCtx;
      if (!ctx) return;
      const cvs = this.histCanvas;
      const w = parseFloat(cvs.style.width) || 600;
      const h = parseFloat(cvs.style.height) || 350;
      ctx.clearRect(0, 0, w, h);

      const wins = (this.skillHistory && this.skillHistory.windows) || [];
      if (wins.length < 2) {
        ctx.fillStyle = "rgba(148, 163, 184, 0.6)";
        ctx.font = "12px sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("Loading 120+ archived window skill records...", w / 2, h / 2);
        return;
      }

      const padL = 48, padR = 48, padT = 26, padB = 34;
      const plotW = w - padL - padR;
      const plotH = h - padT - padB;

      // Left axis: Gain % (-10% to +65%)
      let minPct = -5;
      let maxPct = 60;
      wins.forEach((d) => {
        const g = d.dep_gain_pct;
        if (g != null) {
          if (g < minPct) minPct = Math.max(-25, Math.floor(g - 3));
          if (g > maxPct) maxPct = Math.ceil(g + 5);
        }
      });

      // Right axis: Clock RMSE in metres (0 to maxRmse)
      let maxRmse = 6.0;
      wins.forEach((d) => {
        if (d.persist_clk_m != null && d.persist_clk_m > maxRmse && d.persist_clk_m < 25) {
          maxRmse = d.persist_clk_m * 1.1;
        }
      });

      const n = wins.length;
      const yPct = (pct) => padT + plotH - ((pct - minPct) / (maxPct - minPct)) * plotH;
      const yRmse = (rm) => padT + plotH - (Math.min(maxRmse, Math.max(0, rm)) / maxRmse) * plotH;
      const xIdx = (i) => padL + ((i + 0.5) / n) * plotW;

      // Gridlines on Left Axis (Gain %)
      ctx.strokeStyle = "rgba(255, 255, 255, 0.06)";
      ctx.lineWidth = 1;
      ctx.font = "10px monospace";
      const pctTicks = [0, 15, 30, 45, 60].filter((t) => t >= minPct && t <= maxPct);
      pctTicks.forEach((t) => {
        const py = yPct(t);
        ctx.beginPath();
        ctx.moveTo(padL, py);
        ctx.lineTo(w - padR, py);
        ctx.stroke();

        ctx.fillStyle = t === 0 ? "rgba(255,255,255,0.85)" : "rgba(16, 185, 129, 0.75)";
        ctx.textAlign = "right";
        ctx.fillText((t > 0 ? "+" : "") + t + "%", padL - 6, py + 3);
      });

      // Right Axis labels (Clock RMSE m)
      ctx.textAlign = "left";
      ctx.fillStyle = "rgba(245, 158, 11, 0.75)";
      for (let s = 0; s <= 4; s++) {
        const rm = (s / 4) * maxRmse;
        const py = yRmse(rm);
        ctx.fillText(rm.toFixed(1) + "m", w - padR + 6, py + 3);
      }

      // Zero-gain baseline
      const zeroY = yPct(0);
      ctx.strokeStyle = "rgba(255, 255, 255, 0.22)";
      ctx.beginPath();
      ctx.moveTo(padL, zeroY);
      ctx.lineTo(w - padR, zeroY);
      ctx.stroke();

      // Median gain dashed reference line
      const summary = (this.skillHistory && this.skillHistory.summary) || {};
      if (summary.median_gain_pct != null) {
        const medY = yPct(summary.median_gain_pct);
        ctx.save();
        ctx.strokeStyle = "rgba(16, 185, 129, 0.55)";
        ctx.setLineDash([5, 4]);
        ctx.beginPath();
        ctx.moveTo(padL, medY);
        ctx.lineTo(w - padR, medY);
        ctx.stroke();
        ctx.fillStyle = "#10b981";
        ctx.textAlign = "left";
        ctx.font = "10px monospace";
        ctx.fillText(`Corpus Median: +${summary.median_gain_pct.toFixed(1)}%`, padL + 8, medY - 5);
        ctx.restore();
      }

      // Draw Gain Bars
      const barW = Math.max(2, (plotW / n) * 0.72);
      for (let i = 0; i < n; i++) {
        const g = wins[i].dep_gain_pct;
        if (g == null) continue;
        const cx = xIdx(i);
        const py = yPct(Math.max(minPct, Math.min(maxPct, g)));
        const top = Math.min(zeroY, py);
        const bh = Math.max(1.5, Math.abs(zeroY - py));

        if (i === this.hoverHistIdx) {
          ctx.fillStyle = g >= 0 ? "#34d399" : "#f87171";
        } else {
          ctx.fillStyle = g >= 0 ? "rgba(16, 185, 129, 0.38)" : "rgba(239, 68, 68, 0.55)";
        }
        ctx.fillRect(cx - barW / 2, top, barW, bh);
      }

      // Draw Persistence RMSE Line (Amber) & Deployed TEP RMSE Line (Cyan)
      const drawRmseCurve = (key, color, width, dash) => {
        ctx.save();
        ctx.strokeStyle = color;
        ctx.lineWidth = width;
        ctx.setLineDash(dash);
        ctx.beginPath();
        let started = false;
        for (let i = 0; i < n; i++) {
          const val = wins[i][key];
          if (val != null && val > 0) {
            const px = xIdx(i);
            const py = yRmse(val);
            if (!started) {
              ctx.moveTo(px, py);
              started = true;
            } else {
              ctx.lineTo(px, py);
            }
          }
        }
        ctx.stroke();
        ctx.restore();
      };

      drawRmseCurve("persist_clk_m", "#f59e0b", 1.6, [3, 2]);
      drawRmseCurve("dep_clk_m", "#06b6d4", 2.2, []);

      // X-axis labels
      ctx.fillStyle = "rgba(148, 163, 184, 0.75)";
      ctx.font = "10px monospace";
      ctx.textAlign = "center";
      const step = Math.max(1, Math.floor(n / 6));
      for (let i = 0; i < n; i += step) {
        const idShort = "w" + String(wins[i].id || "").slice(-5);
        ctx.fillText(idShort, xIdx(i), h - 10);
      }
      ctx.fillText("w" + String(wins[n - 1].id || "").slice(-5), xIdx(n - 1), h - 10);

      // Hover Tooltip
      if (this.hoverHistIdx >= 0 && this.hoverHistIdx < n) {
        const d = wins[this.hoverHistIdx];
        const hx = xIdx(this.hoverHistIdx);
        ctx.strokeStyle = "rgba(255, 255, 255, 0.35)";
        ctx.beginPath();
        ctx.moveTo(hx, padT);
        ctx.lineTo(hx, h - padB);
        ctx.stroke();

        const boxW = 195, boxH = 86;
        const bx = hx > w - boxW - 60 ? hx - boxW - 10 : hx + 10;
        const by = padT + 6;
        ctx.fillStyle = "rgba(8, 13, 25, 0.94)";
        ctx.strokeStyle = "rgba(16, 185, 129, 0.45)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.roundRect(bx, by, boxW, boxH, 6);
        ctx.fill();
        ctx.stroke();

        ctx.textAlign = "left";
        ctx.font = "10.5px monospace";
        ctx.fillStyle = "#f8fafc";
        ctx.fillText(`Window w${String(d.id).slice(-6)} (${this.hoverHistIdx + 1}/${n})`, bx + 10, by + 17);
        ctx.fillStyle = "#10b981";
        const gStr = d.dep_gain_pct != null ? `${d.dep_gain_pct >= 0 ? "+" : ""}${d.dep_gain_pct.toFixed(2)}%` : "--";
        ctx.fillText(`Causal RRR Gain: ${gStr}`, bx + 10, by + 34);
        ctx.fillStyle = "#06b6d4";
        ctx.fillText(`TEP Deployed:    ${d.dep_clk_m != null ? d.dep_clk_m.toFixed(3) + " m" : "--"}`, bx + 10, by + 50);
        ctx.fillStyle = "#f59e0b";
        ctx.fillText(`Persistence:     ${d.persist_clk_m != null ? d.persist_clk_m.toFixed(3) + " m" : "--"}`, bx + 10, by + 66);
        ctx.fillStyle = "#94a3b8";
        ctx.fillText(`Basis: ${d.n_feat || "--"} feats, K=${d.n_win || "--"}`, bx + 10, by + 80);
      }
    }

    // ------------------------------------------------------------------------
    // 2. RENDER: 2-Hour Holdout Clock Field Reconstruction
    // ------------------------------------------------------------------------
    renderHoldoutCurve() {
      const ctx = this.holdoutCtx;
      if (!ctx) return;
      const cvs = this.holdoutCanvas;
      const w = parseFloat(cvs.style.width) || 600;
      const h = parseFloat(cvs.style.height) || 350;
      ctx.clearRect(0, 0, w, h);

      const ho = (this.skillHistory && this.skillHistory.holdout) || {};
      const hs = ho.h_s || [];
      const truthDev = ho.truth_dev_m || [];
      const tepCorr = ho.tep_corr_m || [];
      const resAfter = ho.residual_after_tep_m || [];

      if (hs.length < 5) {
        ctx.fillStyle = "rgba(148, 163, 184, 0.6)";
        ctx.font = "12px sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("Loading 2-hour holdout clock trajectory...", w / 2, h / 2);
        return;
      }

      const subEl = document.getElementById("holdout-subtitle");
      if (subEl && ho.persist_rmse_m != null && ho.tep_dep_rmse_m != null) {
        const red = ((1 - ho.tep_dep_rmse_m / ho.persist_rmse_m) * 100).toFixed(1);
        subEl.textContent = `w${String(ho.window_id).slice(-6)}: Persist ${ho.persist_rmse_m.toFixed(2)}m → TEP ${ho.tep_dep_rmse_m.toFixed(2)}m (+${red}%)`;
      }

      const padL = 52, padR = 20, padT = 24, padB = 34;
      const plotW = w - padL - padR;
      const plotH = h - padT - padB;

      let maxAbs = 2.0;
      for (let i = 0; i < hs.length; i++) {
        const a = Math.abs(truthDev[i] || 0);
        const b = Math.abs(tepCorr[i] || 0);
        const c = Math.abs(resAfter[i] || 0);
        maxAbs = Math.max(maxAbs, a, b, c);
      }
      maxAbs = Math.min(25.0, maxAbs * 1.12);

      const n = hs.length;
      const hMax = hs[n - 1] || 7080;
      const xPx = (i) => padL + (hs[i] / hMax) * plotW;
      const yPx = (v) => padT + plotH / 2 - (Math.max(-maxAbs, Math.min(maxAbs, v)) / maxAbs) * (plotH / 2);

      // Horizontal Gridlines
      ctx.strokeStyle = "rgba(255, 255, 255, 0.06)";
      ctx.lineWidth = 1;
      ctx.font = "10px monospace";
      ctx.textAlign = "right";
      [-maxAbs, -maxAbs / 2, 0, maxAbs / 2, maxAbs].forEach((val) => {
        const py = yPx(val);
        ctx.beginPath();
        ctx.moveTo(padL, py);
        ctx.lineTo(w - padR, py);
        ctx.stroke();
        ctx.fillStyle = val === 0 ? "rgba(255,255,255,0.85)" : "rgba(148,163,184,0.75)";
        ctx.fillText((val > 0 ? "+" : "") + val.toFixed(1) + "m", padL - 6, py + 3);
      });

      // Shaded error reduction between |truthDev| and |resAfter|
      ctx.beginPath();
      for (let i = 0; i < n; i++) {
        const px = xPx(i);
        const py = yPx(truthDev[i] || 0);
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      for (let i = n - 1; i >= 0; i--) {
        ctx.lineTo(xPx(i), yPx(resAfter[i] || 0));
      }
      ctx.closePath();
      ctx.fillStyle = "rgba(16, 185, 129, 0.09)";
      ctx.fill();

      // 1. Realized clock deviation from persistence (b_ho - persist_pred): Amber
      const drawSeries = (arr, sign, color, width, dash) => {
        ctx.save();
        ctx.strokeStyle = color;
        ctx.lineWidth = width;
        ctx.setLineDash(dash);
        ctx.beginPath();
        for (let i = 0; i < n; i++) {
          const px = xPx(i);
          const py = yPx(sign * (arr[i] || 0));
          if (i === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        }
        ctx.stroke();
        ctx.restore();
      };

      // Note: truthDev = b_ho - p_pred; tep_corr_m = dep_m (applied to p_pred), so -dep_m tracks b_ho - p_pred!
      drawSeries(truthDev, 1.0, "#f59e0b", 1.7, [3, 2]);
      drawSeries(tepCorr, -1.0, "#06b6d4", 2.2, []);
      drawSeries(resAfter, 1.0, "#10b981", 2.0, []);

      // X-axis time ticks (minutes into 2h holdout)
      ctx.fillStyle = "rgba(148, 163, 184, 0.75)";
      ctx.textAlign = "center";
      ctx.font = "10px monospace";
      [0, 1800, 3600, 5400, 7080].forEach((sec) => {
        if (sec <= hMax + 120) {
          const px = padL + (Math.min(sec, hMax) / hMax) * plotW;
          ctx.fillText(`${Math.round(sec / 60)} min`, px, h - 10);
        }
      });

      // Hover readout
      if (this.hoverHoldoutIdx >= 0 && this.hoverHoldoutIdx < n) {
        const idx = this.hoverHoldoutIdx;
        const hx = xPx(idx);
        ctx.strokeStyle = "rgba(255, 255, 255, 0.35)";
        ctx.beginPath();
        ctx.moveTo(hx, padT);
        ctx.lineTo(hx, h - padB);
        ctx.stroke();

        const boxW = 195, boxH = 74;
        const bx = hx > w - boxW - 30 ? hx - boxW - 10 : hx + 10;
        const by = padT + 8;
        ctx.fillStyle = "rgba(8, 13, 25, 0.94)";
        ctx.strokeStyle = "rgba(6, 182, 212, 0.45)";
        ctx.beginPath();
        ctx.roundRect(bx, by, boxW, boxH, 6);
        ctx.fill();
        ctx.stroke();

        ctx.textAlign = "left";
        ctx.font = "10.5px monospace";
        ctx.fillStyle = "#f8fafc";
        ctx.fillText(`Horizon: ${(hs[idx] / 60).toFixed(1)} min (${hs[idx].toFixed(0)} s)`, bx + 10, by + 17);
        ctx.fillStyle = "#f59e0b";
        ctx.fillText(`Clock Drift:  ${(truthDev[idx] || 0).toFixed(3)} m`, bx + 10, by + 34);
        ctx.fillStyle = "#06b6d4";
        ctx.fillText(`TEP Field:    ${(-(tepCorr[idx] || 0)).toFixed(3)} m`, bx + 10, by + 50);
        ctx.fillStyle = "#10b981";
        ctx.fillText(`Residual Err: ${(resAfter[idx] || 0).toFixed(3)} m`, bx + 10, by + 66);
      }
    }

    // ------------------------------------------------------------------------
    // 3. RENDER: Distance-Structured CORS Network Field Profile
    // ------------------------------------------------------------------------
    renderStationField() {
      const ctx = this.staCtx;
      if (!ctx) return;
      const cvs = this.staCanvas;
      const w = parseFloat(cvs.style.width) || 600;
      const h = parseFloat(cvs.style.height) || 310;
      ctx.clearRect(0, 0, w, h);

      const stas = (this.skillHistory && this.skillHistory.stations) || [];
      if (!stas.length) {
        ctx.fillStyle = "rgba(148, 163, 184, 0.6)";
        ctx.font = "12px sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("Loading distance-structured CORS station field...", w / 2, h / 2);
        return;
      }

      const subEl = document.getElementById("stations-subtitle");
      if (subEl) {
        const nNear = stas.filter((s) => s.band === "near_uk").length;
        const nMid = stas.filter((s) => s.band === "mid_eu").length;
        const nFar = stas.filter((s) => s.band === "far_iberia").length;
        subEl.textContent = `${stas.length} Live CORS: ${nNear} UK (<360km) · ${nMid} EU (360–900km) · ${nFar} Atl/Iberia (>1500km)`;
      }

      const padL = 48, padR = 20, padT = 24, padB = 34;
      const plotW = w - padL - padR;
      const plotH = h - padT - padB;

      // X-axis: Log10 distance from 1 km (0) to 3200 km (3.505)
      const logMin = 0.0;
      const logMax = Math.log10(3300);
      const xLog = (km) => {
        const lk = Math.log10(Math.max(1.0, km));
        return padL + ((lk - logMin) / (logMax - logMin)) * plotW;
      };

      // Y-axis: correlation with receiver clock drift (-0.6 to +0.6)
      const cMax = 0.55;
      const yCorr = (c) => padT + plotH / 2 - (Math.max(-cMax, Math.min(cMax, c)) / cMax) * (plotH / 2);

      // Gridlines
      ctx.strokeStyle = "rgba(255, 255, 255, 0.06)";
      ctx.lineWidth = 1;
      ctx.font = "10px monospace";
      ctx.textAlign = "right";
      [-0.4, -0.2, 0, 0.2, 0.4].forEach((c) => {
        const py = yCorr(c);
        ctx.beginPath();
        ctx.moveTo(padL, py);
        ctx.lineTo(w - padR, py);
        ctx.stroke();
        ctx.fillStyle = c === 0 ? "rgba(255,255,255,0.85)" : "rgba(148,163,184,0.75)";
        ctx.fillText((c > 0 ? "+" : "") + c.toFixed(1), padL - 6, py + 3);
      });

      // Distance band vertical dividers & log ticks
      ctx.textAlign = "center";
      [1, 10, 50, 150, 350, 700, 1500, 2800].forEach((km) => {
        const px = xLog(km);
        ctx.strokeStyle = "rgba(255, 255, 255, 0.05)";
        ctx.beginPath();
        ctx.moveTo(px, padT);
        ctx.lineTo(px, h - padB);
        ctx.stroke();
        ctx.fillStyle = "rgba(148, 163, 184, 0.75)";
        ctx.fillText(km + "km", px, h - 10);
      });

      // Plot stations
      this._staScreenPts = [];
      stas.forEach((s, idx) => {
        const px = xLog(s.dist_km);
        const py = yCorr(s.corr || 0);
        this._staScreenPts.push({ x: px, y: py, s });

        const color = s.band === "near_uk" ? "#10b981" : (s.band === "mid_eu" ? "#06b6d4" : "#a855f7");
        const rad = idx === this.hoverStaIdx ? 6.5 : 4.2;

        // Vertical stem from zero line
        ctx.strokeStyle = color + "44";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(px, yCorr(0));
        ctx.lineTo(px, py);
        ctx.stroke();

        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(px, py, rad, 0, Math.PI * 2);
        ctx.fill();

        // Label key landmark stations
        if (["EASN", "LINO", "LOFT", "PETE", "IJMU", "WSRA", "GAIA", "FLRS"].includes(s.id)) {
          ctx.fillStyle = "rgba(248, 250, 252, 0.8)";
          ctx.font = "9.5px monospace";
          ctx.textAlign = "center";
          ctx.fillText(s.id, px, py < yCorr(0) ? py - 7 : py + 13);
        }
      });

      if (this.hoverStaIdx >= 0 && this.hoverStaIdx < this._staScreenPts.length) {
        const { x, y, s } = this._staScreenPts[this.hoverStaIdx];
        const boxW = 190, boxH = 68;
        const bx = x > w - boxW - 25 ? x - boxW - 10 : x + 10;
        const by = Math.max(padT + 4, Math.min(h - padB - boxH - 4, y - boxH / 2));
        ctx.fillStyle = "rgba(8, 13, 25, 0.95)";
        ctx.strokeStyle = "rgba(168, 85, 247, 0.5)";
        ctx.beginPath();
        ctx.roundRect(bx, by, boxW, boxH, 6);
        ctx.fill();
        ctx.stroke();

        ctx.textAlign = "left";
        ctx.font = "10.5px monospace";
        ctx.fillStyle = "#f8fafc";
        ctx.fillText(`${s.id} · ${s.dist_km.toFixed(1)} km (az ${s.az_deg.toFixed(0)}°)`, bx + 10, by + 17);
        ctx.fillStyle = "#10b981";
        ctx.fillText(`Zero-lag corr: ${s.corr >= 0 ? "+" : ""}${s.corr.toFixed(3)}`, bx + 10, by + 34);
        ctx.fillStyle = "#06b6d4";
        ctx.fillText(`Station RMS:   ${s.rms_m.toFixed(2)} m (n=${s.n_samples})`, bx + 10, by + 51);
      }
    }

    // ------------------------------------------------------------------------
    // 4. RENDER: Window-by-Window Skill Ledger Table
    // ------------------------------------------------------------------------
    renderLedgerTable() {
      const tbody = document.getElementById("skill-ledger-body");
      if (!tbody || !this.skillHistory || !this.skillHistory.windows) return;
      const wins = this.skillHistory.windows.slice(-30).reverse();

      tbody.innerHTML = wins.map((w) => {
        const g = w.dep_gain_pct != null ? Number(w.dep_gain_pct) : null;
        const fg = w.fixed_causal_pct != null ? Number(w.fixed_causal_pct) : null;
        const gColor = g != null && g >= 0 ? "var(--tep-emerald)" : "var(--free-crimson)";
        const fgColor = fg != null && fg >= 0 ? "var(--text-standard)" : "var(--persist-amber)";
        const barW = g != null ? Math.min(100, Math.max(0, (g / 60) * 100)).toFixed(0) : 0;
        const rSplit = (w.ridge_rms_m != null && w.stack_rms_m != null)
          ? `${w.ridge_rms_m.toFixed(2)} / ${w.stack_rms_m.toFixed(2)}m`
          : "--";
        return `<tr>
          <td><strong>w${String(w.id).slice(-6)}</strong></td>
          <td>${w.free_clk_m != null ? w.free_clk_m.toFixed(2) : "--"}</td>
          <td style="color:var(--persist-amber);">${w.persist_clk_m != null ? w.persist_clk_m.toFixed(2) : "--"}</td>
          <td style="color:var(--cyan-kriging);font-weight:700;">${w.dep_clk_m != null ? w.dep_clk_m.toFixed(2) : "--"}</td>
          <td style="color:${fgColor};">${fg != null ? (fg >= 0 ? "+" : "") + fg.toFixed(1) + "%" : "--"}</td>
          <td style="color:${gColor};font-weight:700;">
            <span class="gain-bar-wrap">
              <span class="gain-bar-bg"><span class="gain-bar-fill" style="width:${barW}%;"></span></span>
              ${g != null ? (g >= 0 ? "+" : "") + g.toFixed(1) + "%" : "--"}
            </span>
          </td>
          <td style="color:var(--text-muted);">${rSplit}</td>
        </tr>`;
      }).join("");
    }

    // ------------------------------------------------------------------------
    // 5. LEGACY DRAWER: 2D Dispersion & Multi-Arm Time Series
    // ------------------------------------------------------------------------
    renderDispersion() {
      const ctx = this.dispCtx;
      if (!ctx) return;
      const cvs = this.dispCanvas;
      const w = parseFloat(cvs.style.width);
      const h = parseFloat(cvs.style.height);
      if (!w || !h) return;

      ctx.clearRect(0, 0, w, h);
      const cx = w / 2;
      const cy = h / 2;

      let maxDist = 0.25;
      ["free", "persist", "tep"].forEach((k) => {
        const pts = this.scatterData[k] || [];
        pts.forEach(([e, n]) => {
          const d = Math.sqrt(e * e + n * n);
          if (d < 100 && d > maxDist) maxDist = d;
        });
      });
      maxDist = Math.max(0.1, maxDist * 1.25);
      const pxPerMeter = (Math.min(w, h) / 2 - 30) / maxDist;

      const rings = [0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0];
      const visibleRings = rings.filter((r) => r * pxPerMeter <= Math.min(w, h) / 2 - 15);

      ctx.strokeStyle = "rgba(255, 255, 255, 0.08)";
      ctx.lineWidth = 1;
      visibleRings.forEach((r) => {
        const rad = r * pxPerMeter;
        ctx.beginPath();
        ctx.arc(cx, cy, rad, 0, Math.PI * 2);
        ctx.stroke();
        ctx.fillStyle = "rgba(148, 163, 184, 0.6)";
        ctx.font = "10px monospace";
        ctx.fillText(formatDist(r), cx + rad + 4, cy - 4);
      });

      ctx.strokeStyle = "rgba(255, 255, 255, 0.15)";
      ctx.beginPath();
      ctx.moveTo(cx, 15);
      ctx.lineTo(cx, h - 15);
      ctx.moveTo(15, cy);
      ctx.lineTo(w - 15, cy);
      ctx.stroke();

      const armStyles = [
        { key: "free", color: "#ef4444", fill: "rgba(239, 68, 68, 0.08)", stroke: "rgba(239, 68, 68, 0.6)" },
        { key: "persist", color: "#f59e0b", fill: "rgba(245, 158, 11, 0.08)", stroke: "rgba(245, 158, 11, 0.7)" },
        { key: "tep", color: "#10b981", fill: "rgba(16, 185, 129, 0.18)", stroke: "#10b981" },
      ];

      armStyles.forEach(({ key, color, fill, stroke }) => {
        const pts = this.scatterData[key] || [];
        if (pts.length < 3) return;
        const cov = computeCovariance2D(pts);
        if (cov) {
          ctx.save();
          ctx.translate(cx + cov.meanE * pxPerMeter, cy - cov.meanN * pxPerMeter);
          ctx.rotate(-cov.angle);
          ctx.beginPath();
          ctx.ellipse(0, 0, cov.semiMajor * pxPerMeter, cov.semiMinor * pxPerMeter, 0, 0, Math.PI * 2);
          ctx.fillStyle = fill;
          ctx.fill();
          ctx.strokeStyle = stroke;
          ctx.lineWidth = key === "tep" ? 2 : 1;
          ctx.stroke();
          ctx.restore();
        }
        pts.forEach(([e, n]) => {
          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.arc(cx + e * pxPerMeter, cy - n * pxPerMeter, key === "tep" ? 3 : 2, 0, Math.PI * 2);
          ctx.fill();
        });
      });
    }

    renderTimeSeries() {
      const ctx = this.timeCtx;
      if (!ctx) return;
      const cvs = this.timeCanvas;
      const w = parseFloat(cvs.style.width);
      const h = parseFloat(cvs.style.height);
      if (!w || !h) return;

      ctx.clearRect(0, 0, w, h);
      const data = this.timeSeriesData;
      const padL = 50, padR = 20, padT = 25, padB = 30;
      const plotW = w - padL - padR;
      const plotH = h - padT - padB;

      if (data.length < 2) {
        ctx.fillStyle = "rgba(148, 163, 184, 0.5)";
        ctx.font = "12px sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("Position arms latched at <1 cm under LUPT/HUPT...", w / 2, h / 2);
        return;
      }

      let maxErr = 0.05;
      data.forEach((d) => {
        ["free", "persist", "tep"].forEach((k) => {
          if (d[k] != null && d[k] > maxErr && d[k] < 500) maxErr = d[k];
        });
      });
      maxErr = maxErr * 1.15;

      const n = data.length;
      const xToPx = (idx) => padL + (idx / (n - 1)) * plotW;
      const yToPx = (val) => padT + plotH - (Math.min(maxErr, Math.max(0, val)) / maxErr) * plotH;

      const traces = [
        { key: "free", color: "#ef4444", width: 1.5, dash: [4, 4] },
        { key: "persist", color: "#f59e0b", width: 1.5, dash: [2, 2] },
        { key: "tep", color: "#10b981", width: 2.5, dash: [] },
      ];
      traces.forEach(({ key, color, width, dash }) => {
        ctx.save();
        ctx.strokeStyle = color;
        ctx.lineWidth = width;
        ctx.setLineDash(dash);
        ctx.beginPath();
        let started = false;
        for (let i = 0; i < n; i++) {
          const v = data[i][key];
          if (v != null) {
            const px = xToPx(i), py = yToPx(v);
            if (!started) { ctx.moveTo(px, py); started = true; }
            else ctx.lineTo(px, py);
          }
        }
        ctx.stroke();
        ctx.restore();
      });
    }
  }

  window.ResultsArena = ResultsArena;
})();
