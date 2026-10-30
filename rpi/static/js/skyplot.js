/**
 * skyplot.js — Polar Satellite Skyplot and Constellation Telemetry
 */

(function () {
  "use strict";

  class SkyplotViewer {
    constructor() {
      this.canvas = document.getElementById("skyplot-canvas");
      this.ctx = this.canvas ? this.canvas.getContext("2d") : null;
      this.tableBody = document.getElementById("sat-table-body");

      if (this.canvas) {
        this.resize();
        window.addEventListener("resize", () => this.resize());
      }
    }

    resize() {
      if (!this.canvas) return;
      const dpr = window.devicePixelRatio || 1;
      const rect = this.canvas.parentElement.getBoundingClientRect();
      const size = Math.min(rect.width, 320);
      this.canvas.width = size * dpr;
      this.canvas.height = size * dpr;
      this.canvas.style.width = size + "px";
      this.canvas.style.height = size + "px";
      this.ctx.resetTransform();
      this.ctx.scale(dpr, dpr);
    }

    update(satellites) {
      if (!satellites) return;
      this.renderSkyplot(satellites);
      this.renderTable(satellites);
    }

    renderSkyplot(sats) {
      const ctx = this.ctx;
      if (!ctx) return;
      const cvs = this.canvas;
      const w = parseFloat(cvs.style.width);
      const h = parseFloat(cvs.style.height);

      ctx.clearRect(0, 0, w, h);

      const cx = w / 2;
      const cy = h / 2;
      const maxR = Math.min(w, h) / 2 - 18;

      // Concentric elevation rings: 0°, 30°, 60°
      ctx.strokeStyle = "rgba(255, 255, 255, 0.1)";
      ctx.lineWidth = 1;

      [0, 30, 60].forEach((elev) => {
        const r = maxR * (1 - elev / 90);
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.stroke();

        ctx.fillStyle = "rgba(148, 163, 184, 0.5)";
        ctx.font = "9px monospace";
        ctx.fillText(elev + "°", cx + 4, cy - r + 10);
      });

      // Cardinal axes
      ctx.beginPath();
      ctx.moveTo(cx, cy - maxR);
      ctx.lineTo(cx, cy + maxR);
      ctx.moveTo(cx - maxR, cy);
      ctx.lineTo(cx + maxR, cy);
      ctx.stroke();

      ctx.fillStyle = "var(--text-bright)";
      ctx.font = "10px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("N", cx, cy - maxR - 4);
      ctx.fillText("S", cx, cy + maxR + 12);
      ctx.fillText("E", cx + maxR + 10, cy + 4);
      ctx.fillText("W", cx - maxR - 10, cy + 4);

      // Plot Satellites
      sats.forEach((sat) => {
        const azRad = (sat.azimuth * Math.PI) / 180;
        const elev = Math.max(0, sat.elevation);
        const r = maxR * (1 - elev / 90);

        const sx = cx + r * Math.sin(azRad);
        const sy = cy - r * Math.cos(azRad);

        const used = sat.used;
        const snr = sat.snr || 0;

        ctx.beginPath();
        ctx.arc(sx, sy, used ? 7 : 5, 0, Math.PI * 2);
        ctx.fillStyle = used ? "var(--tep-emerald)" : (snr > 0 ? "rgba(148, 163, 184, 0.6)" : "rgba(239, 68, 68, 0.4)");
        ctx.fill();

        if (used) {
          ctx.strokeStyle = "#ffffff";
          ctx.lineWidth = 1.5;
          ctx.stroke();
        }

        // PRN Label
        ctx.fillStyle = "#ffffff";
        ctx.font = "9px monospace";
        ctx.textAlign = "center";
        ctx.fillText(sat.prn, sx, sy - 8);
      });
    }

    renderTable(sats) {
      if (!this.tableBody) return;
      this.tableBody.innerHTML = "";

      sats.slice(0, 25).forEach((sat) => {
        const tr = document.createElement("tr");

        const snr = sat.snr || 0;
        const snrPct = Math.min(100, (snr / 50) * 100);

        tr.innerHTML = `
          <td><strong>${sat.prn}</strong></td>
          <td>${sat.elevation}°</td>
          <td>${sat.azimuth}°</td>
          <td>
            <div class="snr-meter">
              <div class="snr-fill" style="width: ${snrPct}%; background: ${sat.used ? "var(--tep-emerald)" : "var(--text-muted)"}"></div>
            </div>
            ${snr.toFixed(1)} dB
          </td>
          <td><span class="kpi-badge ${sat.used ? "badge-emerald" : "badge-amber"}">${sat.used ? "FIX" : "TRACK"}</span></td>
        `;
        this.tableBody.appendChild(tr);
      });
    }
  }

  window.SkyplotViewer = SkyplotViewer;
})();
