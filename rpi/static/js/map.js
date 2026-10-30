/**
 * map.js — Leaflet Trajectory and Geodetic Position Visualizer
 */

(function () {
  "use strict";

  class MapViewer {
    constructor() {
      this.mapEl = document.getElementById("map-view");
      this.map = null;
      this.marker = null;
      this.trackLine = null;
      this.trackCoords = [];
      this.initialized = false;
    }

    init(lat, lon) {
      if (!this.mapEl || typeof L === "undefined") return;

      this.map = L.map("map-view", {
        center: [lat, lon],
        zoom: 16,
        zoomControl: true,
        attributionControl: false,
      });

      // CartoDB Dark Matter tiles
      L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
        maxZoom: 20,
        subdomains: "abcd",
      }).addTo(this.map);

      // Pulse receiver marker icon
      const pulseIcon = L.divIcon({
        className: "custom-pulse-marker",
        html: `<div style="width:14px;height:14px;background:#10b981;border:2px solid #ffffff;border-radius:50%;box-shadow:0 0 12px #10b981;"></div>`,
        iconSize: [14, 14],
        iconAnchor: [7, 7],
      });

      this.marker = L.marker([lat, lon], { icon: pulseIcon }).addTo(this.map);

      this.trackLine = L.polyline([], {
        color: "#10b981",
        weight: 3,
        opacity: 0.85,
        smoothFactor: 1,
      }).addTo(this.map);

      this.initialized = true;
    }

    update(lat, lon) {
      if (!lat || !lon || isNaN(lat) || isNaN(lon)) return;

      if (!this.initialized) {
        this.init(lat, lon);
        return;
      }

      const newPos = [lat, lon];
      this.marker.setLatLng(newPos);

      this.trackCoords.push(newPos);
      if (this.trackCoords.length > 300) {
        this.trackCoords.shift();
      }
      this.trackLine.setLatLngs(this.trackCoords);
    }
  }

  window.MapViewer = MapViewer;
})();
