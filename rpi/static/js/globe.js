/**
 * globe.js — 3D Earth Globe and Satellite Constellation Orbit Visualizer
 */

(function () {
  "use strict";

  class GlobeViewer {
    constructor() {
      this.container = document.getElementById("globe-view");
      this.scene = null;
      this.camera = null;
      this.renderer = null;
      this.earth = null;
      this.satGroup = null;
      this.initialized = false;
      this.animId = null;
    }

    init() {
      if (!this.container || typeof THREE === "undefined") return;

      const w = this.container.clientWidth || 300;
      const h = this.container.clientHeight || 300;

      this.scene = new THREE.Scene();
      this.camera = new THREE.PerspectiveCamera(45, w / h, 0.1, 1000);
      this.camera.position.z = 2.8;

      this.renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
      this.renderer.setSize(w, h);
      this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      this.container.appendChild(this.renderer.domElement);

      // Earth Sphere with wireframe / dark stylization
      const earthGeo = new THREE.SphereGeometry(1, 32, 32);
      const earthMat = new THREE.MeshBasicMaterial({
        color: 0x0c192c,
        wireframe: true,
        transparent: true,
        opacity: 0.35,
      });
      this.earth = new THREE.Mesh(earthGeo, earthMat);
      this.scene.add(this.earth);

      // Atmospheric Glow
      const glowGeo = new THREE.SphereGeometry(1.02, 32, 32);
      const glowMat = new THREE.MeshBasicMaterial({
        color: 0x10b981,
        transparent: true,
        opacity: 0.08,
      });
      this.scene.add(new THREE.Mesh(glowGeo, glowMat));

      // Group for satellites
      this.satGroup = new THREE.Group();
      this.scene.add(this.satGroup);

      this.initialized = true;
      this.animate();

      window.addEventListener("resize", () => {
        if (!this.container || !this.renderer) return;
        const rw = this.container.clientWidth;
        const rh = this.container.clientHeight;
        this.camera.aspect = rw / rh;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(rw, rh);
      });
    }

    updateSatellites(sats) {
      if (!this.initialized) {
        this.init();
      }
      if (!this.satGroup || !sats) return;

      // Clear existing sat meshes
      while (this.satGroup.children.length > 0) {
        this.satGroup.remove(this.satGroup.children[0]);
      }

      const orbitRadius = 1.45;
      const satGeo = new THREE.SphereGeometry(0.025, 8, 8);

      sats.slice(0, 30).forEach((sat) => {
        const azRad = (sat.azimuth * Math.PI) / 180;
        const elRad = (sat.elevation * Math.PI) / 180;

        const x = orbitRadius * Math.cos(elRad) * Math.sin(azRad);
        const y = orbitRadius * Math.sin(elRad);
        const z = orbitRadius * Math.cos(elRad) * Math.cos(azRad);

        const satMat = new THREE.MeshBasicMaterial({
          color: sat.used ? 0x10b981 : 0x94a3b8,
        });

        const satMesh = new THREE.Mesh(satGeo, satMat);
        satMesh.position.set(x, y, z);
        this.satGroup.add(satMesh);
      });
    }

    animate() {
      this.animId = requestAnimationFrame(() => this.animate());
      if (this.earth) {
        this.earth.rotation.y += 0.0015;
      }
      if (this.satGroup) {
        this.satGroup.rotation.y += 0.0008;
      }
      if (this.renderer && this.scene && this.camera) {
        this.renderer.render(this.scene, this.camera);
      }
    }
  }

  window.GlobeViewer = GlobeViewer;
})();
