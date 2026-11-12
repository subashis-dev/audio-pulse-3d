/**
 * SONIX3D - Three.js WebGL Interactive 3D Audio Visualizer
 *
 * 3 Reactive Modes:
 * 1. Nebula - 10,000 dynamic orbital particles reacting to sub-bass & transients
 * 2. Terrain - Rolling 3D Spectrogram mountain landscape shifting forward
 * 3. Cymatics - Geodesic wireframe sphere with kinetic orbital resonance rings
 */

class Visualizer3D {
  constructor(canvasContainerId) {
    this.container = document.getElementById(canvasContainerId);
    this.mode = 'nebula'; // 'nebula', 'terrain', 'cymatics'
    this.theme = 'cyber'; // 'cyber', 'matrix', 'violet', 'solar'

    this.scene = null;
    this.camera = null;
    this.renderer = null;
    this.animationId = null;

    // Visualizer objects
    this.nebulaPoints = null;
    this.nebulaOriginalPos = null;
    this.terrainMesh = null;
    this.terrainGeometry = null;
    this.terrainRows = 50;
    this.terrainCols = 50;
    this.terrainHistory = [];

    this.cymaticsSphere = null;
    this.cymaticsSphereOriginal = null;
    this.cymaticsRings = [];

    // Interaction
    this.isDragging = false;
    this.prevMouse = { x: 0, y: 0 };
    this.rotation = { x: 0.2, y: 0 };
    this.targetRotation = { x: 0.2, y: 0 };
    this.cameraDistance = 75;

    // Theme color palettes
    this.themes = {
      cyber: { primary: 0x00f3ff, secondary: 0xff007f, bg: 0x070913 },
      matrix: { primary: 0x00ff66, secondary: 0x008833, bg: 0x030d07 },
      violet: { primary: 0xbd00ff, secondary: 0x00d2ff, bg: 0x090514 },
      solar: { primary: 0xffaa00, secondary: 0xff2200, bg: 0x120703 }
    };

    this.init();
  }

  init() {
    const width = this.container.clientWidth || window.innerWidth;
    const height = this.container.clientHeight || window.innerHeight;

    // 1. Scene
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(this.themes[this.theme].bg);
    this.scene.fog = new THREE.FogExp2(this.themes[this.theme].bg, 0.008);

    // 2. Camera
    this.camera = new THREE.PerspectiveCamera(55, width / height, 0.1, 1000);
    this.updateCameraPos();

    // 3. Renderer
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: "high-performance" });
    this.renderer.setSize(width, height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.container.appendChild(this.renderer.domElement);

    // 4. Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
    this.scene.add(ambientLight);
    const pointLight = new THREE.PointLight(this.themes[this.theme].primary, 2, 200);
    pointLight.position.set(0, 40, 50);
    this.scene.add(pointLight);

    // Build the visualizer scenes
    this.buildNebula();
    this.buildTerrain();
    this.buildCymatics();

    this.setMode('nebula');

    // Event listeners
    window.addEventListener('resize', () => this.onWindowResize());
    this.setupInteractivity();

    // Start render loop
    this.animate();
  }

  setupInteractivity() {
    const el = this.renderer.domElement;

    el.addEventListener('mousedown', (e) => {
      this.isDragging = true;
      this.prevMouse = { x: e.clientX, y: e.clientY };
    });

    window.addEventListener('mouseup', () => {
      this.isDragging = false;
    });

    el.addEventListener('mousemove', (e) => {
      if (!this.isDragging) return;
      const deltaX = e.clientX - this.prevMouse.x;
      const deltaY = e.clientY - this.prevMouse.y;
      this.targetRotation.y += deltaX * 0.008;
      this.targetRotation.x = Math.max(-Math.PI / 3, Math.min(Math.PI / 3, this.targetRotation.x + deltaY * 0.008));
      this.prevMouse = { x: e.clientX, y: e.clientY };
    });

    el.addEventListener('wheel', (e) => {
      e.preventDefault();
      this.cameraDistance = Math.max(30, Math.min(140, this.cameraDistance + e.deltaY * 0.05));
      this.updateCameraPos();
    }, { passive: false });

    // Touch support
    el.addEventListener('touchstart', (e) => {
      if (e.touches.length === 1) {
        this.isDragging = true;
        this.prevMouse = { x: e.touches[0].clientX, y: e.touches[0].clientY };
      }
    });

    el.addEventListener('touchmove', (e) => {
      if (!this.isDragging || e.touches.length !== 1) return;
      const deltaX = e.touches[0].clientX - this.prevMouse.x;
      const deltaY = e.touches[0].clientY - this.prevMouse.y;
      this.targetRotation.y += deltaX * 0.008;
      this.targetRotation.x = Math.max(-Math.PI / 3, Math.min(Math.PI / 3, this.targetRotation.x + deltaY * 0.008));
      this.prevMouse = { x: e.touches[0].clientX, y: e.touches[0].clientY };
    });

    el.addEventListener('touchend', () => {
      this.isDragging = false;
    });
  }

  updateCameraPos() {
    const x = this.cameraDistance * Math.sin(this.rotation.y) * Math.cos(this.rotation.x);
    const y = this.cameraDistance * Math.sin(this.rotation.x) + (this.mode === 'terrain' ? 25 : 0);
    const z = this.cameraDistance * Math.cos(this.rotation.y) * Math.cos(this.rotation.x);
    this.camera.position.set(x, y, z);
    this.camera.lookAt(0, (this.mode === 'terrain' ? 5 : 0), 0);
  }

  /* ----------------- Visualizer 1: Nebula ----------------- */
  buildNebula() {
    const particleCount = 8000;
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(particleCount * 3);
    const colors = new Float32Array(particleCount * 3);

    const c1 = new THREE.Color(this.themes[this.theme].primary);
    const c2 = new THREE.Color(this.themes[this.theme].secondary);

    for (let i = 0; i < particleCount; i++) {
      // Golden spiral galaxy distribution
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos((Math.random() * 2) - 1);
      const r = 10 + Math.pow(Math.random(), 1.6) * 35;

      const x = r * Math.sin(phi) * Math.cos(theta);
      const y = (r * Math.sin(phi) * Math.sin(theta)) * 0.5; // flatten disc
      const z = r * Math.cos(phi);

      positions[i * 3] = x;
      positions[i * 3 + 1] = y;
      positions[i * 3 + 2] = z;

      // Color gradient
      const mixRatio = Math.sin(r * 0.15) * 0.5 + 0.5;
      const c = c1.clone().lerp(c2, mixRatio);
      colors[i * 3] = c.r;
      colors[i * 3 + 1] = c.g;
      colors[i * 3 + 2] = c.b;
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    this.nebulaOriginalPos = new Float32Array(positions);

    const material = new THREE.PointsMaterial({
      size: 1.2,
      vertexColors: true,
      transparent: true,
      opacity: 0.85,
      blending: THREE.AdditiveBlending,
      depthWrite: false
    });

    this.nebulaPoints = new THREE.Points(geometry, material);
    this.scene.add(this.nebulaPoints);
  }

  /* ----------------- Visualizer 2: 3D Spectrogram Terrain ----------------- */
  buildTerrain() {
    const width = 80;
    const depth = 90;
    this.terrainGeometry = new THREE.PlaneGeometry(width, depth, this.terrainCols - 1, this.terrainRows - 1);
    this.terrainGeometry.rotateX(-Math.PI / 2);

    const material = new THREE.MeshBasicMaterial({
      color: this.themes[this.theme].primary,
      wireframe: true,
      transparent: true,
      opacity: 0.75
    });

    this.terrainMesh = new THREE.Mesh(this.terrainGeometry, material);
    this.terrainMesh.position.set(0, -10, 0);

    // Initialize terrain history
    for (let r = 0; r < this.terrainRows; r++) {
      this.terrainHistory.push(new Float32Array(this.terrainCols));
    }

    this.scene.add(this.terrainMesh);
  }

  /* ----------------- Visualizer 3: Cymatics Geometric Sphere ----------------- */
  buildCymatics() {
    this.cymaticsGroup = new THREE.Group();

    // Central Icosahedron
    const sphereGeo = new THREE.IcosahedronGeometry(18, 4);
    this.cymaticsSphereOriginal = new Float32Array(sphereGeo.attributes.position.array);

    const sphereMat = new THREE.MeshStandardMaterial({
      color: this.themes[this.theme].primary,
      wireframe: true,
      emissive: this.themes[this.theme].primary,
      emissiveIntensity: 0.4,
      roughness: 0.2
    });

    this.cymaticsSphere = new THREE.Mesh(sphereGeo, sphereMat);
    this.cymaticsGroup.add(this.cymaticsSphere);

    // Concentric orbital rings
    const ringRadii = [24, 30, 36];
    this.cymaticsRings = [];
    ringRadii.forEach((rad, idx) => {
      const ringGeo = new THREE.TorusGeometry(rad, 0.35, 12, 100);
      const ringMat = new THREE.MeshBasicMaterial({
        color: idx % 2 === 0 ? this.themes[this.theme].primary : this.themes[this.theme].secondary,
        transparent: true,
        opacity: 0.8
      });
      const ring = new THREE.Mesh(ringGeo, ringMat);
      ring.rotation.x = (Math.PI / 4) * (idx + 1);
      ring.rotation.y = (Math.PI / 6) * (idx + 1);
      this.cymaticsRings.push(ring);
      this.cymaticsGroup.add(ring);
    });

    this.scene.add(this.cymaticsGroup);
  }

  setMode(mode) {
    this.mode = mode;
    if (this.nebulaPoints) this.nebulaPoints.visible = (mode === 'nebula');
    if (this.terrainMesh) this.terrainMesh.visible = (mode === 'terrain');
    if (this.cymaticsGroup) this.cymaticsGroup.visible = (mode === 'cymatics');

    if (mode === 'terrain') {
      this.targetRotation.x = 0.45;
      this.cameraDistance = 85;
    } else {
      this.cameraDistance = 75;
    }
    this.updateCameraPos();
  }

  setTheme(themeName) {
    if (!this.themes[themeName]) return;
    this.theme = themeName;
    const theme = this.themes[themeName];

    this.scene.background = new THREE.Color(theme.bg);
    this.scene.fog.color = new THREE.Color(theme.bg);

    if (this.terrainMesh) {
      this.terrainMesh.material.color.setHex(theme.primary);
    }

    if (this.cymaticsSphere) {
      this.cymaticsSphere.material.color.setHex(theme.primary);
      this.cymaticsSphere.material.emissive.setHex(theme.primary);
    }
    if (this.cymaticsRings.length) {
      this.cymaticsRings.forEach((ring, i) => {
        ring.material.color.setHex(i % 2 === 0 ? theme.primary : theme.secondary);
      });
    }

    // Refresh nebula colors
    if (this.nebulaPoints) {
      const colors = this.nebulaPoints.geometry.attributes.color.array;
      const c1 = new THREE.Color(theme.primary);
      const c2 = new THREE.Color(theme.secondary);
      for (let i = 0; i < colors.length / 3; i++) {
        const mixRatio = Math.sin(i * 0.1) * 0.5 + 0.5;
        const c = c1.clone().lerp(c2, mixRatio);
        colors[i * 3] = c.r;
        colors[i * 3 + 1] = c.g;
        colors[i * 3 + 2] = c.b;
      }
      this.nebulaPoints.geometry.attributes.color.needsUpdate = true;
    }
  }

  animate() {
    this.animationId = requestAnimationFrame(() => this.animate());

    // Smooth camera damping
    this.rotation.x += (this.targetRotation.x - this.rotation.x) * 0.08;
    this.rotation.y += (this.targetRotation.y - this.rotation.y) * 0.08;
    this.updateCameraPos();

    // Query real-time audio metrics
    const audioData = window.SonixDsp ? window.SonixDsp.getRealtimeData() : null;
    const bands = audioData ? audioData.bands : { subBass: 0, bass: 0, mid: 0, highMid: 0, presence: 0 };
    const isPlaying = window.SonixDsp && window.SonixDsp.isPlaying;

    const bassImpact = Math.max(0.05, bands.subBass * 1.5 + bands.bass * 0.8);
    const midEnergy = Math.max(0.05, bands.mid * 1.2);
    const treble = Math.max(0.05, bands.presence * 1.0);

    /* 1. Nebula Update */
    if (this.mode === 'nebula' && this.nebulaPoints) {
      const positions = this.nebulaPoints.geometry.attributes.position.array;
      const count = positions.length / 3;
      const expansion = isPlaying ? (1.0 + bassImpact * 0.7) : 1.0;
      const pulseTime = performance.now() * 0.001;

      for (let i = 0; i < count; i++) {
        const ox = this.nebulaOriginalPos[i * 3];
        const oy = this.nebulaOriginalPos[i * 3 + 1];
        const oz = this.nebulaOriginalPos[i * 3 + 2];

        const harmonic = Math.sin(pulseTime * 2 + i * 0.05) * midEnergy * 3.5;
        positions[i * 3] = ox * expansion + (ox !== 0 ? Math.sign(ox) * harmonic : 0);
        positions[i * 3 + 1] = oy * expansion + Math.cos(pulseTime + i * 0.02) * treble * 2;
        positions[i * 3 + 2] = oz * expansion + harmonic;
      }
      this.nebulaPoints.geometry.attributes.position.needsUpdate = true;
      this.nebulaPoints.rotation.y += 0.002 + bassImpact * 0.005;
    }

    /* 2. Terrain Update */
    if (this.mode === 'terrain' && this.terrainMesh && audioData) {
      // Create new front row from real-time frequencies
      const newRow = new Float32Array(this.terrainCols);
      const freq = audioData.freq;
      const step = Math.floor(freq.length / this.terrainCols);

      for (let c = 0; c < this.terrainCols; c++) {
        const val = isPlaying ? (freq[c * step] / 255) : 0.02;
        // Apply bell curve weight towards center
        const centerWeight = Math.sin((c / this.terrainCols) * Math.PI);
        newRow[c] = Math.pow(val, 1.4) * 22 * centerWeight;
      }

      // Shift terrain rows
      this.terrainHistory.unshift(newRow);
      this.terrainHistory.pop();

      const pos = this.terrainGeometry.attributes.position.array;
      for (let r = 0; r < this.terrainRows; r++) {
        const rowData = this.terrainHistory[r];
        for (let c = 0; c < this.terrainCols; c++) {
          const idx = (r * this.terrainCols + c) * 3;
          pos[idx + 1] = rowData[c]; // Y displacement
        }
      }
      this.terrainGeometry.attributes.position.needsUpdate = true;
      this.terrainGeometry.computeVertexNormals();
    }

    /* 3. Cymatics Sphere Update */
    if (this.mode === 'cymatics' && this.cymaticsSphere) {
      const pos = this.cymaticsSphere.geometry.attributes.position.array;
      const orig = this.cymaticsSphereOriginal;
      const count = pos.length / 3;
      const time = performance.now() * 0.003;

      for (let i = 0; i < count; i++) {
        const ox = orig[i * 3];
        const oy = orig[i * 3 + 1];
        const oz = orig[i * 3 + 2];

        // Normal displacement based on harmonics
        const disp = 1.0 + (isPlaying ? (Math.sin(time + ox * 0.3) * Math.cos(time + oy * 0.3) * bassImpact * 0.5) : 0);
        pos[i * 3] = ox * disp;
        pos[i * 3 + 1] = oy * disp;
        pos[i * 3 + 2] = oz * disp;
      }
      this.cymaticsSphere.geometry.attributes.position.needsUpdate = true;
      this.cymaticsSphere.rotation.y += 0.005;
      this.cymaticsSphere.rotation.x += 0.003;

      // Orbit rings rotation
      this.cymaticsRings.forEach((ring, idx) => {
        const speed = (idx + 1) * 0.004 * (1 + treble * 2);
        ring.rotation.z += speed;
        ring.rotation.x += speed * 0.5;
        const scale = 1.0 + bassImpact * 0.15 * (idx === 0 ? 1.5 : 1);
        ring.scale.set(scale, scale, scale);
      });
    }

    this.renderer.render(this.scene, this.camera);
  }

  onWindowResize() {
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  }
}

window.Visualizer3D = Visualizer3D;
