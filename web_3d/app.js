/**
 * AEGIS-3D BIOROBOTICS — SIMULATED WALKER DAMAGE & RECOVERY
 * High-Fidelity 3D WebGL Robotics & Neural Lesion Simulator
 * Features:
 *   - Multiple 3D Robot Models:
 *       1. X-BOT 3D Humanoid Android (Skeletal Rigged Character with GLTF animations)
 *       2. SPARK-E Bionic Automaton (Articulated Android with Expressive Locomotion)
 *       3. AEGIS-MK4 Titan (Heavy Combat Mech with Procedural Dual-Hydraulics)
 *   - Neural Lesion Injection (Actor Ablation: Neuron Kill, Weight Zero, Noise)
 *   - Neuroplastic Rehabilitation Progress Engine
 *   - Biomechanical Kinematic Telemetry HUD & Live Actuator Oscilloscope
 *   - Web Audio Robotic Servos & Alarm Synthesizer
 *
 * Author: Geo Mathew Joseph
 */

// ──────────────────────────────────────────────
// Simulation State
// ──────────────────────────────────────────────
const state = {
  isRunning: true,
  simSpeed: 1.0,
  currentRobot: "xbot",   // "xbot", "sparke", "aegis"
  policyMode: "standard", // "standard", "evolved"
  severity: 0.0,          // 0.0 to 1.0
  mode: "neuron_kill",    // "neuron_kill", "weight_zero", "noise"
  status: "healthy",      // "healthy", "damaged", "rehabilitating", "recovered"
  stepCount: 450,
  rehabProgress: 1.0,     // 0.0 to 1.0
  isRehabilitating: false,
  autoCam: false,
  showGhost: false,
  currentEnv: "hangar",   // "hangar", "cyber", "studio"
  showGantry: true,
  showMocap: true,
  showForcePlates: true,
  cameraMode: "orbit",    // "orbit", "chase", "sagittal", "top"
  showSynapse: true,
  showVectors: true,
  audioEnabled: true,
  time: 0,

  // Biomechanical Live Telemetry
  vx: 1.48,
  reward: 308.4,
  tilt: -1.2,
  symmetry: 98.4,
  torques: [0.05, 0.62, -0.45, -0.12, -0.05, -0.38, 0.82, 0.15], // 8 DOFs: hip_roll_l, hip_pitch_l, knee_l, ankle_l, hip_roll_r, hip_pitch_r, knee_r, ankle_r
  contacts: [true, false],             // legL, legR
};

// ──────────────────────────────────────────────
// Online Neural Plasticity & RL Engine Instance
// ──────────────────────────────────────────────
const neuralEngine = new window.NeuralPlasticityEngine();

// ──────────────────────────────────────────────
// Web Audio Synthesizer (Robotic Servos & Alarms)
// ──────────────────────────────────────────────
let audioCtx = null;
function initAudio() {
  if (!audioCtx) {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  }
}

function playServoSound(pitch = 120, duration = 0.08) {
  if (!state.audioEnabled || !audioCtx) return;
  try {
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = "sawtooth";
    osc.frequency.setValueAtTime(pitch, audioCtx.currentTime);
    osc.frequency.exponentialRampToValueAtTime(pitch * 0.7, audioCtx.currentTime + duration);
    gain.gain.setValueAtTime(0.015, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + duration);
    osc.connect(gain);
    gain.connect(audioCtx.destination);
    osc.start();
    osc.stop(audioCtx.currentTime + duration);
  } catch (e) {}
}

function playFootImpactSound() {
  if (!state.audioEnabled || !audioCtx) return;
  try {
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = "sine";
    osc.frequency.setValueAtTime(90, audioCtx.currentTime);
    osc.frequency.exponentialRampToValueAtTime(32, audioCtx.currentTime + 0.12);
    gain.gain.setValueAtTime(0.035, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.12);
    osc.connect(gain);
    gain.connect(audioCtx.destination);
    osc.start();
    osc.stop(audioCtx.currentTime + 0.12);
  } catch (e) {}
}

function playDamageSpark() {
  if (!state.audioEnabled || !audioCtx) return;
  try {
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = "square";
    osc.frequency.setValueAtTime(600, audioCtx.currentTime);
    osc.frequency.exponentialRampToValueAtTime(80, audioCtx.currentTime + 0.35);
    gain.gain.setValueAtTime(0.08, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.35);
    osc.connect(gain);
    gain.connect(audioCtx.destination);
    osc.start();
    osc.stop(audioCtx.currentTime + 0.35);
  } catch (e) {}
}

function playSuccessChime() {
  if (!state.audioEnabled || !audioCtx) return;
  try {
    [523.25, 659.25, 783.99, 1046.50].forEach((freq, i) => {
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      osc.type = "sine";
      osc.frequency.value = freq;
      gain.gain.setValueAtTime(0.04, audioCtx.currentTime + i * 0.09);
      gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + i * 0.09 + 0.25);
      osc.connect(gain);
      gain.connect(audioCtx.destination);
      osc.start(audioCtx.currentTime + i * 0.09);
      osc.stop(audioCtx.currentTime + i * 0.09 + 0.25);
    });
  } catch (e) {}
}

// ──────────────────────────────────────────────
// Three.js Scene & Camera Setup
// ──────────────────────────────────────────────
const container = document.getElementById("canvas-container");
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0c1017);
scene.fog = new THREE.FogExp2(0x0c1017, 0.024);

const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(0, 3.2, 8.2);

const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;
container.appendChild(renderer.domElement);

// WebGL Context Loss Error Boundary (IMP-11)
renderer.domElement.addEventListener("webglcontextlost", (e) => {
  e.preventDefault();
  console.warn("[AEGIS WebGL] GPU context lost. Pausing simulation...");
  const badge = document.getElementById("global-status-badge");
  if (badge) badge.className = "status-indicator damaged";
  const statusTxt = document.getElementById("status-text");
  if (statusTxt) statusTxt.textContent = "GPU Context Lost — Restoring...";
});
renderer.domElement.addEventListener("webglcontextrestored", () => {
  console.info("[AEGIS WebGL] GPU context restored. Refreshing pipeline...");
  location.reload();
});

const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.maxPolarAngle = Math.PI / 2 - 0.02; // Don't go below floor
controls.minDistance = 2.5;
controls.maxDistance = 25;
controls.target.set(0, 1.7, 0);

// ──────────────────────────────────────────────
// High-Fidelity PBR Textures & HDRI Environment Engine
// ──────────────────────────────────────────────
const textureLoader = new THREE.TextureLoader();

// 1. Industrial Hangar Concrete Floor Textures
const hangarDiff = textureLoader.load("textures/hangar_diff_1k.jpg");
const hangarNor = textureLoader.load("textures/hangar_nor_1k.jpg");
const hangarRough = textureLoader.load("textures/hangar_rough_1k.jpg");
[hangarDiff, hangarNor, hangarRough].forEach((tex) => {
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(12, 12);
});

// 2. High-Traction Rubberized Locomotion Runway Textures
const rubberDiff = textureLoader.load("textures/rubber_diff_1k.jpg");
const rubberNor = textureLoader.load("textures/rubber_nor_1k.jpg");
const rubberRough = textureLoader.load("textures/rubber_rough_1k.jpg");
[rubberDiff, rubberNor, rubberRough].forEach((tex) => {
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(2, 18);
});

// 3. HDRI Image-Based Lighting (IBL) Pipeline
const pmremGenerator = new THREE.PMREMGenerator(renderer);
pmremGenerator.compileEquirectangularShader();
const rgbeLoader = new THREE.RGBELoader();

const envMaps = {};
function loadHdrEnv(name, filePath, onLoaded) {
  rgbeLoader.load(
    filePath,
    (texture) => {
      const envMap = pmremGenerator.fromEquirectangular(texture).texture;
      envMaps[name] = envMap;
      texture.dispose();
      console.info(`[AEGIS IBL] Loaded 32-bit HDRI Environment: ${name}`);
      if (onLoaded) onLoaded(envMap);
    },
    undefined,
    (err) => console.warn(`[AEGIS IBL] Could not load HDR ${filePath}:`, err)
  );
}

// Load default hangar HDRI & studio HDRI
loadHdrEnv("hangar", "textures/aerodynamics_workshop_1k.hdr", (envMap) => {
  if (state.currentEnv === "hangar") {
    scene.environment = envMap;
  }
});
loadHdrEnv("studio", "textures/blue_photo_studio_1k.hdr");

// ──────────────────────────────────────────────
// Lighting Setup (Cinematic Studio & Industrial Lighting)
// ──────────────────────────────────────────────
const ambientLight = new THREE.AmbientLight(0x475569, 0.9);
scene.add(ambientLight);

// Primary High-Bay Softbox Spotlight (Key Light)
const mainSpot = new THREE.SpotLight(0xffffff, 2.5);
mainSpot.position.set(4, 13, 5);
mainSpot.angle = Math.PI / 4.2;
mainSpot.penumbra = 0.65;
mainSpot.castShadow = true;
mainSpot.shadow.mapSize.width = 2048;
mainSpot.shadow.mapSize.height = 2048;
mainSpot.shadow.camera.near = 2.0;
mainSpot.shadow.camera.far = 25.0;
mainSpot.shadow.bias = -0.00008;
scene.add(mainSpot);

// Cool-Blue Edge Kicker Light (Rim Light)
const rimLight = new THREE.DirectionalLight(0x38bdf8, 0.85);
rimLight.position.set(-6, 9, -7);
scene.add(rimLight);

// Warm-Amber Fill Light
const warmFill = new THREE.DirectionalLight(0xf59e0b, 0.45);
warmFill.position.set(7, 6, -5);
scene.add(warmFill);

// Ground Contact Accent Fill
const groundAccent = new THREE.PointLight(0x0284c7, 0.35, 15);
groundAccent.position.set(0, 0.35, 0);
scene.add(groundAccent);

// ──────────────────────────────────────────────
// Photorealistic Dual-Layer Ground Architecture
// ──────────────────────────────────────────────
const envGroup = new THREE.Group();
scene.add(envGroup);

// 1. Industrial Hangar Concrete Sub-Floor
const floorGeo = new THREE.PlaneGeometry(80, 80);
const floorMat = new THREE.MeshStandardMaterial({
  map: hangarDiff,
  normalMap: hangarNor,
  roughnessMap: hangarRough,
  roughness: 0.65,
  metalness: 0.15,
});
const floorMesh = new THREE.Mesh(floorGeo, floorMat);
floorMesh.rotation.x = -Math.PI / 2;
floorMesh.receiveShadow = true;
envGroup.add(floorMesh);

// 2. High-Traction Locomotion Testing Runway
const runwayGeo = new THREE.BoxGeometry(4.2, 0.05, 36);
const runwayMat = new THREE.MeshStandardMaterial({
  map: rubberDiff,
  normalMap: rubberNor,
  roughnessMap: rubberRough,
  roughness: 0.82,
  metalness: 0.08,
});
const runwayMesh = new THREE.Mesh(runwayGeo, runwayMat);
runwayMesh.position.set(0, 0.025, 0);
runwayMesh.receiveShadow = true;
envGroup.add(runwayMesh);

// Runway Luminescent Border Curbs (Left & Right)
const curbGeo = new THREE.BoxGeometry(0.08, 0.07, 36);
const curbMat = new THREE.MeshStandardMaterial({
  color: 0x0284c7,
  emissive: 0x0369a1,
  emissiveIntensity: 0.6,
  roughness: 0.2,
  metalness: 0.8,
});
const curbL = new THREE.Mesh(curbGeo, curbMat);
curbL.position.set(-2.14, 0.035, 0);
envGroup.add(curbL);

const curbR = new THREE.Mesh(curbGeo, curbMat);
curbR.position.set(2.14, 0.035, 0);
envGroup.add(curbR);

// 3. Runway Metric Calibration Canvas Texture
function createRunwayMarkingsTexture() {
  const canvas = document.createElement("canvas");
  canvas.width = 512;
  canvas.height = 2048;
  const ctx = canvas.getContext("2d");

  ctx.fillStyle = "rgba(0, 0, 0, 0)";
  ctx.fillRect(0, 0, 512, 2048);

  ctx.fillStyle = "#ffffff";
  ctx.font = "bold 26px 'JetBrains Mono', monospace";
  ctx.textAlign = "center";

  // Center alignment dashed line
  ctx.strokeStyle = "rgba(255, 255, 255, 0.35)";
  ctx.lineWidth = 3;
  ctx.setLineDash([18, 18]);
  ctx.beginPath();
  ctx.moveTo(256, 0);
  ctx.lineTo(256, 2048);
  ctx.stroke();

  // Distance markings every 2 meters
  const labels = [
    { text: "+8.0 m", y: 200 },
    { text: "+6.0 m", y: 400 },
    { text: "+4.0 m", y: 600 },
    { text: "+2.0 m", y: 800 },
    { text: "ORIGIN 0.0 m", y: 1024, origin: true },
    { text: "-2.0 m", y: 1248 },
    { text: "-4.0 m", y: 1448 },
    { text: "-6.0 m", y: 1648 },
    { text: "-8.0 m", y: 1848 },
  ];

  ctx.setLineDash([]);
  labels.forEach((lbl) => {
    ctx.strokeStyle = lbl.origin ? "rgba(56, 189, 248, 0.85)" : "rgba(255, 255, 255, 0.3)";
    ctx.lineWidth = lbl.origin ? 5 : 2;
    ctx.beginPath();
    ctx.moveTo(70, lbl.y);
    ctx.lineTo(442, lbl.y);
    ctx.stroke();

    ctx.fillStyle = lbl.origin ? "#38bdf8" : "#94a3b8";
    ctx.fillText(lbl.text, 256, lbl.y - 12);
  });

  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = THREE.ClampToEdgeWrapping;
  texture.wrapT = THREE.ClampToEdgeWrapping;
  return texture;
}

const markingsGeo = new THREE.PlaneGeometry(3.6, 34);
const markingsMat = new THREE.MeshBasicMaterial({
  map: createRunwayMarkingsTexture(),
  transparent: true,
  opacity: 0.85,
  depthWrite: false,
});
const markingsMesh = new THREE.Mesh(markingsGeo, markingsMat);
markingsMesh.rotation.x = -Math.PI / 2;
markingsMesh.position.set(0, 0.052, 0);
envGroup.add(markingsMesh);

// 4. Embedded Ground Reaction Force (GRF) Force Plates
const forcePlateGroup = new THREE.Group();
envGroup.add(forcePlateGroup);

const fpGeo = new THREE.BoxGeometry(0.72, 0.015, 1.25);
const fpMatL = new THREE.MeshStandardMaterial({
  color: 0x1e293b,
  emissive: 0x0284c7,
  emissiveIntensity: 0.1,
  roughness: 0.3,
  metalness: 0.8,
});
const fpMatR = new THREE.MeshStandardMaterial({
  color: 0x1e293b,
  emissive: 0x0284c7,
  emissiveIntensity: 0.1,
  roughness: 0.3,
  metalness: 0.8,
});

const forcePlateL = new THREE.Mesh(fpGeo, fpMatL);
forcePlateL.position.set(-0.46, 0.053, 0);
forcePlateGroup.add(forcePlateL);

const forcePlateR = new THREE.Mesh(fpGeo, fpMatR);
forcePlateR.position.set(0.46, 0.053, 0);
forcePlateGroup.add(forcePlateR);

// 5. Dynamic Foot-Strike Concentric Ripple Rings
const shockwaveGroup = new THREE.Group();
envGroup.add(shockwaveGroup);

const rippleGeo = new THREE.RingGeometry(0.08, 0.16, 32);
const ripplePool = [];
for (let i = 0; i < 4; i++) {
  const rMat = new THREE.MeshBasicMaterial({
    color: 0x38bdf8,
    transparent: true,
    opacity: 0.0,
    side: THREE.DoubleSide,
    depthWrite: false,
  });
  const rMesh = new THREE.Mesh(rippleGeo, rMat);
  rMesh.rotation.x = -Math.PI / 2;
  rMesh.position.y = 0.055;
  rMesh.visible = false;
  shockwaveGroup.add(rMesh);
  ripplePool.push({ mesh: rMesh, mat: rMat, life: 0, maxLife: 0.45 });
}

function triggerGroundShockwave(x, z) {
  const freeRipple = ripplePool.find((r) => r.life <= 0);
  if (freeRipple) {
    freeRipple.mesh.position.set(x, 0.055, z);
    freeRipple.mesh.scale.set(0.5, 0.5, 0.5);
    freeRipple.mesh.visible = true;
    freeRipple.mat.opacity = 0.85;
    freeRipple.life = freeRipple.maxLife;
  }
}

// 6. Overhead Safety Gantry Rig & Motorized Trolley
const gantryGroup = new THREE.Group();
envGroup.add(gantryGroup);

const beamGeo = new THREE.BoxGeometry(0.24, 0.38, 38);
const beamMat = new THREE.MeshStandardMaterial({
  color: 0x1e293b,
  roughness: 0.35,
  metalness: 0.85,
});

const beamL = new THREE.Mesh(beamGeo, beamMat);
beamL.position.set(-1.8, 6.8, 0);
gantryGroup.add(beamL);

const beamR = new THREE.Mesh(beamGeo, beamMat);
beamR.position.set(1.8, 6.8, 0);
gantryGroup.add(beamR);

// Structural Columns & Overhead Cross Trusses
const columnGeo = new THREE.BoxGeometry(0.22, 7.0, 0.22);
[-16, -8, 0, 8, 16].forEach((zPos) => {
  const colL = new THREE.Mesh(columnGeo, beamMat);
  colL.position.set(-3.2, 3.5, zPos);
  gantryGroup.add(colL);

  const colR = new THREE.Mesh(columnGeo, beamMat);
  colR.position.set(3.2, 3.5, zPos);
  gantryGroup.add(colR);

  const crossGeo = new THREE.BoxGeometry(6.64, 0.22, 0.22);
  const crossMesh = new THREE.Mesh(crossGeo, beamMat);
  crossMesh.position.set(0, 6.9, zPos);
  gantryGroup.add(crossMesh);
});

// Motorized Gantry Carriage Trolley
const trolleyGeo = new THREE.BoxGeometry(3.8, 0.28, 0.9);
const trolleyMat = new THREE.MeshStandardMaterial({
  color: 0x0f172a,
  roughness: 0.25,
  metalness: 0.9,
});
const trolleyMesh = new THREE.Mesh(trolleyGeo, trolleyMat);
trolleyMesh.position.set(0, 6.7, 0);
gantryGroup.add(trolleyMesh);

// Pneumatic Damper & Safety Cable
const tetherCylinderGeo = new THREE.CylinderGeometry(0.08, 0.08, 1.2, 16);
const damperMat = new THREE.MeshStandardMaterial({
  color: 0x38bdf8,
  metalness: 0.9,
  roughness: 0.2,
});
const damperMesh = new THREE.Mesh(tetherCylinderGeo, damperMat);
damperMesh.position.set(0, 5.8, 0);
gantryGroup.add(damperMesh);

const cableGeo = new THREE.CylinderGeometry(0.012, 0.012, 3.2, 8);
const cableMat = new THREE.MeshStandardMaterial({
  color: 0x0284c7,
  metalness: 0.95,
  roughness: 0.1,
});
const cableMesh = new THREE.Mesh(cableGeo, cableMat);
cableMesh.position.set(0, 4.0, 0);
gantryGroup.add(cableMesh);

// 7. Perimeter Motion Capture (MoCap) Optical Towers
const mocapGroup = new THREE.Group();
envGroup.add(mocapGroup);

const cameraTowerPositions = [
  { x: -3.8, z: -12 },
  { x: 3.8, z: -12 },
  { x: -4.2, z: -4 },
  { x: 4.2, z: -4 },
  { x: -4.2, z: 4 },
  { x: 4.2, z: 4 },
  { x: -3.8, z: 12 },
  { x: 3.8, z: 12 },
];

const camBodyGeo = new THREE.BoxGeometry(0.24, 0.18, 0.28);
const camBodyMat = new THREE.MeshStandardMaterial({
  color: 0x090d16,
  roughness: 0.3,
  metalness: 0.7,
});
const irRingGeo = new THREE.RingGeometry(0.04, 0.075, 20);
const irRingMat = new THREE.MeshBasicMaterial({
  color: 0xef4444,
  side: THREE.DoubleSide,
});

cameraTowerPositions.forEach((pos) => {
  const poleGeo = new THREE.CylinderGeometry(0.035, 0.035, 3.2, 12);
  const poleMesh = new THREE.Mesh(poleGeo, beamMat);
  poleMesh.position.set(pos.x, 1.6, pos.z);
  mocapGroup.add(poleMesh);

  const camHead = new THREE.Group();
  camHead.position.set(pos.x, 3.1, pos.z);

  const camBody = new THREE.Mesh(camBodyGeo, camBodyMat);
  camHead.add(camBody);

  const irRing = new THREE.Mesh(irRingGeo, irRingMat);
  irRing.position.set(0, 0, 0.145);
  camHead.add(irRing);

  camHead.lookAt(0, 1.6, 0);
  mocapGroup.add(camHead);
});

// 8. Atmospheric Ambient Floating Particles (Air Motes)
const moteCount = 200;
const moteGeo = new THREE.BufferGeometry();
const motePositions = new Float32Array(moteCount * 3);
for (let i = 0; i < moteCount; i++) {
  motePositions[i * 3] = (Math.random() - 0.5) * 14;
  motePositions[i * 3 + 1] = 0.5 + Math.random() * 6.5;
  motePositions[i * 3 + 2] = (Math.random() - 0.5) * 20;
}
moteGeo.setAttribute("position", new THREE.BufferAttribute(motePositions, 3));
const moteMat = new THREE.PointsMaterial({
  color: 0x94a3b8,
  size: 0.045,
  transparent: true,
  opacity: 0.4,
});
const motes = new THREE.Points(moteGeo, moteMat);
envGroup.add(motes);

// ──────────────────────────────────────────────
// Robot Materials & Surface Shaders
// ──────────────────────────────────────────────
const metalMatDark = new THREE.MeshStandardMaterial({
  color: 0x151d2c,
  roughness: 0.28,
  metalness: 0.88,
});

const metalMatArmor = new THREE.MeshStandardMaterial({
  color: 0x2e3d52,
  roughness: 0.22,
  metalness: 0.92,
});

const chromeMat = new THREE.MeshStandardMaterial({
  color: 0xe2e8f0,
  roughness: 0.08,
  metalness: 0.98,
});

const cyanGlowMat = new THREE.MeshStandardMaterial({
  color: 0x38bdf8,
  emissive: 0x0284c7,
  emissiveIntensity: 0.45,
  roughness: 0.3,
  metalness: 0.3,
});

const redGlowMat = new THREE.MeshStandardMaterial({
  color: 0xdc2626,
  emissive: 0x991b1b,
  emissiveIntensity: 0.45,
  roughness: 0.3,
  metalness: 0.3,
});

const goldGlowMat = new THREE.MeshStandardMaterial({
  color: 0xd97706,
  emissive: 0xb45309,
  emissiveIntensity: 0.45,
  roughness: 0.3,
  metalness: 0.3,
});

// ──────────────────────────────────────────────
// Master Robot Groups Container
// ──────────────────────────────────────────────
const robotGroup = new THREE.Group();
scene.add(robotGroup);

// 1. X-BOT Humanoid Rig Group
const xbotGroup = new THREE.Group();
robotGroup.add(xbotGroup);
let xbotMixer = null;
let xbotActions = {};
let xbotCurrentAction = null;
let xbotModel = null;
let xbotLeftLegBone = null;

// 2. SPARK-E Bionic Android Group
const sparkeGroup = new THREE.Group();
robotGroup.add(sparkeGroup);
sparkeGroup.visible = false;
let sparkeMixer = null;
let sparkeActions = {};
let sparkeCurrentAction = null;
let sparkeModel = null;

// 3. AEGIS Humanoid Mech Group
const aegisGroup = new THREE.Group();
robotGroup.add(aegisGroup);
aegisGroup.visible = false;

// ──────────────────────────────────────────────
// GLTF Loader for High-Fidelity 3D Models
// ──────────────────────────────────────────────
const gltfLoader = new THREE.GLTFLoader();

// Load Model 1: X-BOT Humanoid Android
gltfLoader.load(
  "models/Xbot.glb",
  (gltf) => {
    xbotModel = gltf.scene;
    xbotModel.scale.set(1.4, 1.4, 1.4);
    xbotModel.position.set(0, 0, 0);

    xbotModel.traverse((child) => {
      if (child.isMesh) {
        child.castShadow = true;
        child.receiveShadow = true;
        if (child.isSkinnedMesh && child.material) {
          child.material.skinning = true;
        }
      }
      if (child.isBone && child.name.includes("LeftLeg")) {
        xbotLeftLegBone = child;
      }
    });

    xbotGroup.add(xbotModel);
    xbotMixer = new THREE.AnimationMixer(xbotModel);

    gltf.animations.forEach((clip) => {
      const name = clip.name.toLowerCase();
      const action = xbotMixer.clipAction(clip);
      xbotActions[name] = action;
    });

    // Start with walking animation
    if (xbotActions["walk"]) {
      xbotActions["walk"].play();
      xbotCurrentAction = xbotActions["walk"];
    }

    console.log("X-BOT 3D Humanoid Loaded. Available actions:", Object.keys(xbotActions));
  },
  undefined,
  (err) => console.warn("X-BOT GLTF Load Warning:", err)
);

// Load Model 2: SPARK-E Bionic Automaton
gltfLoader.load(
  "models/RobotExpressive.glb",
  (gltf) => {
    sparkeModel = gltf.scene;
    sparkeModel.scale.set(0.65, 0.65, 0.65);
    sparkeModel.position.set(0, 0, 0);

    sparkeModel.traverse((child) => {
      if (child.isMesh) {
        child.castShadow = true;
        child.receiveShadow = true;
      }
    });

    sparkeGroup.add(sparkeModel);
    sparkeMixer = new THREE.AnimationMixer(sparkeModel);

    gltf.animations.forEach((clip) => {
      const name = clip.name.toLowerCase();
      const action = sparkeMixer.clipAction(clip);
      sparkeActions[name] = action;
    });

    if (sparkeActions["walking"]) {
      sparkeActions["walking"].play();
      sparkeCurrentAction = sparkeActions["walking"];
    }

    console.log("SPARK-E 3D Automaton Loaded. Available actions:", Object.keys(sparkeActions));
  },
  undefined,
  (err) => console.warn("SPARK-E GLTF Load Warning:", err)
);

// ──────────────────────────────────────────────
// Procedural AEGIS Humanoid Cyber-Android Construction
// ──────────────────────────────────────────────
const torsoGroup = new THREE.Group();
torsoGroup.position.y = 2.4;
aegisGroup.add(torsoGroup);

// 1. Humanoid Chest & Ribcage (Athletic V-Taper Silhouette)
const chestGeo = new THREE.BoxGeometry(0.68, 0.46, 0.38);
const chestMesh = new THREE.Mesh(chestGeo, metalMatArmor);
chestMesh.position.set(0, 0.22, 0);
chestMesh.castShadow = true;
torsoGroup.add(chestMesh);

// Sculpted Left & Right Pectoral Armor Plates
[-0.18, 0.18].forEach((px) => {
  const pecGeo = new THREE.BoxGeometry(0.28, 0.22, 0.08);
  const pec = new THREE.Mesh(pecGeo, metalMatDark);
  pec.position.set(px, 0.24, 0.21);
  pec.rotation.y = px > 0 ? -0.08 : 0.08;
  torsoGroup.add(pec);
});

// Central Glowing Arc Reactor Power Core
const coreRingGeo = new THREE.CylinderGeometry(0.11, 0.11, 0.04, 24);
coreRingGeo.rotateX(Math.PI / 2);
const coreRing = new THREE.Mesh(coreRingGeo, chromeMat);
coreRing.position.set(0, 0.24, 0.21);
torsoGroup.add(coreRing);

const eyeGeo = new THREE.SphereGeometry(0.08, 16, 16);
const eyeMesh = new THREE.Mesh(eyeGeo, cyanGlowMat);
eyeMesh.position.set(0, 0.24, 0.22);
torsoGroup.add(eyeMesh);

// 2. Abdominal Core & Cybernetic Spine
const abGeo = new THREE.BoxGeometry(0.48, 0.34, 0.32);
const abMesh = new THREE.Mesh(abGeo, metalMatDark);
abMesh.position.set(0, -0.14, 0);
abMesh.castShadow = true;
torsoGroup.add(abMesh);

[-0.08, -0.18, -0.26].forEach((ay, i) => {
  const plateGeo = new THREE.BoxGeometry(0.36 - i * 0.04, 0.06, 0.05);
  const plate = new THREE.Mesh(plateGeo, chromeMat);
  plate.position.set(0, ay, 0.17);
  torsoGroup.add(plate);
});

// Spinal Column Vertebrae (Back)
[0.35, 0.20, 0.05, -0.10, -0.25].forEach((sy) => {
  const vertGeo = new THREE.BoxGeometry(0.12, 0.08, 0.10);
  const vert = new THREE.Mesh(vertGeo, chromeMat);
  vert.position.set(0, sy, -0.21);
  torsoGroup.add(vert);
});

// 3. Humanoid Neck & Articulated Cranium
const neckGeo = new THREE.CylinderGeometry(0.09, 0.11, 0.18, 16);
const neckMesh = new THREE.Mesh(neckGeo, metalMatDark);
neckMesh.position.set(0, 0.50, 0);
torsoGroup.add(neckMesh);

[-0.09, 0.09].forEach((nx) => {
  const ndGeo = new THREE.CylinderGeometry(0.02, 0.02, 0.16, 8);
  const nd = new THREE.Mesh(ndGeo, chromeMat);
  nd.position.set(nx, 0.50, -0.04);
  torsoGroup.add(nd);
});

const headGroup = new THREE.Group();
headGroup.position.set(0, 0.68, 0.02);
torsoGroup.add(headGroup);

const craniumGeo = new THREE.BoxGeometry(0.24, 0.26, 0.28);
const craniumMesh = new THREE.Mesh(craniumGeo, metalMatArmor);
craniumMesh.castShadow = true;
headGroup.add(craniumMesh);

const chinGeo = new THREE.BoxGeometry(0.16, 0.10, 0.18);
const chinMesh = new THREE.Mesh(chinGeo, metalMatDark);
chinMesh.position.set(0, -0.10, 0.06);
headGroup.add(chinMesh);

// Glowing Cyan Visor Eye-Band
const visorGeo = new THREE.BoxGeometry(0.22, 0.055, 0.12);
const visorMesh = new THREE.Mesh(visorGeo, cyanGlowMat);
visorMesh.position.set(0, 0.02, 0.13);
headGroup.add(visorMesh);

[-0.05, 0.05].forEach((ex) => {
  const lens = new THREE.Mesh(new THREE.CylinderGeometry(0.02, 0.02, 0.03, 12), chromeMat);
  lens.rotateX(Math.PI / 2);
  lens.position.set(ex, 0.02, 0.19);
  headGroup.add(lens);
});

[-0.13, 0.13].forEach((tx) => {
  const earGeo = new THREE.CylinderGeometry(0.045, 0.045, 0.04, 16);
  earGeo.rotateZ(Math.PI / 2);
  const ear = new THREE.Mesh(earGeo, chromeMat);
  ear.position.set(tx, 0.02, -0.02);
  headGroup.add(ear);
});

const crestGeo = new THREE.BoxGeometry(0.06, 0.04, 0.24);
const crest = new THREE.Mesh(crestGeo, chromeMat);
crest.position.set(0, 0.14, -0.01);
headGroup.add(crest);

// 4. Pelvis Gimbal (Base for Legs)
const pelvisGeo = new THREE.CylinderGeometry(0.28, 0.24, 0.22, 16);
pelvisGeo.rotateZ(Math.PI / 2);
const pelvisMesh = new THREE.Mesh(pelvisGeo, metalMatDark);
pelvisMesh.position.y = -0.38;
torsoGroup.add(pelvisMesh);

// 5. Articulated Humanoid Arms & Hands
function createAegisArm(isLeft = true) {
  const arm = {
    group: new THREE.Group(),
    shoulder: new THREE.Group(),
    bicep: null,
    elbow: new THREE.Group(),
    forearm: null,
    hand: new THREE.Group(),
  };

  const xPos = isLeft ? -0.46 : 0.46;
  arm.group.position.set(xPos, 0.32, 0);
  torsoGroup.add(arm.group);

  // Shoulder Ball Joint
  const shoulderBallGeo = new THREE.SphereGeometry(0.10, 16, 16);
  const shoulderBall = new THREE.Mesh(shoulderBallGeo, metalMatDark);
  arm.shoulder.add(shoulderBall);

  // Deltoid Armor Cap
  const deltoidGeo = new THREE.SphereGeometry(0.13, 16, 12, 0, Math.PI * 2, 0, Math.PI * 0.65);
  const deltoid = new THREE.Mesh(deltoidGeo, metalMatArmor);
  deltoid.position.set(isLeft ? -0.04 : 0.04, 0.02, 0);
  deltoid.rotation.z = isLeft ? 0.35 : -0.35;
  arm.shoulder.add(deltoid);

  // Bicep Armature with Chrome Hydraulic Piston
  const bicepGeo = new THREE.BoxGeometry(0.11, 0.42, 0.13);
  const bicepMesh = new THREE.Mesh(bicepGeo, metalMatDark);
  bicepMesh.position.set(0, -0.22, 0);
  bicepMesh.castShadow = true;
  arm.shoulder.add(bicepMesh);

  const bPiston = new THREE.Mesh(new THREE.CylinderGeometry(0.022, 0.022, 0.36, 12), chromeMat);
  bPiston.position.set(0, -0.20, 0.07);
  arm.shoulder.add(bPiston);

  arm.group.add(arm.shoulder);

  // Elbow Joint Pivot
  arm.elbow.position.set(0, -0.44, 0);
  arm.shoulder.add(arm.elbow);

  const elbowPivotGeo = new THREE.CylinderGeometry(0.065, 0.065, 0.13, 14);
  elbowPivotGeo.rotateZ(Math.PI / 2);
  const elbowPivot = new THREE.Mesh(elbowPivotGeo, chromeMat);
  arm.elbow.add(elbowPivot);

  const elbowGlowRing = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.04, 0.14, 12), cyanGlowMat);
  elbowGlowRing.rotateZ(Math.PI / 2);
  arm.elbow.add(elbowGlowRing);

  // Forearm Armature
  const forearmGeo = new THREE.BoxGeometry(0.10, 0.38, 0.11);
  const forearmMesh = new THREE.Mesh(forearmGeo, metalMatArmor);
  forearmMesh.position.set(0, -0.20, 0);
  forearmMesh.castShadow = true;
  arm.elbow.add(forearmMesh);

  // Wrist Gimbal
  arm.hand.position.set(0, -0.40, 0);
  arm.elbow.add(arm.hand);

  const wristGeo = new THREE.CylinderGeometry(0.045, 0.045, 0.06, 12);
  const wrist = new THREE.Mesh(wristGeo, chromeMat);
  arm.hand.add(wrist);

  // Hand Palm & Articulated Humanoid Fingers
  const palmGeo = new THREE.BoxGeometry(0.08, 0.10, 0.045);
  const palm = new THREE.Mesh(palmGeo, metalMatDark);
  palm.position.set(0, -0.06, 0);
  arm.hand.add(palm);

  [-0.027, -0.009, 0.009, 0.027].forEach((fx) => {
    const fingerGeo = new THREE.BoxGeometry(0.015, 0.07, 0.025);
    const finger = new THREE.Mesh(fingerGeo, chromeMat);
    finger.position.set(fx, -0.13, 0.01);
    arm.hand.add(finger);
  });

  const thumbGeo = new THREE.BoxGeometry(0.018, 0.05, 0.025);
  const thumb = new THREE.Mesh(thumbGeo, chromeMat);
  thumb.position.set(isLeft ? 0.042 : -0.042, -0.08, 0.03);
  thumb.rotation.z = isLeft ? -0.45 : 0.45;
  arm.hand.add(thumb);

  return arm;
}

const armL = createAegisArm(true);
const armR = createAegisArm(false);

// Articulated AEGIS Legs
function createAegisLeg(isLeft = true) {
  const leg = {
    group: new THREE.Group(),
    hip: new THREE.Group(),
    thighPiston: null,
    knee: new THREE.Group(),
    kneeCap: null,
    shin: new THREE.Group(),
    foot: new THREE.Group(),
    sparkEmitter: null,
  };

  const xOffset = isLeft ? -0.45 : 0.45;
  leg.group.position.set(xOffset, -0.45, 0);
  torsoGroup.add(leg.group);

  // Hip Joint Motor
  const hipMotorGeo = new THREE.CylinderGeometry(0.22, 0.22, 0.28, 16);
  hipMotorGeo.rotateZ(Math.PI / 2);
  const hipMotor = new THREE.Mesh(hipMotorGeo, metalMatDark);
  leg.hip.add(hipMotor);
  leg.group.add(leg.hip);

  // Thigh Armature
  const thighGeo = new THREE.BoxGeometry(0.18, 0.95, 0.24);
  const thighMesh = new THREE.Mesh(thighGeo, metalMatArmor);
  thighMesh.position.set(0, -0.48, 0);
  thighMesh.castShadow = true;
  leg.hip.add(thighMesh);

  // Hydraulic Chrome Piston
  const pistonGeo = new THREE.CylinderGeometry(0.04, 0.04, 0.8, 12);
  const pistonMesh = new THREE.Mesh(pistonGeo, chromeMat);
  pistonMesh.position.set(0, -0.45, 0.14);
  leg.hip.add(pistonMesh);
  leg.thighPiston = pistonMesh;

  // Knee Joint Pivot
  leg.knee.position.set(0, -0.95, 0);
  leg.hip.add(leg.knee);

  const kneePivotGeo = new THREE.CylinderGeometry(0.16, 0.16, 0.26, 16);
  kneePivotGeo.rotateZ(Math.PI / 2);
  const kneePivot = new THREE.Mesh(kneePivotGeo, metalMatDark);
  leg.knee.add(kneePivot);

  // Knee Accent Light
  const kneeCapGeo = new THREE.CylinderGeometry(0.09, 0.09, 0.28, 12);
  kneeCapGeo.rotateZ(Math.PI / 2);
  const kneeCap = new THREE.Mesh(kneeCapGeo, cyanGlowMat);
  leg.knee.add(kneeCap);
  leg.kneeCap = kneeCap;

  // Shin Armature
  const shinGeo = new THREE.BoxGeometry(0.16, 1.05, 0.2);
  const shinMesh = new THREE.Mesh(shinGeo, metalMatArmor);
  shinMesh.position.set(0, -0.52, 0);
  shinMesh.castShadow = true;
  leg.knee.add(shinMesh);

  // Ankle & Footpad
  leg.foot.position.set(0, -1.05, 0);
  leg.knee.add(leg.foot);

  const footGeo = new THREE.BoxGeometry(0.36, 0.12, 0.85);
  const footMesh = new THREE.Mesh(footGeo, metalMatDark);
  footMesh.position.set(0, -0.06, 0.12);
  footMesh.castShadow = true;
  leg.foot.add(footMesh);

  // Claws and Heel Pads
  const toePadGeo = new THREE.BoxGeometry(0.34, 0.06, 0.25);
  const toePad = new THREE.Mesh(toePadGeo, chromeMat);
  toePad.position.set(0, -0.12, 0.4);
  leg.foot.add(toePad);

  const heelPad = new THREE.Mesh(toePadGeo, chromeMat);
  heelPad.position.set(0, -0.12, -0.15);
  leg.foot.add(heelPad);

  // Spark Particle System for Damaged State
  const sparkCount = 30;
  const sparkGeo = new THREE.BufferGeometry();
  const sparkPos = new Float32Array(sparkCount * 3);
  sparkGeo.setAttribute("position", new THREE.BufferAttribute(sparkPos, 3));
  const sparkMat = new THREE.PointsMaterial({
    color: 0xffaa00,
    size: 0.06,
    transparent: true,
    opacity: 0,
  });
  const sparkPoints = new THREE.Points(sparkGeo, sparkMat);
  leg.knee.add(sparkPoints);
  leg.sparkEmitter = { points: sparkPoints, pos: sparkPos, mat: sparkMat };

  return leg;
}

const legL = createAegisLeg(true);
const legR = createAegisLeg(false);

// ──────────────────────────────────────────────
// Holographic Ghost Walker (Healthy Baseline Twin)
// ──────────────────────────────────────────────
const ghostGroup = new THREE.Group();
ghostGroup.position.set(-1.85, 0, 0); // Parallel lane alongside main walker
ghostGroup.visible = false;
scene.add(ghostGroup);

const ghostWireMat = new THREE.MeshBasicMaterial({
  color: 0x64748b,
  wireframe: true,
  transparent: true,
  opacity: 0.22,
});

const ghostInnerMat = new THREE.MeshBasicMaterial({
  color: 0x334155,
  transparent: true,
  opacity: 0.06,
});

const ghostTorso = new THREE.Group();
ghostTorso.position.y = 2.4;
ghostGroup.add(ghostTorso);

// Humanoid Ghost Chest
const ghostChestMesh = new THREE.Mesh(new THREE.BoxGeometry(0.66, 0.44, 0.36), ghostWireMat);
ghostChestMesh.position.set(0, 0.22, 0);
ghostTorso.add(ghostChestMesh);

// Humanoid Ghost Head
const ghostHeadMesh = new THREE.Mesh(new THREE.BoxGeometry(0.24, 0.24, 0.26), ghostWireMat);
ghostHeadMesh.position.set(0, 0.64, 0.02);
ghostTorso.add(ghostHeadMesh);

// Humanoid Ghost Pelvis
const ghostPelvisGeo = new THREE.CylinderGeometry(0.28, 0.24, 0.22, 12);
ghostPelvisGeo.rotateZ(Math.PI / 2);
const ghostPelvisMesh = new THREE.Mesh(ghostPelvisGeo, ghostWireMat);
ghostPelvisMesh.position.y = -0.38;
ghostTorso.add(ghostPelvisMesh);

// Humanoid Ghost Arms
function createGhostArm(isLeft = true) {
  const arm = {
    group: new THREE.Group(),
    shoulder: new THREE.Group(),
    elbow: new THREE.Group(),
  };
  arm.group.position.set(isLeft ? -0.46 : 0.46, 0.32, 0);
  ghostTorso.add(arm.group);

  const bicepMesh = new THREE.Mesh(new THREE.BoxGeometry(0.10, 0.40, 0.12), ghostWireMat);
  bicepMesh.position.set(0, -0.20, 0);
  arm.shoulder.add(bicepMesh);
  arm.group.add(arm.shoulder);

  arm.elbow.position.set(0, -0.40, 0);
  arm.shoulder.add(arm.elbow);

  const forearmMesh = new THREE.Mesh(new THREE.BoxGeometry(0.09, 0.36, 0.10), ghostWireMat);
  forearmMesh.position.set(0, -0.18, 0);
  arm.elbow.add(forearmMesh);

  return arm;
}
const ghostArmL = createGhostArm(true);
const ghostArmR = createGhostArm(false);

function createGhostLeg(isLeft = true) {
  const gleg = {
    group: new THREE.Group(),
    hip: new THREE.Group(),
    knee: new THREE.Group(),
    foot: new THREE.Group(),
  };

  const xOff = isLeft ? -0.45 : 0.45;
  gleg.group.position.set(xOff, -0.45, 0);
  ghostTorso.add(gleg.group);

  // Thigh
  const gThighGeo = new THREE.BoxGeometry(0.16, 0.92, 0.22);
  const gThighMesh = new THREE.Mesh(gThighGeo, ghostWireMat);
  gThighMesh.position.set(0, -0.46, 0);
  gleg.hip.add(gThighMesh);
  gleg.group.add(gleg.hip);

  // Knee
  gleg.knee.position.set(0, -0.92, 0);
  gleg.hip.add(gleg.knee);

  // Shin
  const gShinGeo = new THREE.BoxGeometry(0.14, 1.02, 0.18);
  const gShinMesh = new THREE.Mesh(gShinGeo, ghostWireMat);
  gShinMesh.position.set(0, -0.51, 0);
  gleg.knee.add(gShinMesh);

  // Foot
  gleg.foot.position.set(0, -1.02, 0);
  gleg.knee.add(gleg.foot);
  const gFootGeo = new THREE.BoxGeometry(0.32, 0.10, 0.80);
  const gFootMesh = new THREE.Mesh(gFootGeo, ghostWireMat);
  gFootMesh.position.set(0, -0.05, 0.12);
  gleg.foot.add(gFootMesh);

  return gleg;
}

const ghostLegL = createGhostLeg(true);
const ghostLegR = createGhostLeg(false);

// ──────────────────────────────────────────────
// 3D Neural Synapse Layer (Technical Reference Constellation)
// ──────────────────────────────────────────────
const synapseGroup = new THREE.Group();
synapseGroup.position.set(0, 3.1, 0);
robotGroup.add(synapseGroup);

const nodeCount = 18;
const synapseNodes = [];
const nodeGeo = new THREE.SphereGeometry(0.032, 10, 10);

for (let i = 0; i < nodeCount; i++) {
  const theta = (i / nodeCount) * Math.PI * 2;
  const rad = 0.55 + Math.sin(i * 3) * 0.12;
  const y = (i % 3) * 0.20 - 0.20;

  const nodeMat = new THREE.MeshStandardMaterial({
    color: 0x38bdf8,
    emissive: 0x0284c7,
    emissiveIntensity: 0.35,
    roughness: 0.5,
    metalness: 0.2,
  });
  const node = new THREE.Mesh(nodeGeo, nodeMat);
  node.position.set(Math.cos(theta) * rad, y, Math.sin(theta) * rad);
  synapseGroup.add(node);
  synapseNodes.push({ mesh: node, basePos: node.position.clone(), isAblated: false });
}

// Synapse Connecting Lines
const lineMat = new THREE.LineBasicMaterial({
  color: 0x38bdf8,
  transparent: true,
  opacity: 0.20,
});
const lineGeo = new THREE.BufferGeometry();
const linePositions = new Float32Array(nodeCount * 6);
lineGeo.setAttribute("position", new THREE.BufferAttribute(linePositions, 3));
const synapseLines = new THREE.LineSegments(lineGeo, lineMat);
synapseGroup.add(synapseLines);

function updateSynapseLines() {
  const pos = synapseLines.geometry.attributes.position.array;
  let idx = 0;
  for (let i = 0; i < nodeCount; i++) {
    const next = (i + 1) % nodeCount;
    const n1 = synapseNodes[i].mesh.position;
    const n2 = synapseNodes[next].mesh.position;
    pos[idx++] = n1.x; pos[idx++] = n1.y; pos[idx++] = n1.z;
    pos[idx++] = n2.x; pos[idx++] = n2.y; pos[idx++] = n2.z;
  }
  synapseLines.geometry.attributes.position.needsUpdate = true;
}

// ──────────────────────────────────────────────
// Robot Chassis Model Switching
// ──────────────────────────────────────────────
function setRobotChassis(type) {
  state.currentRobot = type;

  xbotGroup.visible = (type === "xbot");
  sparkeGroup.visible = (type === "sparke");
  aegisGroup.visible = (type === "aegis");

  document.querySelectorAll(".btn-robot").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.robot === type);
  });

  const badge = document.getElementById("robot-badge");
  if (badge) {
    if (type === "xbot") badge.textContent = "Humanoid";
    else if (type === "sparke") badge.textContent = "Bionic";
    else badge.textContent = "Cyber-Humanoid";
  }

  // Adjust synapse height according to robot stature
  if (type === "sparke") {
    synapseGroup.position.set(0, 2.4, 0);
  } else if (type === "xbot") {
    synapseGroup.position.set(0, 3.1, 0);
  } else {
    synapseGroup.position.set(0, 3.4, 0);
  }
}

// ──────────────────────────────────────────────
// Evolutionary Policy Mode Switching
// ──────────────────────────────────────────────
function setPolicyMode(mode) {
  state.policyMode = mode;

  document.querySelectorAll(".btn-policy").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.policy === mode);
  });

  const badge = document.getElementById("badge-policy-state");
  if (badge) {
    badge.textContent = mode === "evolved" ? "Evolved Agent" : "Standard PPO";
  }

  updateNeuralSynapseAblation();
}

// ──────────────────────────────────────────────
// Kinematic Gait & Animation Loop
// ──────────────────────────────────────────────
let walkPhase = 0;
let lastStepFloor = 0;

function updateWalkingKinematics(delta) {
  if (!state.isRunning) return;

  const isEvolved = (state.policyMode === "evolved");
  const effectiveSpeed = state.simSpeed * (state.status === "damaged" ? (isEvolved ? 0.90 : 0.65) : 1.0);
  walkPhase += delta * 4.2 * effectiveSpeed;
  state.time += delta * effectiveSpeed;

  const sev = state.severity;
  const isHealthy = state.status === "healthy" || (state.status === "recovered" && sev <= 0.3);
  const isSevere = sev >= 0.5 && state.status === "damaged";
  const isCritical = sev >= 0.7 && state.status === "damaged";

  // 1. Update X-BOT Model
  if (xbotMixer && xbotGroup.visible) {
    xbotMixer.update(delta * effectiveSpeed);

    if (state.status === "damaged" && !isEvolved) {
      if (isCritical) {
        // Critical collapse: blend sad_pose
        if (xbotActions["sad_pose"] && xbotCurrentAction !== xbotActions["sad_pose"]) {
          xbotCurrentAction?.fadeOut(0.3);
          xbotActions["sad_pose"].reset().fadeIn(0.3).play();
          xbotCurrentAction = xbotActions["sad_pose"];
        }
      } else if (isSevere) {
        // Severe limp: slow walk with twitch
        if (xbotActions["walk"] && xbotCurrentAction !== xbotActions["walk"]) {
          xbotCurrentAction?.fadeOut(0.2);
          xbotActions["walk"].reset().fadeIn(0.2).play();
          xbotCurrentAction = xbotActions["walk"];
        }
        if (xbotActions["walk"]) xbotActions["walk"].timeScale = 0.55;
        if (xbotLeftLegBone) {
          xbotLeftLegBone.rotation.z = Math.sin(state.time * 12) * 0.18;
        }
      } else {
        // Mild damage
        if (xbotActions["walk"] && xbotCurrentAction !== xbotActions["walk"]) {
          xbotCurrentAction?.fadeOut(0.2);
          xbotActions["walk"].reset().fadeIn(0.2).play();
          xbotCurrentAction = xbotActions["walk"];
        }
        if (xbotActions["walk"]) xbotActions["walk"].timeScale = 0.85;
      }
    } else {
      // Healthy, Recovered, or Evolved Agent Resilience
      if (xbotActions["walk"] && xbotCurrentAction !== xbotActions["walk"]) {
        xbotCurrentAction?.fadeOut(0.3);
        xbotActions["walk"].reset().fadeIn(0.3).play();
        xbotCurrentAction = xbotActions["walk"];
      }
      if (xbotActions["walk"]) {
        xbotActions["walk"].timeScale = isEvolved && state.status === "damaged" ? 0.90 : 1.0;
      }
    }
  }

  // 2. Update SPARK-E Model
  if (sparkeMixer && sparkeGroup.visible) {
    sparkeMixer.update(delta * effectiveSpeed);

    if (state.status === "damaged" && !isEvolved) {
      if (isCritical) {
        if (sparkeActions["death"] && sparkeCurrentAction !== sparkeActions["death"]) {
          sparkeCurrentAction?.fadeOut(0.2);
          sparkeActions["death"].reset().fadeIn(0.2).play();
          sparkeCurrentAction = sparkeActions["death"];
        }
      } else if (isSevere) {
        if (sparkeActions["sitting"] && sparkeCurrentAction !== sparkeActions["sitting"]) {
          sparkeCurrentAction?.fadeOut(0.3);
          sparkeActions["sitting"].reset().fadeIn(0.3).play();
          sparkeCurrentAction = sparkeActions["sitting"];
        }
      } else {
        if (sparkeActions["walking"] && sparkeCurrentAction !== sparkeActions["walking"]) {
          sparkeCurrentAction?.fadeOut(0.2);
          sparkeActions["walking"].reset().fadeIn(0.2).play();
          sparkeCurrentAction = sparkeActions["walking"];
        }
        if (sparkeActions["walking"]) sparkeActions["walking"].timeScale = 0.65;
      }
    } else {
      // Recovered / Healthy / Evolved
      if (sparkeActions["walking"] && sparkeCurrentAction !== sparkeActions["walking"]) {
        sparkeCurrentAction?.fadeOut(0.3);
        sparkeActions["walking"].reset().fadeIn(0.3).play();
        sparkeCurrentAction = sparkeActions["walking"];
      }
      if (sparkeActions["walking"]) {
        sparkeActions["walking"].timeScale = isEvolved && state.status === "damaged" ? 0.90 : 1.0;
      }
    }
  }

  // 3. Update AEGIS Mech with Neural Plasticity Engine
  const res = neuralEngine.step(delta);

  // Update 8-DOF physical telemetry (4 neural policy outputs + lateral stabilization & compliant ankle torques)
  const rollTorque = Math.max(-1.0, Math.min(1.0, -neuralEngine.physics.roll * 1.1 - neuralEngine.physics.droll * 0.2));
  state.torques[0] = rollTorque;                                                        // Hip Roll L
  state.torques[1] = res.actions[0];                                                    // Hip Pitch L
  state.torques[2] = res.actions[1];                                                    // Knee L
  state.torques[3] = state.contacts[0] ? Math.max(-1.0, Math.min(1.0, -res.actions[1] * 0.35 + 0.1)) : 0.0; // Ankle L

  state.torques[4] = -rollTorque;                                                       // Hip Roll R
  state.torques[5] = res.actions[2];                                                    // Hip Pitch R
  state.torques[6] = res.actions[3];                                                    // Knee R
  state.torques[7] = state.contacts[1] ? Math.max(-1.0, Math.min(1.0, -res.actions[3] * 0.35 + 0.1)) : 0.0; // Ankle R

  state.vx = parseFloat(res.vx.toFixed(2));
  state.reward = parseFloat((res.reward * 75.0).toFixed(1));
  state.symmetry = parseFloat(res.symmetry.toFixed(1));
  state.tilt = parseFloat((neuralEngine.physics.pitch * 57.3).toFixed(1));
  state.contacts[0] = neuralEngine.physics.contacts[0];
  state.contacts[1] = neuralEngine.physics.contacts[1];

  // Autonomous Anomaly & Self-Healing State Transitions
  if (res.isAdapting && state.status === "damaged") {
    state.status = "rehabilitating";
    state.isRehabilitating = true;
    updateStatusBadge();
  } else if (!res.isAdapting && state.status === "rehabilitating" && res.symmetry > 86.0) {
    state.status = "recovered";
    state.isRehabilitating = false;
    updateStatusBadge();
    updateNeuralSynapseAblation();
    playSuccessChime();
  }

  // Driven kinematics from neural engine's dynamic multi-joint physics
  const q = neuralEngine.physics.q;

  legL.hip.rotation.x = -q[0] * 0.72;
  legL.knee.rotation.x = Math.max(0.08, q[1]);
  legL.foot.rotation.x = -legL.knee.rotation.x * 0.45;

  legR.hip.rotation.x = -q[2] * 0.72;
  legR.knee.rotation.x = Math.max(0.08, q[3]);
  legR.foot.rotation.x = -legR.knee.rotation.x * 0.45;

  torsoGroup.position.y = neuralEngine.physics.torsoY;
  torsoGroup.rotation.x = neuralEngine.physics.pitch;
  torsoGroup.rotation.z = neuralEngine.physics.roll;

  legL.thighPiston.scale.y = 1.0 + legL.knee.rotation.x * 0.25;
  legR.thighPiston.scale.y = 1.0 + legR.knee.rotation.x * 0.25;

  // Dynamic Humanoid Reciprocal Arm Swing (Natural Locomotion Counter-balance)
  const armSwingL = Math.sin(walkPhase + Math.PI) * 0.45;
  const armSwingR = Math.sin(walkPhase) * 0.45;

  armL.shoulder.rotation.x = armSwingL;
  armL.shoulder.rotation.z = -0.16 - Math.abs(neuralEngine.physics.roll) * 0.45;
  armL.elbow.rotation.x = Math.max(0.12, -armSwingL * 0.45 + 0.28);

  armR.shoulder.rotation.x = armSwingR;
  armR.shoulder.rotation.z = 0.16 + Math.abs(neuralEngine.physics.roll) * 0.45;
  armR.elbow.rotation.x = Math.max(0.12, -armSwingR * 0.45 + 0.28);

  // Gaze Stabilization & Head Compensation
  if (typeof headGroup !== "undefined" && headGroup) {
    headGroup.rotation.x = -neuralEngine.physics.pitch * 0.55;
    headGroup.rotation.y = -neuralEngine.physics.roll * 0.35;
  }

  triggerSparks(legL.sparkEmitter, state.status === "damaged" && sev >= 0.3 && Math.sin(walkPhase * 2) > 0.4);

  // Servo audio on foot contact shifts & ground impact thump
  const currentStepFloor = Math.floor(walkPhase / Math.PI);
  if (currentStepFloor !== lastStepFloor) {
    lastStepFloor = currentStepFloor;
    playServoSound(110 + Math.abs(state.torques[0]) * 55, 0.08);
    playFootImpactSound();
  }

  // Update Holographic Ghost Walker (Pristine Healthy Baseline Kinematics)
  if (ghostGroup && state.showGhost) {
    ghostGroup.visible = true;
    const gSinL = Math.sin(walkPhase);
    const gCosL = Math.cos(walkPhase);
    const gSinR = Math.sin(walkPhase + Math.PI);
    const gCosR = Math.cos(walkPhase + Math.PI);

    ghostTorso.position.y = 2.4 + Math.abs(Math.sin(walkPhase * 2)) * 0.12;
    ghostTorso.rotation.x = -0.06 + Math.sin(walkPhase * 2) * 0.03;
    ghostTorso.rotation.z = Math.sin(walkPhase) * 0.04;

    ghostLegL.hip.rotation.x = gSinL * 0.55;
    ghostLegL.knee.rotation.x = Math.max(0.1, -gCosL * 0.92);
    ghostLegL.foot.rotation.x = -ghostLegL.knee.rotation.x * 0.45;

    ghostLegR.hip.rotation.x = gSinR * 0.55;
    ghostLegR.knee.rotation.x = Math.max(0.1, -gCosR * 0.92);
    ghostLegR.foot.rotation.x = -ghostLegR.knee.rotation.x * 0.45;

    // Humanoid Ghost Reciprocal Arm Swing
    ghostArmL.shoulder.rotation.x = -gSinL * 0.45;
    ghostArmL.elbow.rotation.x = Math.max(0.1, gSinL * 0.3 + 0.2);
    ghostArmR.shoulder.rotation.x = -gSinR * 0.45;
    ghostArmR.elbow.rotation.x = Math.max(0.1, gSinR * 0.3 + 0.2);
  } else if (ghostGroup) {
    ghostGroup.visible = false;
  }

  // Update Synapse Constellation Rotation
  synapseGroup.rotation.y += delta * 0.6;
  synapseGroup.visible = state.showSynapse;
  updateSynapseLines();

  // Update Dynamic GRF Force Plates
  if (typeof forcePlateGroup !== "undefined" && forcePlateGroup) {
    forcePlateGroup.visible = state.showForcePlates;
    const targetL = state.contacts[0] ? 0.85 : 0.05;
    const targetR = state.contacts[1] ? 0.85 : 0.05;
    fpMatL.emissiveIntensity += (targetL - fpMatL.emissiveIntensity) * 0.25;
    fpMatR.emissiveIntensity += (targetR - fpMatR.emissiveIntensity) * 0.25;
  }

  // Check foot strike transitions for ground shockwaves
  if (typeof window.lastContactL === "undefined") {
    window.lastContactL = false;
    window.lastContactR = false;
  }
  if (state.contacts[0] && !window.lastContactL) {
    triggerGroundShockwave(-0.46, 0);
  }
  if (state.contacts[1] && !window.lastContactR) {
    triggerGroundShockwave(0.46, 0);
  }
  window.lastContactL = state.contacts[0];
  window.lastContactR = state.contacts[1];

  // Update Expanding Ground Shockwave Ripples
  if (typeof ripplePool !== "undefined") {
    ripplePool.forEach((r) => {
      if (r.life > 0) {
        r.life -= delta;
        const progress = 1.0 - r.life / r.maxLife;
        const s = 0.5 + progress * 2.2;
        r.mesh.scale.set(s, s, s);
        r.mat.opacity = (1.0 - progress) * 0.75;
        if (r.life <= 0) r.mesh.visible = false;
      }
    });
  }

  // Update Overhead Gantry Trolley tracking
  if (typeof gantryGroup !== "undefined" && gantryGroup) {
    gantryGroup.visible = state.showGantry;
    if (typeof trolleyMesh !== "undefined" && trolleyMesh) {
      const targetZ = robotGroup.position.z;
      trolleyMesh.position.z += (targetZ - trolleyMesh.position.z) * 0.1;
      damperMesh.position.z = trolleyMesh.position.z;
      cableMesh.position.z = trolleyMesh.position.z;
    }
  }

  // Update MoCap Optical Towers visibility
  if (typeof mocapGroup !== "undefined" && mocapGroup) {
    mocapGroup.visible = state.showMocap;
  }

  // Update Atmospheric Ambient Motes
  if (typeof motes !== "undefined" && motes) {
    const pos = motes.geometry.attributes.position.array;
    for (let i = 0; i < moteCount; i++) {
      pos[i * 3 + 1] += Math.sin(state.time * 2 + i) * 0.003;
      if (pos[i * 3 + 1] < 0.4) pos[i * 3 + 1] = 6.8;
      if (pos[i * 3 + 1] > 7.0) pos[i * 3 + 1] = 0.5;
    }
    motes.geometry.attributes.position.needsUpdate = true;
  }
}

function triggerSparks(emitter, active) {
  if (!emitter) return;
  emitter.mat.opacity = active ? 0.9 : 0;
  if (active) {
    const pos = emitter.pos;
    for (let i = 0; i < pos.length; i += 3) {
      pos[i] = (Math.random() - 0.5) * 0.35;
      pos[i + 1] = (Math.random() - 0.5) * 0.35;
      pos[i + 2] = (Math.random() - 0.5) * 0.35;
    }
    emitter.points.geometry.attributes.position.needsUpdate = true;
  }
}

// ──────────────────────────────────────────────
// Neural Synapse Ablation Update
// ──────────────────────────────────────────────
function updateNeuralSynapseAblation() {
  const sev = state.severity;
  const numToKill = Math.round(sev * nodeCount);
  const isEvolved = (state.policyMode === "evolved");

  synapseNodes.forEach((node, i) => {
    const shouldKill = i < numToKill && state.status === "damaged";
    node.isAblated = shouldKill;

    if (shouldKill) {
      if (isEvolved) {
        // Evolved Agent: Compensatory bypass (muted amber)
        node.mesh.material.color.setHex(0xd97706);
        node.mesh.material.emissive.setHex(0xb45309);
        node.mesh.material.emissiveIntensity = 0.4;
        node.mesh.scale.set(0.9, 0.9, 0.9);
      } else {
        node.mesh.material.color.setHex(0xdc2626);
        node.mesh.material.emissive.setHex(0x991b1b);
        node.mesh.material.emissiveIntensity = 0.35;
        node.mesh.scale.set(0.7, 0.7, 0.7);
      }
    } else if (isEvolved) {
      const col = i % 2 === 0 ? 0xd97706 : 0x0284c7;
      node.mesh.material.color.setHex(col);
      node.mesh.material.emissive.setHex(col);
      node.mesh.material.emissiveIntensity = 0.35;
      node.mesh.scale.set(1.0, 1.0, 1.0);
    } else if (state.status === "recovered") {
      const col = i % 2 === 0 ? 0x0d9488 : 0x0284c7;
      node.mesh.material.color.setHex(col);
      node.mesh.material.emissive.setHex(col);
      node.mesh.material.emissiveIntensity = 0.3;
      node.mesh.scale.set(1.0, 1.0, 1.0);
    } else {
      node.mesh.material.color.setHex(0x38bdf8);
      node.mesh.material.emissive.setHex(0x0284c7);
      node.mesh.material.emissiveIntensity = 0.35;
      node.mesh.scale.set(1.0, 1.0, 1.0);
    }
  });

  if (state.status === "damaged") {
    eyeMesh.material = isEvolved ? goldGlowMat : redGlowMat;
    legL.kneeCap.material = isEvolved ? goldGlowMat : redGlowMat;
  } else {
    eyeMesh.material = isEvolved ? goldGlowMat : cyanGlowMat;
    legL.kneeCap.material = isEvolved ? goldGlowMat : cyanGlowMat;
  }
}

// ──────────────────────────────────────────────
// Rehabilitation Routine Animation
// ──────────────────────────────────────────────
function startRehabilitation() {
  if (state.severity === 0 && state.status !== "damaged") {
    state.status = "healthy";
    updateStatusBadge();
    return;
  }

  state.isRehabilitating = true;
  state.status = "rehabilitating";
  updateStatusBadge();

  // Engage active online learning adaptation in the neural engine
  neuralEngine.isAdapting = true;
  neuralEngine.adaptationStep = 0;
  neuralEngine.isAutoLearningActive = true;
}

// ──────────────────────────────────────────────
// UI Controls & Event Listeners
// ──────────────────────────────────────────────
const sliderSeverity = document.getElementById("slider-severity");
const valSeverityDisplay = document.getElementById("val-severity-display");
const statusBadge = document.getElementById("global-status-badge");
const statusText = document.getElementById("status-text");

function updateStatusBadge() {
  statusBadge.className = `status-indicator ${state.status}`;
  if (state.status === "healthy") {
    statusText.textContent = "Healthy · 100% integrity";
  } else if (state.status === "damaged") {
    statusText.textContent = `Lesion active · ${(state.severity * 100).toFixed(0)}% ablated`;
  } else if (state.status === "rehabilitating") {
    statusText.textContent = `Autonomous adaptive plasticity active`;
  } else if (state.status === "recovered") {
    statusText.textContent = `Rehabilitated · Compensatory gait`;
  }
}

function setSeverity(sev) {
  state.severity = sev;
  sliderSeverity.value = Math.round(sev * 100);
  valSeverityDisplay.textContent = `${Math.round(sev * 100)}%`;

  document.querySelectorAll(".btn-preset").forEach((btn) => {
    btn.classList.toggle("active", parseFloat(btn.dataset.sev) === sev);
  });

  neuralEngine.setLesion(state.mode, sev);

  if (sev > 0 && state.status !== "rehabilitating") {
    state.status = "damaged";
    playDamageSpark();
  } else if (sev === 0) {
    state.status = "healthy";
  }
  updateStatusBadge();
  updateNeuralSynapseAblation();
}

sliderSeverity.addEventListener("input", (e) => {
  initAudio();
  setSeverity(parseInt(e.target.value) / 100.0);
});

document.querySelectorAll(".btn-preset").forEach((btn) => {
  btn.addEventListener("click", () => {
    initAudio();
    setSeverity(parseFloat(btn.dataset.sev));
  });
});

document.querySelectorAll(".btn-mode").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".btn-mode").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    state.mode = btn.dataset.mode;
    neuralEngine.setLesion(state.mode, state.severity);
  });
});

// Autonomous Learning Checkboxes
const chkAutoHeal = document.getElementById("chk-auto-heal");
if (chkAutoHeal) {
  chkAutoHeal.addEventListener("change", (e) => {
    neuralEngine.autoHealEnabled = e.target.checked;
  });
}
const chkContinuousImprove = document.getElementById("chk-continuous-improve");
if (chkContinuousImprove) {
  chkContinuousImprove.addEventListener("change", (e) => {
    neuralEngine.continuousImproveEnabled = e.target.checked;
  });
}

// Robot Model Selection Buttons
document.querySelectorAll(".btn-robot").forEach((btn) => {
  btn.addEventListener("click", () => {
    initAudio();
    setRobotChassis(btn.dataset.robot);
  });
});

// Policy Evolution Mode Selector
function setPolicyMode(policy) {
  state.policyMode = policy;
  document.querySelectorAll(".btn-policy").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.policy === policy);
  });
  const isEvolved = (policy === "evolved");
  const badge = document.getElementById("badge-policy-state");
  if (badge) {
    badge.style.display = "inline-block";
    badge.textContent = isEvolved ? "EVOLVED AGENT (RESILIENT)" : "STANDARD PPO";
    badge.className = `status-pill ${isEvolved ? "evolved" : "standard"}`;
  }
  updateNeuralSynapseAblation();
}

// Policy Evolution Selection Buttons
document.querySelectorAll(".btn-policy").forEach((btn) => {
  btn.addEventListener("click", () => {
    initAudio();
    setPolicyMode(btn.dataset.policy);
  });
});

document.getElementById("btn-apply-damage").addEventListener("click", () => {
  initAudio();
  if (state.severity === 0) setSeverity(0.5);
  state.status = "damaged";
  playDamageSpark();
  updateStatusBadge();
  updateNeuralSynapseAblation();
});

document.getElementById("btn-rehabilitate").addEventListener("click", () => {
  initAudio();
  startRehabilitation();
});

document.getElementById("btn-play-pause").addEventListener("click", () => {
  initAudio();
  state.isRunning = !state.isRunning;
  document.getElementById("icon-play").innerHTML = state.isRunning
    ? '<path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/>'
    : '<path d="M8 5v14l11-7z"/>';
});

document.querySelectorAll(".btn-speed").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".btn-speed").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    state.simSpeed = parseFloat(btn.dataset.speed);
  });
});

document.getElementById("btn-reset-cam").addEventListener("click", () => {
  camera.position.set(0, 3.2, 8.2);
  controls.target.set(0, 1.7, 0);
  controls.update();
});

// Toggles
const chkGhost = document.getElementById("chk-ghost-walker");
if (chkGhost) {
  chkGhost.addEventListener("change", (e) => state.showGhost = e.target.checked);
}
document.getElementById("chk-neural-synapse").addEventListener("change", (e) => state.showSynapse = e.target.checked);
document.getElementById("chk-joint-forces").addEventListener("change", (e) => state.showVectors = e.target.checked);
document.getElementById("chk-auto-cam").addEventListener("change", (e) => state.autoCam = e.target.checked);
document.getElementById("chk-audio").addEventListener("change", (e) => {
  state.audioEnabled = e.target.checked;
  if (state.audioEnabled) initAudio();
});

// Camera Mode Button Group
document.querySelectorAll(".btn-cam-mode").forEach((btn) => {
  btn.addEventListener("click", () => {
    initAudio();
    document.querySelectorAll(".btn-cam-mode").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    state.cameraMode = btn.dataset.cam;
  });
});

// Telemetry Export Buttons
const btnExportCsv = document.getElementById("btn-export-csv");
if (btnExportCsv) {
  btnExportCsv.addEventListener("click", () => exportTelemetry("csv"));
}
const btnExportJson = document.getElementById("btn-export-json");
if (btnExportJson) {
  btnExportJson.addEventListener("click", () => exportTelemetry("json"));
}

// ──────────────────────────────────────────────
// Environment Preset Configuration Engine
// ──────────────────────────────────────────────
function setEnvironment(envName) {
  state.currentEnv = envName;
  document.querySelectorAll(".btn-env").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.env === envName);
  });

  if (envName === "hangar") {
    scene.background = new THREE.Color(0x0a0f1d);
    scene.fog = new THREE.FogExp2(0x0a0f1d, 0.026);
    if (envMaps["hangar"]) scene.environment = envMaps["hangar"];
    floorMat.map = hangarDiff;
    floorMat.normalMap = hangarNor;
    floorMat.roughnessMap = hangarRough;
    floorMat.color.setHex(0xffffff);
    floorMat.roughness = 0.65;
    floorMat.metalness = 0.15;
    floorMat.needsUpdate = true;
    runwayMesh.visible = true;
    markingsMesh.visible = true;
    mainSpot.color.setHex(0xffffff);
    mainSpot.intensity = 2.5;
    rimLight.color.setHex(0x38bdf8);
    warmFill.color.setHex(0xf59e0b);
  } else if (envName === "cyber") {
    scene.background = new THREE.Color(0x050811);
    scene.fog = new THREE.FogExp2(0x050811, 0.035);
    if (envMaps["studio"]) scene.environment = envMaps["studio"];
    floorMat.map = null;
    floorMat.normalMap = null;
    floorMat.roughnessMap = null;
    floorMat.color.setHex(0x0a0f18);
    floorMat.roughness = 0.22;
    floorMat.metalness = 0.85;
    floorMat.needsUpdate = true;
    runwayMesh.visible = true;
    markingsMesh.visible = true;
    mainSpot.color.setHex(0x38bdf8);
    mainSpot.intensity = 2.2;
    rimLight.color.setHex(0x0284c7);
    warmFill.color.setHex(0x6366f1);
  } else if (envName === "studio") {
    scene.background = new THREE.Color(0x1e293b);
    scene.fog = new THREE.FogExp2(0x1e293b, 0.018);
    if (envMaps["studio"]) scene.environment = envMaps["studio"];
    floorMat.map = null;
    floorMat.normalMap = null;
    floorMat.roughnessMap = null;
    floorMat.color.setHex(0x334155);
    floorMat.roughness = 0.75;
    floorMat.metalness = 0.08;
    floorMat.needsUpdate = true;
    runwayMesh.visible = false;
    markingsMesh.visible = false;
    mainSpot.color.setHex(0xffffff);
    mainSpot.intensity = 3.0;
    rimLight.color.setHex(0x94a3b8);
    warmFill.color.setHex(0xcfd8dc);
  }
  console.info(`[AEGIS Environment] Switched to preset: ${envName}`);
}

// Environment Preset Selector Buttons
document.querySelectorAll(".btn-env").forEach((btn) => {
  btn.addEventListener("click", () => {
    initAudio();
    setEnvironment(btn.dataset.env);
  });
});

// Facility Layer Toggles
const chkGantry = document.getElementById("chk-gantry-harness");
if (chkGantry) {
  chkGantry.addEventListener("change", (e) => {
    state.showGantry = e.target.checked;
    if (typeof gantryGroup !== "undefined" && gantryGroup) gantryGroup.visible = state.showGantry;
  });
}
const chkMocap = document.getElementById("chk-mocap-towers");
if (chkMocap) {
  chkMocap.addEventListener("change", (e) => {
    state.showMocap = e.target.checked;
    if (typeof mocapGroup !== "undefined" && mocapGroup) mocapGroup.visible = state.showMocap;
  });
}
const chkForcePlates = document.getElementById("chk-force-plates");
if (chkForcePlates) {
  chkForcePlates.addEventListener("change", (e) => {
    state.showForcePlates = e.target.checked;
    if (typeof forcePlateGroup !== "undefined" && forcePlateGroup) forcePlateGroup.visible = state.showForcePlates;
  });
}

// ──────────────────────────────────────────────
// Cinematic Camera Rig Dynamics
// ──────────────────────────────────────────────
const targetCamPos = new THREE.Vector3(0, 3.2, 8.2);
const targetLookAt = new THREE.Vector3(0, 1.7, 0);

function updateCameraRig() {
  if (state.cameraMode === "orbit") {
    if (state.autoCam) {
      controls.target.set(robotGroup.position.x, 1.8, robotGroup.position.z);
    }
    return;
  }

  const rx = robotGroup.position.x;
  const ry = robotGroup.position.y;
  const rz = robotGroup.position.z;

  if (state.cameraMode === "chase") {
    targetCamPos.set(rx, ry + 2.4, rz + 6.4);
    targetLookAt.set(rx, ry + 1.7, rz);
  } else if (state.cameraMode === "sagittal") {
    targetCamPos.set(rx + 6.8, ry + 1.8, rz);
    targetLookAt.set(rx, ry + 1.7, rz);
  } else if (state.cameraMode === "top") {
    targetCamPos.set(rx, ry + 9.5, rz + 0.2);
    targetLookAt.set(rx, ry, rz);
  }

  camera.position.lerp(targetCamPos, 0.05);
  controls.target.lerp(targetLookAt, 0.06);
}

// ──────────────────────────────────────────────
// Telemetry & Oscilloscope Rendering
// ──────────────────────────────────────────────
const oscCanvas = document.getElementById("canvas-oscilloscope");
const oscCtx = oscCanvas.getContext("2d");
const oscHistory = [[], [], [], []];
const maxOscSamples = 70;

const learnCanvas = document.getElementById("canvas-learning-curve");
const learnCtx = learnCanvas ? learnCanvas.getContext("2d") : null;

const matrixCanvas = document.getElementById("canvas-neural-matrix");
const mCtx = matrixCanvas ? matrixCanvas.getContext("2d") : null;
const matrixBadge = document.getElementById("matrix-status-badge");

const telemetryBuffer = [];
let lastRecordStep = -1;

function recordTelemetrySample() {
  if (state.stepCount === lastRecordStep) return;
  lastRecordStep = state.stepCount;

  telemetryBuffer.push({
    step: state.stepCount,
    time_s: parseFloat(state.time.toFixed(2)),
    velocity_mps: parseFloat(state.vx.toFixed(2)),
    reward: parseFloat(state.reward.toFixed(1)),
    pitch_deg: parseFloat(state.tilt.toFixed(1)),
    symmetry_pct: parseFloat(state.symmetry.toFixed(1)),
    hip_roll_l_torque: parseFloat(state.torques[0].toFixed(2)),
    hip_pitch_l_torque: parseFloat(state.torques[1].toFixed(2)),
    knee_l_torque: parseFloat(state.torques[2].toFixed(2)),
    ankle_l_torque: parseFloat(state.torques[3].toFixed(2)),
    hip_roll_r_torque: parseFloat(state.torques[4].toFixed(2)),
    hip_pitch_r_torque: parseFloat(state.torques[5].toFixed(2)),
    knee_r_torque: parseFloat(state.torques[6].toFixed(2)),
    ankle_r_torque: parseFloat(state.torques[7].toFixed(2)),
    leg_l_contact: state.contacts[0] ? 1 : 0,
    leg_r_contact: state.contacts[1] ? 1 : 0,
    robot: state.currentRobot,
    policy: state.policyMode,
    severity: state.severity,
    status: state.status,
  });

  if (telemetryBuffer.length > 800) telemetryBuffer.shift();
}

function exportTelemetry(format = "csv") {
  if (telemetryBuffer.length === 0) {
    recordTelemetrySample();
  }
  const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
  let blob = null;
  let filename = "";

  if (format === "csv") {
    const headers = Object.keys(telemetryBuffer[0]).join(",");
    const rows = telemetryBuffer.map((obj) => Object.values(obj).join(",")).join("\n");
    const csv = `${headers}\n${rows}`;
    blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    filename = `aegis_telemetry_${timestamp}.csv`;
  } else {
    const jsonStr = JSON.stringify({
      project: "Walker Damage and Recovery",
      lead: "Geo Mathew Joseph",
      timestamp: new Date().toISOString(),
      state: state,
      recordsCount: telemetryBuffer.length,
      records: telemetryBuffer,
    }, null, 2);
    blob = new Blob([jsonStr], { type: "application/json;charset=utf-8;" });
    filename = `aegis_telemetry_${timestamp}.json`;
  }

  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

function updateTelemetryUI() {
  document.getElementById("val-vx").textContent = state.vx.toFixed(2);
  document.getElementById("bar-vx").style.width = `${Math.min(100, state.vx * 50)}%`;

  const rewElem = document.getElementById("val-reward");
  rewElem.textContent = (state.reward >= 0 ? "+" : "") + state.reward.toFixed(1);
  rewElem.className = `metric-val mono ${state.reward >= 0 ? "positive" : "negative"}`;

  document.getElementById("val-tilt").textContent = state.tilt.toFixed(1);
  document.getElementById("val-symmetry").textContent = state.symmetry.toFixed(1);

  const actIds = [
    "act-hip-roll-l", "act-hip-pitch-l", "act-knee-l", "act-ankle-l",
    "act-hip-roll-r", "act-hip-pitch-r", "act-knee-r", "act-ankle-r"
  ];
  const valIds = [
    "val-hip-roll-l", "val-hip-pitch-l", "val-knee-l", "val-ankle-l",
    "val-hip-roll-r", "val-hip-pitch-r", "val-knee-r", "val-ankle-r"
  ];

  state.torques.forEach((val, i) => {
    const fill = document.getElementById(actIds[i]);
    const valText = document.getElementById(valIds[i]);
    if (!fill || !valText) return;
    const pct = Math.abs(val) * 50;

    if (val >= 0) {
      fill.style.left = "50%";
      fill.style.width = `${pct}%`;
      fill.style.background = "var(--accent-primary)";
    } else {
      fill.style.left = `${50 - pct}%`;
      fill.style.width = `${pct}%`;
      fill.style.background = "var(--state-damaged-text)";
    }
    valText.textContent = (val >= 0 ? "+" : "") + val.toFixed(2);
  });

  // Track primary sagittal joint torques on oscilloscope
  const oscIndices = [1, 2, 5, 6];
  oscIndices.forEach((torqueIdx, oIdx) => {
    oscHistory[oIdx].push(state.torques[torqueIdx]);
    if (oscHistory[oIdx].length > maxOscSamples) oscHistory[oIdx].shift();
  });

  const contactLElem = document.getElementById("contact-l");
  const contactRElem = document.getElementById("contact-r");
  if (contactLElem) {
    contactLElem.className = `contact-indicator ${state.contacts[0] ? "on" : ""}`;
  }
  if (contactRElem) {
    contactRElem.className = `contact-indicator ${state.contacts[1] ? "on" : ""}`;
  }

  state.stepCount = (state.stepCount + 1) % 1600;
  document.getElementById("val-step-count").textContent = String(state.stepCount).padStart(4, "0");

  // Dynamic Timeline update based on active neural plasticity
  if (state.isRehabilitating || neuralEngine.isAdapting) {
    const p = Math.min(1.0, Math.max(0.05, (state.symmetry - 35) / 55));
    document.getElementById("rehab-bar-fill").style.width = `${(p * 100).toFixed(0)}%`;
    document.getElementById("rehab-pct-text").textContent = `Adaptive Plasticity ${(p * 100).toFixed(0)}%`;
    document.getElementById("rehab-phase-text").textContent =
      p < 0.4 ? "Optimizing temporal-difference policy gradients..." :
      p < 0.8 ? "Re-weighting synaptic connections across intact pathways..." :
      "Compensatory bipedal equilibrium restabilized";
  } else if (state.status === "recovered") {
    document.getElementById("rehab-bar-fill").style.width = "100%";
    document.getElementById("rehab-pct-text").textContent = "Rehabilitation 100%";
    document.getElementById("rehab-phase-text").textContent = "Rehabilitation complete · Compensatory gait active";
  } else if (state.status === "healthy") {
    document.getElementById("rehab-bar-fill").style.width = "100%";
    document.getElementById("rehab-pct-text").textContent = "Baseline 100%";
    document.getElementById("rehab-phase-text").textContent = "Locomotion running normally";
  }

  const pill = document.getElementById("learning-status-pill");
  if (pill) {
    pill.textContent = `${neuralEngine.totalUpdates} updates · 50 Hz`;
  }

  drawOscilloscope();
  drawLearningCurve();
  drawNeuralActivationMatrix();
  recordTelemetrySample();
}

const oscColors = ["#0284c7", "#94a3b8", "#f59e0b", "#64748b"];
function drawOscilloscope() {
  const w = oscCanvas.width;
  const h = oscCanvas.height;
  oscCtx.fillStyle = "#090d14";
  oscCtx.fillRect(0, 0, w, h);

  oscCtx.strokeStyle = "rgba(255, 255, 255, 0.05)";
  oscCtx.lineWidth = 1;
  oscCtx.beginPath();
  oscCtx.moveTo(0, h / 2);
  oscCtx.lineTo(w, h / 2);
  oscCtx.moveTo(0, h * 0.25);
  oscCtx.lineTo(w, h * 0.25);
  oscCtx.moveTo(0, h * 0.75);
  oscCtx.lineTo(w, h * 0.75);
  oscCtx.stroke();

  oscHistory.forEach((series, sIdx) => {
    if (series.length < 2) return;
    oscCtx.strokeStyle = oscColors[sIdx];
    oscCtx.lineWidth = 1.2;
    oscCtx.beginPath();
    const stepX = w / (maxOscSamples - 1);
    series.forEach((v, idx) => {
      const x = idx * stepX;
      const y = h / 2 - (v * (h * 0.42));
      if (idx === 0) oscCtx.moveTo(x, y);
      else oscCtx.lineTo(x, y);
    });
    oscCtx.stroke();
  });
}

function drawLearningCurve() {
  if (!learnCanvas || !learnCtx) return;
  const w = learnCanvas.width;
  const h = learnCanvas.height;
  learnCtx.fillStyle = "#090d14";
  learnCtx.fillRect(0, 0, w, h);

  learnCtx.strokeStyle = "rgba(255, 255, 255, 0.05)";
  learnCtx.lineWidth = 1;
  learnCtx.beginPath();
  learnCtx.moveTo(0, h / 2);
  learnCtx.lineTo(w, h / 2);
  learnCtx.stroke();

  const losses = neuralEngine.recentLosses;
  const rewards = neuralEngine.recentRewards;
  if (losses.length < 2) return;

  const stepX = w / 50;

  // 1. Draw TD Error curve (Sky blue #38bdf8)
  learnCtx.strokeStyle = "#38bdf8";
  learnCtx.lineWidth = 1.2;
  learnCtx.beginPath();
  losses.forEach((loss, idx) => {
    const x = idx * stepX;
    const y = Math.max(4, Math.min(h - 4, h * 0.75 - loss * 14));
    if (idx === 0) learnCtx.moveTo(x, y);
    else learnCtx.lineTo(x, y);
  });
  learnCtx.stroke();

  // 2. Draw Step Reward curve (Emerald green #10b981)
  if (rewards.length >= 2) {
    learnCtx.strokeStyle = "#10b981";
    learnCtx.lineWidth = 1.2;
    learnCtx.beginPath();
    rewards.forEach((r, idx) => {
      const x = idx * stepX;
      const y = Math.max(4, Math.min(h - 4, h * 0.5 - (r - 1.0) * 10));
      if (idx === 0) learnCtx.moveTo(x, y);
      else learnCtx.lineTo(x, y);
    });
    learnCtx.stroke();
  }
}

function drawNeuralActivationMatrix() {
  if (!matrixCanvas || !mCtx) return;
  const w = matrixCanvas.width;
  const h = matrixCanvas.height;
  mCtx.fillStyle = "#090d14";
  mCtx.fillRect(0, 0, w, h);

  const cols = 16;
  const rows = 4;
  const padX = 3;
  const padY = 3;
  const cellW = (w - padX * (cols + 1)) / cols;
  const cellH = (h - padY * (rows + 1)) / rows;

  if (matrixBadge) {
    matrixBadge.textContent = "64 units";
  }

  // Inter-layer connection traces
  mCtx.strokeStyle = "rgba(255, 255, 255, 0.04)";
  mCtx.lineWidth = 0.8;
  for (let c = 0; c < cols - 1; c += 2) {
    const x1 = padX + c * (cellW + padX) + cellW / 2;
    const y1 = padY + 1 * (cellH + padY) + cellH / 2;
    const x2 = padX + (c + 1) * (cellW + padX) + cellW / 2;
    const y2 = padY + 2 * (cellH + padY) + cellH / 2;
    mCtx.beginPath();
    mCtx.moveTo(x1, y1);
    mCtx.lineTo(x2, y2);
    mCtx.stroke();
  }

  const mask1 = neuralEngine.policy.maskH1;
  const mask2 = neuralEngine.policy.maskH2;

  for (let i = 0; i < 64; i++) {
    const col = i % cols;
    const row = Math.floor(i / cols);
    const x = padX + col * (cellW + padX);
    const y = padY + row * (cellH + padY);

    const isLayer1 = (i < 32);
    const unitIdx = isLayer1 ? i : (i - 32);
    const isMasked = isLayer1 ? (mask1[unitIdx] === 0) : (mask2[unitIdx] === 0);

    const baseFreq = 4.0 + (i % 5) * 1.2;
    const act = 0.5 + 0.5 * Math.sin(state.time * baseFreq + i * 0.4);

    if (isMasked && state.status !== "healthy") {
      mCtx.fillStyle = "rgba(220, 38, 38, 0.15)";
      mCtx.strokeStyle = "rgba(220, 38, 38, 0.7)";
      mCtx.lineWidth = 1.0;
      mCtx.fillRect(x, y, cellW, cellH);
      mCtx.strokeRect(x, y, cellW, cellH);
      mCtx.strokeStyle = "rgba(220, 38, 38, 0.85)";
      mCtx.beginPath();
      mCtx.moveTo(x + 2, y + 2);
      mCtx.lineTo(x + cellW - 2, y + cellH - 2);
      mCtx.moveTo(x + cellW - 2, y + 2);
      mCtx.lineTo(x + 2, y + cellH - 2);
      mCtx.stroke();
    } else if (neuralEngine.isAdapting) {
      // Compensatory active plasticity firing
      const isCompensating = (i % 3 === 0);
      mCtx.fillStyle = isCompensating
        ? `rgba(245, 158, 11, ${0.25 + 0.5 * act})`
        : `rgba(2, 132, 199, ${0.15 + 0.45 * act})`;
      mCtx.strokeStyle = isCompensating ? "rgba(245, 158, 11, 0.7)" : "rgba(2, 132, 199, 0.5)";
      mCtx.lineWidth = 1.0;
      mCtx.fillRect(x, y, cellW, cellH);
      mCtx.strokeRect(x, y, cellW, cellH);
    } else if (state.status === "recovered") {
      const isAlt = (i % 2 === 0);
      mCtx.fillStyle = isAlt
        ? `rgba(13, 148, 136, ${0.15 + 0.45 * act})`
        : `rgba(2, 132, 199, ${0.15 + 0.45 * act})`;
      mCtx.strokeStyle = isAlt ? "rgba(13, 148, 136, 0.5)" : "rgba(2, 132, 199, 0.5)";
      mCtx.lineWidth = 1.0;
      mCtx.fillRect(x, y, cellW, cellH);
      mCtx.strokeRect(x, y, cellW, cellH);
    } else {
      mCtx.fillStyle = `rgba(2, 132, 199, ${0.12 + 0.45 * act})`;
      mCtx.strokeStyle = "rgba(2, 132, 199, 0.4)";
      mCtx.lineWidth = 1.0;
      mCtx.fillRect(x, y, cellW, cellH);
      mCtx.strokeRect(x, y, cellW, cellH);
    }
  }
}

// ──────────────────────────────────────────────
// Main Animation Loop
// ──────────────────────────────────────────────
let lastTime = performance.now();

function animate() {
  requestAnimationFrame(animate);
  const now = performance.now();
  const delta = Math.min(0.1, (now - lastTime) / 1000);
  lastTime = now;

  updateWalkingKinematics(delta);
  updateTelemetryUI();
  updateCameraRig();

  controls.update();
  renderer.render(scene, camera);
}

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// Initialize
animate();
