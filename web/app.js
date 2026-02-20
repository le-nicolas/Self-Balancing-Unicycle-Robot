import * as THREE from "./vendor/three.module.js";
import { OrbitControls } from "./vendor/OrbitControls.js";
import loadMujoco from "./vendor/mujoco_wasm.js";

THREE.Object3D.DEFAULT_UP.set(0, 0, 1);

const MUJOCO_LOAD_TIMEOUT_MS = 45000;
const MODEL_XML_PATH = "../final/final.xml";
const API_BASE = "/api";
const BACKEND_POLL_MS = 33;

const ui = {
  canvas: document.getElementById("simCanvas"),
  massRange: document.getElementById("massRange"),
  massNumber: document.getElementById("massNumber"),
  applyBtn: document.getElementById("applyBtn"),
  pauseBtn: document.getElementById("pauseBtn"),
  spawnPropBtn: document.getElementById("spawnPropBtn"),
  clearPropsBtn: document.getElementById("clearPropsBtn"),
  statusValue: document.getElementById("statusValue"),
  elapsedValue: document.getElementById("elapsedValue"),
  comValue: document.getElementById("comValue"),
  maxStableValue: document.getElementById("maxStableValue"),
};

const sim = {
  mujoco: null,
  modelXmlText: "",
  model: null,
  data: null,
  requestedMassKg: Number(ui.massRange.value),
  effectiveMassKg: Number(ui.massRange.value),
  maxStableMassKg: 0.0,
  elapsedS: 0.0,
  comDistM: 0.0,
  failed: false,
  failureReason: "",
  paused: false,
  scene: null,
  camera: null,
  renderer: null,
  controls: null,
  geomVisuals: [],
  backendState: null,
  statePollHandle: null,
  statePollInFlight: false,
  raycaster: new THREE.Raycaster(),
  pointerNdc: new THREE.Vector2(),
  dragPlane: new THREE.Plane(new THREE.Vector3(0, 0, 1), 0),
  dragPoint: new THREE.Vector3(),
  dragOffset: new THREE.Vector3(),
  draggingProp: null,
  props: [],
  robotDropTargets: [],
};

initUiBindings();
boot().catch((err) => {
  console.error(err);
  setStatus(`Load failed: ${err.message}`, true);
});

function initUiBindings() {
  ui.massRange.addEventListener("input", () => {
    ui.massNumber.value = ui.massRange.value;
  });

  ui.massNumber.addEventListener("input", () => {
    const parsed = Number(ui.massNumber.value);
    if (!Number.isFinite(parsed)) {
      return;
    }
    const clamped = clamp(parsed, Number(ui.massRange.min), Number(ui.massRange.max));
    ui.massRange.value = clamped.toFixed(2);
    ui.massNumber.value = clamped.toFixed(2);
  });

  ui.applyBtn.addEventListener("click", async () => {
    try {
      await requestResetMass(Number(ui.massNumber.value));
    } catch (err) {
      console.error(err);
      setStatus(`Reset failed: ${err.message}`, true);
    }
  });

  ui.pauseBtn.addEventListener("click", async () => {
    try {
      await requestPauseToggle();
    } catch (err) {
      console.error(err);
      setStatus(`Pause failed: ${err.message}`, true);
    }
  });

  ui.spawnPropBtn.addEventListener("click", () => {
    spawnRandomGroundProp();
  });

  ui.clearPropsBtn.addEventListener("click", () => {
    clearGroundProps();
  });

  ui.canvas.addEventListener("pointerdown", onPointerDown);
  window.addEventListener("pointermove", onPointerMove);
  window.addEventListener("pointerup", onPointerUp);
  window.addEventListener("resize", onResize);
}

async function boot() {
  setStatus("Loading final model XML...", false);
  const modelResp = await fetch(MODEL_XML_PATH);
  if (!modelResp.ok) {
    throw new Error(`Cannot load ${MODEL_XML_PATH} (${modelResp.status})`);
  }
  sim.modelXmlText = await modelResp.text();

  setStatus("Loading MuJoCo WebAssembly...", false);
  sim.mujoco = await withTimeout(loadMujoco(), MUJOCO_LOAD_TIMEOUT_MS, "MuJoCo load timed out");
  configureVirtualFs();

  initThreeScene();
  loadModelForVisuals();

  for (let i = 0; i < 6; i += 1) {
    spawnRandomGroundProp();
  }

  setStatus("Connecting Python final.py runtime...", false);
  await requestResetMass(Number(ui.massRange.value));

  sim.statePollHandle = window.setInterval(() => {
    fetchBackendState().catch((err) => {
      console.error(err);
      setStatus(`State stream failed: ${err.message}`, true);
    });
  }, BACKEND_POLL_MS);

  animate();
}

function initThreeScene() {
  sim.scene = new THREE.Scene();
  sim.scene.background = new THREE.Color(0x111a2b);
  sim.scene.fog = new THREE.Fog(0x111a2b, 5.0, 24.0);

  sim.camera = new THREE.PerspectiveCamera(48, 1, 0.01, 80);
  sim.camera.up.set(0, 0, 1);
  sim.camera.position.set(0.0, -3.8, 1.55);
  sim.camera.lookAt(0.0, 0.0, 0.45);

  sim.renderer = new THREE.WebGLRenderer({ canvas: ui.canvas, antialias: true });
  sim.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2.0));
  sim.renderer.outputColorSpace = THREE.SRGBColorSpace;
  sim.renderer.shadowMap.enabled = true;

  sim.controls = new OrbitControls(sim.camera, sim.renderer.domElement);
  sim.controls.target.set(0.0, 0.0, 0.45);
  sim.controls.enableDamping = true;
  sim.controls.enablePan = true;
  sim.controls.minDistance = 0.7;
  sim.controls.maxDistance = 12.0;
  sim.controls.minPolarAngle = 0.01;
  sim.controls.maxPolarAngle = Math.PI - 0.01;

  const keyLight = new THREE.DirectionalLight(0xffffff, 1.1);
  keyLight.position.set(2.2, -2.2, 4.0);
  keyLight.castShadow = true;
  sim.scene.add(keyLight);
  sim.scene.add(new THREE.AmbientLight(0x89a9cf, 0.52));

  onResize();
}

function onResize() {
  if (!sim.renderer || !sim.camera) {
    return;
  }
  const { clientWidth, clientHeight } = ui.canvas;
  const width = Math.max(clientWidth, 320);
  const height = Math.max(clientHeight, 220);
  sim.camera.aspect = width / height;
  sim.camera.updateProjectionMatrix();
  sim.renderer.setSize(width, height, false);
}

function pointerToNdc(event) {
  const rect = ui.canvas.getBoundingClientRect();
  const x = ((event.clientX - rect.left) / Math.max(rect.width, 1)) * 2.0 - 1.0;
  const y = -(((event.clientY - rect.top) / Math.max(rect.height, 1)) * 2.0 - 1.0);
  sim.pointerNdc.set(x, y);
}

function onPointerDown(event) {
  if (!sim.camera) {
    return;
  }
  pointerToNdc(event);
  sim.raycaster.setFromCamera(sim.pointerNdc, sim.camera);
  const hits = sim.raycaster.intersectObjects(sim.props, false);
  if (hits.length === 0) {
    return;
  }
  sim.draggingProp = hits[0].object;
  const groundZ = Number(sim.draggingProp.userData.groundZ ?? 0.0);
  sim.dragPlane.set(new THREE.Vector3(0, 0, 1), -groundZ);
  if (sim.raycaster.ray.intersectPlane(sim.dragPlane, sim.dragPoint)) {
    sim.dragOffset.copy(sim.draggingProp.position).sub(sim.dragPoint);
  } else {
    sim.dragOffset.set(0, 0, 0);
  }
  if (sim.controls) {
    sim.controls.enabled = false;
  }
}

function onPointerMove(event) {
  if (!sim.draggingProp || !sim.camera) {
    return;
  }
  pointerToNdc(event);
  sim.raycaster.setFromCamera(sim.pointerNdc, sim.camera);

  const botHits = sim.raycaster.intersectObjects(sim.robotDropTargets, false);
  const halfHeight = Number(sim.draggingProp.userData.halfHeight ?? 0.04);
  if (botHits.length > 0 && botHits[0].point.z > 0.08) {
    const hit = botHits[0];
    const normal = hit.face
      ? hit.face.normal.clone().transformDirection(hit.object.matrixWorld)
      : new THREE.Vector3(0, 0, 1);
    if (normal.z < 0.25) {
      normal.set(0, 0, 1);
    }
    const place = hit.point.clone().addScaledVector(normal, halfHeight + 0.01);
    sim.draggingProp.position.copy(place);
    sim.draggingProp.userData.groundZ = place.z;
    return;
  }

  if (!sim.raycaster.ray.intersectPlane(sim.dragPlane, sim.dragPoint)) {
    return;
  }
  const nextPos = sim.dragPoint.clone().add(sim.dragOffset);
  nextPos.x = clamp(nextPos.x, -2.6, 2.6);
  nextPos.y = clamp(nextPos.y, -2.6, 2.6);
  nextPos.z = Number(sim.draggingProp.userData.groundZ ?? nextPos.z);
  sim.draggingProp.position.copy(nextPos);
}

function onPointerUp() {
  if (sim.draggingProp && sim.controls) {
    sim.controls.enabled = true;
  }
  sim.draggingProp = null;
}

function spawnRandomGroundProp() {
  if (!sim.scene) {
    return;
  }
  const pick = Math.floor(Math.random() * 3);
  let geom = null;
  let groundZ = 0.0;
  let halfHeight = 0.04;

  if (pick === 0) {
    const sx = rand(0.08, 0.18);
    const sy = rand(0.08, 0.20);
    const sz = rand(0.04, 0.10);
    geom = new THREE.BoxGeometry(sx, sy, sz);
    groundZ = sz * 0.5;
    halfHeight = sz * 0.5;
  } else if (pick === 1) {
    const radius = rand(0.04, 0.09);
    geom = new THREE.SphereGeometry(radius, 20, 16);
    groundZ = radius;
    halfHeight = radius;
  } else {
    const radius = rand(0.03, 0.06);
    const len = rand(0.10, 0.20);
    geom = new THREE.CylinderGeometry(radius, radius, len, 20);
    groundZ = radius;
    halfHeight = radius;
  }

  const mat = new THREE.MeshStandardMaterial({
    color: new THREE.Color().setHSL(rand(0.03, 0.16), rand(0.55, 0.78), rand(0.48, 0.66)),
    roughness: rand(0.35, 0.75),
    metalness: rand(0.05, 0.35),
  });

  const mesh = new THREE.Mesh(geom, mat);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  if (pick === 2) {
    mesh.rotation.z = rand(0.0, Math.PI);
  }

  const angle = rand(0, Math.PI * 2.0);
  const radiusRing = rand(0.45, 2.1);
  mesh.position.set(Math.cos(angle) * radiusRing, Math.sin(angle) * radiusRing, groundZ);
  mesh.userData.draggable = true;
  mesh.userData.groundZ = groundZ;
  mesh.userData.halfHeight = halfHeight;

  sim.scene.add(mesh);
  sim.props.push(mesh);
}

function clearGroundProps() {
  for (const mesh of sim.props) {
    if (mesh.parent) {
      mesh.parent.remove(mesh);
    }
    mesh.geometry.dispose();
    mesh.material.dispose();
  }
  sim.props = [];
}

async function requestResetMass(requestedMassKg) {
  const clampedMass = clamp(requestedMassKg, 0.0, 3.0);
  sim.requestedMassKg = clampedMass;
  ui.massRange.value = clampedMass.toFixed(2);
  ui.massNumber.value = clampedMass.toFixed(2);

  setStatus("Resetting backend runtime...", false);
  const state = await postJson(`${API_BASE}/reset`, {
    payload_mass_kg: clampedMass,
  });
  consumeBackendState(state);
}

async function requestPauseToggle() {
  const state = await postJson(`${API_BASE}/pause`, {
    paused: !sim.paused,
  });
  consumeBackendState(state);
}

async function fetchBackendState() {
  if (sim.statePollInFlight) {
    return;
  }
  sim.statePollInFlight = true;
  try {
    const resp = await fetch(`${API_BASE}/state`, {
      cache: "no-store",
      headers: { "Cache-Control": "no-cache" },
    });
    if (!resp.ok) {
      throw new Error(`state endpoint returned ${resp.status}`);
    }
    const state = await resp.json();
    consumeBackendState(state);
  } finally {
    sim.statePollInFlight = false;
  }
}

function consumeBackendState(state) {
  sim.backendState = state;
  sim.paused = Boolean(state.paused);
  sim.failed = Boolean(state.failed);
  sim.failureReason = String(state.failure_reason ?? "");
  sim.elapsedS = Number(state.elapsed_s ?? sim.elapsedS);
  sim.comDistM = Number(state.com_dist_m ?? sim.comDistM);
  sim.maxStableMassKg = Number(state.max_stable_mass_kg ?? sim.maxStableMassKg);
  sim.effectiveMassKg = Number(state.effective_mass_kg ?? sim.effectiveMassKg);

  ui.pauseBtn.textContent = sim.paused ? "Resume" : "Pause";
  const statusText = String(
    state.status ?? (sim.failed ? `Failed: ${sim.failureReason}` : sim.paused ? "Paused" : "Running"),
  );
  setStatus(statusText, sim.failed);
  updateHud();
}

function loadModelForVisuals() {
  if (!sim.mujoco || !sim.modelXmlText) {
    return;
  }
  if (sim.data) {
    sim.data.delete();
    sim.data = null;
  }
  if (sim.model) {
    sim.model.delete();
    sim.model = null;
  }

  const modelPath = "/working/final.xml";
  ensureFsDir("/working");
  try {
    sim.mujoco.FS.unlink(modelPath);
  } catch {}
  sim.mujoco.FS.writeFile(modelPath, sim.modelXmlText);

  sim.model = sim.mujoco.MjModel.loadFromXML(modelPath);
  sim.data = new sim.mujoco.MjData(sim.model);
  buildGeomVisualsFromModel();
}

function ensureFsDir(path) {
  try {
    sim.mujoco.FS.stat(path);
  } catch {
    sim.mujoco.FS.mkdir(path);
  }
}

function configureVirtualFs() {
  ensureFsDir("/working");
  try {
    sim.mujoco.FS.mount(sim.mujoco.MEMFS, { root: "." }, "/working");
  } catch (err) {
    const msg = String(err && err.message ? err.message : err).toLowerCase();
    if (!msg.includes("already mounted") && !msg.includes("busy")) {
      throw err;
    }
  }
}

function clearGeomVisuals() {
  for (const entry of sim.geomVisuals) {
    const mesh = entry.mesh;
    if (mesh.parent) {
      mesh.parent.remove(mesh);
    }
    mesh.geometry.dispose();
    mesh.material.dispose();
  }
  sim.geomVisuals = [];
  sim.robotDropTargets = [];
}

function buildGeomVisualsFromModel() {
  if (!sim.model || !sim.scene) {
    return;
  }
  clearGeomVisuals();
  const OBJ_BODY = enumToInt(sim.mujoco.mjtObj.mjOBJ_BODY);
  const OBJ_GEOM = enumToInt(sim.mujoco.mjtObj.mjOBJ_GEOM);

  for (let gid = 0; gid < sim.model.ngeom; gid += 1) {
    const type = Number(sim.model.geom_type[gid]);
    const bodyId = Number(sim.model.geom_bodyid[gid]);
    const geomName = sim.mujoco.mj_id2name(sim.model, OBJ_GEOM, gid) || "";
    if (geomName === "payload_geom") {
      continue;
    }
    const style = getGeomStyle(gid);
    if (style.alpha <= 0.01) {
      continue;
    }

    const s = 3 * gid;
    const sx = Number(sim.model.geom_size[s + 0]);
    const sy = Number(sim.model.geom_size[s + 1]);
    const sz = Number(sim.model.geom_size[s + 2]);

    const mesh = createMeshForGeom(type, sx, sy, sz, style);
    if (!mesh) {
      continue;
    }

    mesh.castShadow = type !== 0;
    mesh.receiveShadow = true;
    sim.scene.add(mesh);

    sim.geomVisuals.push({
      geomId: gid,
      bodyId,
      mesh,
    });

    const bodyName = sim.mujoco.mj_id2name(sim.model, OBJ_BODY, bodyId) || "";
    if (isRobotBodyName(bodyName) && type !== 0) {
      sim.robotDropTargets.push(mesh);
    }
  }
}

function getGeomStyle(gid) {
  const rgbaIndex = 4 * gid;
  let r = Number(sim.model.geom_rgba[rgbaIndex + 0]);
  let g = Number(sim.model.geom_rgba[rgbaIndex + 1]);
  let b = Number(sim.model.geom_rgba[rgbaIndex + 2]);
  let a = Number(sim.model.geom_rgba[rgbaIndex + 3]);
  let roughness = 0.52;
  let metalness = 0.16;

  const matId = Number(sim.model.geom_matid[gid]);
  if (matId >= 0 && sim.model.mat_rgba) {
    const m = 4 * matId;
    r *= Number(sim.model.mat_rgba[m + 0]);
    g *= Number(sim.model.mat_rgba[m + 1]);
    b *= Number(sim.model.mat_rgba[m + 2]);
    a *= Number(sim.model.mat_rgba[m + 3]);
    if (sim.model.mat_shininess) {
      roughness = clamp(1.0 - Number(sim.model.mat_shininess[matId]) * 0.85, 0.05, 0.98);
    }
    if (sim.model.mat_specular) {
      metalness = clamp(Number(sim.model.mat_specular[matId]) * 0.28, 0.0, 0.42);
    }
  }

  return {
    color: new THREE.Color(clamp(r, 0, 1), clamp(g, 0, 1), clamp(b, 0, 1)),
    alpha: clamp(a, 0, 1),
    roughness,
    metalness,
  };
}

function createMeshForGeom(type, sx, sy, sz, style) {
  let geometry = null;
  if (type === 0) {
    geometry = new THREE.PlaneGeometry(Math.max(2 * sx, 20.0), Math.max(2 * sy, 20.0));
  } else if (type === 2) {
    geometry = new THREE.SphereGeometry(Math.max(sx, 1e-4), 22, 16);
  } else if (type === 3) {
    geometry = new THREE.CapsuleGeometry(Math.max(sx, 1e-4), Math.max(2 * sy, 1e-4), 8, 16);
    geometry.rotateX(Math.PI * 0.5);
  } else if (type === 5) {
    geometry = new THREE.CylinderGeometry(Math.max(sx, 1e-4), Math.max(sx, 1e-4), Math.max(2 * sy, 1e-4), 24);
    geometry.rotateX(Math.PI * 0.5);
  } else if (type === 6) {
    geometry = new THREE.BoxGeometry(Math.max(2 * sx, 1e-4), Math.max(2 * sy, 1e-4), Math.max(2 * sz, 1e-4));
  } else {
    return null;
  }

  const material = new THREE.MeshStandardMaterial({
    color: style.color,
    roughness: style.roughness,
    metalness: style.metalness,
    transparent: style.alpha < 0.999,
    opacity: style.alpha,
    side: type === 0 ? THREE.DoubleSide : THREE.FrontSide,
  });

  return new THREE.Mesh(geometry, material);
}

function isRobotBodyName(name) {
  return (
    name === "base_x" ||
    name === "base_y" ||
    name === "stick_pitch_frame" ||
    name === "stick" ||
    name === "wheel" ||
    name === "payload"
  );
}

function animate() {
  requestAnimationFrame(animate);
  syncVisualsFromBackend();
  updateHud();

  if (sim.controls) {
    sim.controls.update();
  }
  if (sim.renderer && sim.scene && sim.camera) {
    sim.renderer.render(sim.scene, sim.camera);
  }
}

function syncVisualsFromBackend() {
  if (!sim.backendState || sim.geomVisuals.length === 0) {
    return;
  }
  const xpos = sim.backendState.geom_xpos;
  const xmat = sim.backendState.geom_xmat;
  if (!Array.isArray(xpos) || !Array.isArray(xmat)) {
    return;
  }

  const rot = new THREE.Matrix4();
  for (const entry of sim.geomVisuals) {
    const g3 = 3 * entry.geomId;
    const g9 = 9 * entry.geomId;
    if (g3 + 2 >= xpos.length || g9 + 8 >= xmat.length) {
      continue;
    }

    entry.mesh.position.set(
      Number(xpos[g3 + 0]),
      Number(xpos[g3 + 1]),
      Number(xpos[g3 + 2]),
    );

    rot.set(
      Number(xmat[g9 + 0]), Number(xmat[g9 + 1]), Number(xmat[g9 + 2]), 0,
      Number(xmat[g9 + 3]), Number(xmat[g9 + 4]), Number(xmat[g9 + 5]), 0,
      Number(xmat[g9 + 6]), Number(xmat[g9 + 7]), Number(xmat[g9 + 8]), 0,
      0, 0, 0, 1,
    );
    entry.mesh.quaternion.setFromRotationMatrix(rot);
  }
}

function updateHud() {
  ui.elapsedValue.textContent = `${sim.elapsedS.toFixed(2)} s`;
  ui.comValue.textContent = `${sim.comDistM.toFixed(3)} m`;
  ui.maxStableValue.textContent = `${sim.maxStableMassKg.toFixed(2)} kg`;
}

function setStatus(text, danger) {
  ui.statusValue.textContent = text;
  ui.statusValue.classList.toggle("danger", Boolean(danger));
}

async function postJson(url, body) {
  const resp = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body ?? {}),
  });
  if (!resp.ok) {
    let detail = "";
    try {
      const errorJson = await resp.json();
      detail = errorJson.error ? `: ${errorJson.error}` : "";
    } catch {}
    throw new Error(`HTTP ${resp.status}${detail}`);
  }
  return resp.json();
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function rand(min, max) {
  return min + Math.random() * (max - min);
}

function enumToInt(value) {
  if (typeof value === "number" && Number.isInteger(value)) {
    return value;
  }
  if (value && typeof value.value === "number" && Number.isInteger(value.value)) {
    return value.value;
  }
  const parsed = Number(value?.valueOf?.() ?? value);
  if (Number.isInteger(parsed)) {
    return parsed;
  }
  throw new Error(`Cannot convert enum "${String(value)}" to int`);
}

async function withTimeout(promise, timeoutMs, timeoutMessage) {
  let timer = null;
  try {
    return await Promise.race([
      promise,
      new Promise((_, reject) => {
        timer = setTimeout(() => reject(new Error(timeoutMessage)), timeoutMs);
      }),
    ]);
  } finally {
    if (timer !== null) {
      clearTimeout(timer);
    }
  }
}
