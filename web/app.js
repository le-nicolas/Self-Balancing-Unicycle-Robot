import * as THREE from "./vendor/three.module.js";
import { OrbitControls } from "./vendor/OrbitControls.js";
import loadMujoco from "./vendor/mujoco_wasm.js";

THREE.Object3D.DEFAULT_UP.set(0, 0, 1);

const SUPPORT_RADIUS_M = 0.145;
const COM_FAIL_STEPS = 15;
const CRASH_ANGLE_RAD = 0.43;
const STABLE_CONFIRM_S = 4.0;
const MUJOCO_LOAD_TIMEOUT_MS = 45000;
const MODEL_XML_PATH = "../final/final.xml";

const K_DU = [
  [-1115.086114, -162.0646188, -6.232475209, -0.2375496975, 3.012484216, 12.35882966, -0.6579379636, 4.91174706, -0.2437628147, 0.596297455, 0.001105790165, -0.00003663409173],
  [-2.990833599, -0.10448059, -0.02685189045, -0.003530024176, -0.0004612034575, 181.8491029, -1.92773575, 1.87272461, 0.0109954547, 0.0000764481337, 0.5517559189, 0.000002178081547],
  [0.05421640515, -18.57332974, 0.001329465557, 0.03151928863, 0.00006153184554, 5.680097555, 0.1997352114, 0.02180464311, 0.02397373605, -0.0000009948775989, 0.0000008555870347, 0.3999954113],
];
const MAX_U = [80.0, 10.0, 10.0];
const MAX_DU = [18.210732648355684, 5.886188088635976, 15.0];

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
  ids: null,
  requestedMassKg: Number(ui.massRange.value),
  effectiveMassKg: Number(ui.massRange.value),
  maxStableMassKg: 0.0,
  elapsedS: 0.0,
  comDistM: 0.0,
  comFailStreak: 0,
  failed: false,
  failureReason: "",
  paused: false,
  stableRecorded: false,
  stepsPerFrame: 8,
  uApplied: [0.0, 0.0, 0.0],
  scene: null,
  camera: null,
  renderer: null,
  controls: null,
  geomVisuals: [],
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
  ui.applyBtn.addEventListener("click", () => {
    rebuildSimulation(Number(ui.massNumber.value)).catch((err) => {
      console.error(err);
      setStatus(`Rebuild failed: ${err.message}`, true);
    });
  });
  ui.pauseBtn.addEventListener("click", () => {
    sim.paused = !sim.paused;
    ui.pauseBtn.textContent = sim.paused ? "Resume" : "Pause";
    if (sim.paused && !sim.failed) {
      setStatus("Paused", false);
    } else if (!sim.failed) {
      setStatus("Running", false);
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
  for (let i = 0; i < 6; i += 1) {
    spawnRandomGroundProp();
  }
  await rebuildSimulation(Number(ui.massRange.value));
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
  sim.controls.minPolarAngle = 0.06;
  sim.controls.maxPolarAngle = Math.PI * 0.49;

  const keyLight = new THREE.DirectionalLight(0xffffff, 1.1);
  keyLight.position.set(2.2, -2.2, 4.0);
  keyLight.castShadow = true;
  sim.scene.add(keyLight);
  sim.scene.add(new THREE.AmbientLight(0x89a9cf, 0.52));

  sim.robotDropTargets = [];

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
  if (!sim.camera || !sim.scene) {
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

async function rebuildSimulation(requestedMassKg) {
  if (!sim.mujoco || !sim.modelXmlText) {
    return;
  }
  const clampedMass = clamp(requestedMassKg, 0.0, 3.0);
  sim.requestedMassKg = clampedMass;
  sim.effectiveMassKg = Math.max(clampedMass, 1e-6);
  ui.massRange.value = clampedMass.toFixed(2);
  ui.massNumber.value = clampedMass.toFixed(2);

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
  sim.ids = resolveIds(sim.model);
  setPayloadMassRuntime(sim.effectiveMassKg);
  buildGeomVisualsFromModel();
  resetState();

  sim.failed = false;
  sim.failureReason = "";
  sim.comFailStreak = 0;
  sim.uApplied = [0.0, 0.0, 0.0];
  sim.comDistM = computeComDistance();
  sim.elapsedS = 0.0;
  sim.stableRecorded = false;
  sim.paused = false;
  ui.pauseBtn.textContent = "Pause";
  updateHud();
  setStatus("Running", false);
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

function resolveIds(model) {
  const mjtObj = sim.mujoco.mjtObj;
  const OBJ_JOINT = mjtObj.mjOBJ_JOINT.value;
  const OBJ_ACTUATOR = mjtObj.mjOBJ_ACTUATOR.value;
  const OBJ_BODY = mjtObj.mjOBJ_BODY.value;
  const OBJ_GEOM = mjtObj.mjOBJ_GEOM.value;
  const jidPitch = sim.mujoco.mj_name2id(model, OBJ_JOINT, "stick_pitch");
  const jidRoll = sim.mujoco.mj_name2id(model, OBJ_JOINT, "stick_roll");
  const jidRw = sim.mujoco.mj_name2id(model, OBJ_JOINT, "wheel_spin");
  const jidBaseX = sim.mujoco.mj_name2id(model, OBJ_JOINT, "base_x_slide");
  const jidBaseY = sim.mujoco.mj_name2id(model, OBJ_JOINT, "base_y_slide");

  return {
    qPitch: model.jnt_qposadr[jidPitch],
    qRoll: model.jnt_qposadr[jidRoll],
    qBaseX: model.jnt_qposadr[jidBaseX],
    qBaseY: model.jnt_qposadr[jidBaseY],
    vPitch: model.jnt_dofadr[jidPitch],
    vRoll: model.jnt_dofadr[jidRoll],
    vRw: model.jnt_dofadr[jidRw],
    vBaseX: model.jnt_dofadr[jidBaseX],
    vBaseY: model.jnt_dofadr[jidBaseY],
    aidRw: sim.mujoco.mj_name2id(model, OBJ_ACTUATOR, "wheel_spin"),
    aidBaseX: sim.mujoco.mj_name2id(model, OBJ_ACTUATOR, "base_x_force"),
    aidBaseY: sim.mujoco.mj_name2id(model, OBJ_ACTUATOR, "base_y_force"),
    bidBaseY: sim.mujoco.mj_name2id(model, OBJ_BODY, "base_y"),
    bidStick: sim.mujoco.mj_name2id(model, OBJ_BODY, "stick"),
    bidWheel: sim.mujoco.mj_name2id(model, OBJ_BODY, "wheel"),
    bidPayload: sim.mujoco.mj_name2id(model, OBJ_BODY, "payload"),
    gidPayloadGeom: sim.mujoco.mj_name2id(model, OBJ_GEOM, "payload_geom"),
  };
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
  const OBJ_BODY = sim.mujoco.mjtObj.mjOBJ_BODY.value;
  for (let gid = 0; gid < sim.model.ngeom; gid += 1) {
    const type = Number(sim.model.geom_type[gid]);
    const bodyId = Number(sim.model.geom_bodyid[gid]);
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

    const lp = 3 * gid;
    const lq = 4 * gid;
    sim.geomVisuals.push({
      mesh,
      geomId: gid,
      bodyId,
      localPos: new THREE.Vector3(
        Number(sim.model.geom_pos[lp + 0]),
        Number(sim.model.geom_pos[lp + 1]),
        Number(sim.model.geom_pos[lp + 2]),
      ),
      localQuat: normalizeQuat(
        quatFromMujocoWxyz(
          Number(sim.model.geom_quat[lq + 0]),
          Number(sim.model.geom_quat[lq + 1]),
          Number(sim.model.geom_quat[lq + 2]),
          Number(sim.model.geom_quat[lq + 3]),
        ),
      ),
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
  if (style.alpha <= 0.01) {
    material.visible = false;
  }
  return new THREE.Mesh(geometry, material);
}

function quatFromMujocoWxyz(w, x, y, z) {
  return new THREE.Quaternion(x, y, z, w);
}

function normalizeQuat(q) {
  if (q.lengthSq() < 1e-12) {
    return q.identity();
  }
  return q.normalize();
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

function setPayloadMassRuntime(payloadMassKg) {
  const massTarget = Math.max(payloadMassKg, 0.0);
  const massRuntime = Math.max(massTarget, 1e-6);
  const g3 = 3 * sim.ids.gidPayloadGeom;
  const sx = sim.model.geom_size[g3];
  const sy = sim.model.geom_size[g3 + 1];
  const sz = sim.model.geom_size[g3 + 2];
  const ixx = (massRuntime / 3.0) * (sy * sy + sz * sz);
  const iyy = (massRuntime / 3.0) * (sx * sx + sz * sz);
  const izz = (massRuntime / 3.0) * (sx * sx + sy * sy);
  sim.model.body_mass[sim.ids.bidPayload] = massRuntime;
  const b3 = 3 * sim.ids.bidPayload;
  sim.model.body_inertia[b3] = ixx;
  sim.model.body_inertia[b3 + 1] = iyy;
  sim.model.body_inertia[b3 + 2] = izz;
  sim.mujoco.mj_setConst(sim.model, sim.data);
  sim.mujoco.mj_forward(sim.model, sim.data);
}

function resetState() {
  sim.data.qpos.fill(0);
  sim.data.qvel.fill(0);
  sim.data.ctrl.fill(0);
  sim.uApplied = [0.0, 0.0, 0.0];
  sim.data.qpos[sim.ids.qRoll] = 0.02;
  sim.mujoco.mj_forward(sim.model, sim.data);
}

function controllerStep() {
  const x = [
    sim.data.qpos[sim.ids.qPitch],
    sim.data.qpos[sim.ids.qRoll],
    sim.data.qvel[sim.ids.vPitch],
    sim.data.qvel[sim.ids.vRoll],
    sim.data.qvel[sim.ids.vRw],
    sim.data.qpos[sim.ids.qBaseX],
    sim.data.qpos[sim.ids.qBaseY],
    sim.data.qvel[sim.ids.vBaseX],
    sim.data.qvel[sim.ids.vBaseY],
  ];
  const z = [...x, ...sim.uApplied];
  const duCmd = [0.0, 0.0, 0.0];
  for (let row = 0; row < 3; row += 1) {
    let acc = 0.0;
    for (let col = 0; col < 12; col += 1) {
      acc += K_DU[row][col] * z[col];
    }
    duCmd[row] = -acc;
  }
  const du = [
    clamp(duCmd[0], -MAX_DU[0], MAX_DU[0]),
    clamp(duCmd[1], -MAX_DU[1], MAX_DU[1]),
    clamp(duCmd[2], -MAX_DU[2], MAX_DU[2]),
  ];
  const u = [
    clamp(sim.uApplied[0] + du[0], -MAX_U[0], MAX_U[0]),
    clamp(sim.uApplied[1] + du[1], -MAX_U[1], MAX_U[1]),
    clamp(sim.uApplied[2] + du[2], -MAX_U[2], MAX_U[2]),
  ];
  sim.uApplied = u;
  sim.data.ctrl[sim.ids.aidRw] = u[0];
  sim.data.ctrl[sim.ids.aidBaseX] = u[1];
  sim.data.ctrl[sim.ids.aidBaseY] = u[2];
}

function computeComDistance() {
  let totalMass = 0.0;
  let sumX = 0.0;
  let sumY = 0.0;
  for (let bodyId = 1; bodyId < sim.model.nbody; bodyId += 1) {
    const mass = sim.model.body_mass[bodyId];
    totalMass += mass;
    sumX += mass * sim.data.xipos[3 * bodyId];
    sumY += mass * sim.data.xipos[3 * bodyId + 1];
  }
  if (totalMass <= 1e-12) {
    return 0.0;
  }
  const comX = sumX / totalMass;
  const comY = sumY / totalMass;
  const baseX = sim.data.xpos[3 * sim.ids.bidBaseY];
  const baseY = sim.data.xpos[3 * sim.ids.bidBaseY + 1];
  return Math.hypot(comX - baseX, comY - baseY);
}

function stepSimulation() {
  controllerStep();
  sim.mujoco.mj_step(sim.model, sim.data);
  sim.elapsedS += sim.model.opt.timestep;
  sim.comDistM = computeComDistance();

  const pitch = sim.data.qpos[sim.ids.qPitch];
  const roll = sim.data.qpos[sim.ids.qRoll];
  const tiltFail = Math.abs(pitch) >= CRASH_ANGLE_RAD || Math.abs(roll) >= CRASH_ANGLE_RAD;

  if (sim.comDistM > SUPPORT_RADIUS_M) {
    sim.comFailStreak += 1;
  } else {
    sim.comFailStreak = 0;
  }
  const comFail = sim.comFailStreak >= COM_FAIL_STEPS;

  if (tiltFail || comFail) {
    sim.failed = true;
    sim.failureReason = comFail ? "COM overload" : "Tilt overload";
    setStatus(`Failed: ${sim.failureReason}`, true);
  } else if (!sim.stableRecorded && sim.elapsedS >= STABLE_CONFIRM_S) {
    sim.stableRecorded = true;
    sim.maxStableMassKg = Math.max(sim.maxStableMassKg, sim.requestedMassKg);
    setStatus("Stable", false);
  }
}

function animate() {
  requestAnimationFrame(animate);
  if (sim.model && sim.data && !sim.paused && !sim.failed) {
    for (let i = 0; i < sim.stepsPerFrame; i += 1) {
      stepSimulation();
      if (sim.failed) {
        break;
      }
    }
  }
  if (sim.model && sim.data) {
    syncVisuals();
    updateHud();
  }
  if (sim.controls) {
    sim.controls.update();
  }
  if (sim.renderer && sim.scene && sim.camera) {
    sim.renderer.render(sim.scene, sim.camera);
  }
}

function syncVisuals() {
  const hasGeomWorld = Boolean(sim.data.geom_xpos) && Boolean(sim.data.geom_xmat);
  const rot = new THREE.Matrix4();

  for (const entry of sim.geomVisuals) {
    if (hasGeomWorld) {
      const g3 = 3 * entry.geomId;
      const g9 = 9 * entry.geomId;
      const gx = Number(sim.data.geom_xpos[g3 + 0]);
      const gy = Number(sim.data.geom_xpos[g3 + 1]);
      const gz = Number(sim.data.geom_xpos[g3 + 2]);
      rot.set(
        Number(sim.data.geom_xmat[g9 + 0]), Number(sim.data.geom_xmat[g9 + 1]), Number(sim.data.geom_xmat[g9 + 2]), 0,
        Number(sim.data.geom_xmat[g9 + 3]), Number(sim.data.geom_xmat[g9 + 4]), Number(sim.data.geom_xmat[g9 + 5]), 0,
        Number(sim.data.geom_xmat[g9 + 6]), Number(sim.data.geom_xmat[g9 + 7]), Number(sim.data.geom_xmat[g9 + 8]), 0,
        0, 0, 0, 1,
      );
      entry.mesh.position.set(gx, gy, gz);
      entry.mesh.quaternion.setFromRotationMatrix(rot);
      continue;
    }

    const b3 = 3 * entry.bodyId;
    const b4 = 4 * entry.bodyId;
    const bodyPos = new THREE.Vector3(
      Number(sim.data.xpos[b3 + 0]),
      Number(sim.data.xpos[b3 + 1]),
      Number(sim.data.xpos[b3 + 2]),
    );
    const bodyQuat = normalizeQuat(
      quatFromMujocoWxyz(
        Number(sim.data.xquat[b4 + 0]),
        Number(sim.data.xquat[b4 + 1]),
        Number(sim.data.xquat[b4 + 2]),
        Number(sim.data.xquat[b4 + 3]),
      ),
    );
    const worldPos = entry.localPos.clone().applyQuaternion(bodyQuat).add(bodyPos);
    const worldQuat = bodyQuat.clone().multiply(entry.localQuat);
    entry.mesh.position.copy(worldPos);
    entry.mesh.quaternion.copy(worldQuat);
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

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function rand(min, max) {
  return min + Math.random() * (max - min);
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
