import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js";
import { OrbitControls } from "https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/controls/OrbitControls.js";
import loadMujoco from "https://cdn.jsdelivr.net/npm/mujoco-js@0.0.7/dist/mujoco_wasm.js";

const SUPPORT_RADIUS_M = 0.145;
const COM_FAIL_STEPS = 15;
const CRASH_ANGLE_RAD = 0.43;
const STABLE_CONFIRM_S = 4.0;

const CTRL = {
  rwKp: 210.0,
  rwKd: 46.0,
  rwWheelRateKd: 0.18,
  bxKp: 90.0,
  bxKd: 20.0,
  byKp: 72.0,
  byKd: 16.0,
  baseCenterK: 1.8,
  baseDampK: 3.0,
  rwMax: 180.0,
  baseMax: 140.0,
};

const ui = {
  canvas: document.getElementById("simCanvas"),
  massRange: document.getElementById("massRange"),
  massNumber: document.getElementById("massNumber"),
  applyBtn: document.getElementById("applyBtn"),
  pauseBtn: document.getElementById("pauseBtn"),
  statusValue: document.getElementById("statusValue"),
  elapsedValue: document.getElementById("elapsedValue"),
  comValue: document.getElementById("comValue"),
  maxStableValue: document.getElementById("maxStableValue"),
};

const sim = {
  mujoco: null,
  xmlTemplate: "",
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
  scene: null,
  camera: null,
  renderer: null,
  controls: null,
  visuals: {},
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
  window.addEventListener("resize", onResize);
}

async function boot() {
  setStatus("Loading MuJoCo WebAssembly...", false);
  sim.mujoco = await loadMujoco();
  const templateResp = await fetch("./assets/sidequest_template.xml");
  if (!templateResp.ok) {
    throw new Error(`Cannot load XML template (${templateResp.status})`);
  }
  sim.xmlTemplate = await templateResp.text();

  initThreeScene();
  await rebuildSimulation(Number(ui.massRange.value));
  animate();
}

function initThreeScene() {
  sim.scene = new THREE.Scene();
  sim.scene.background = new THREE.Color(0x111a2b);
  sim.scene.fog = new THREE.Fog(0x111a2b, 3.0, 16.0);

  sim.camera = new THREE.PerspectiveCamera(48, 1, 0.01, 80);
  sim.camera.position.set(2.5, -3.3, 1.6);

  sim.renderer = new THREE.WebGLRenderer({ canvas: ui.canvas, antialias: true });
  sim.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2.0));
  sim.renderer.shadowMap.enabled = true;

  sim.controls = new OrbitControls(sim.camera, ui.canvas);
  sim.controls.target.set(0.0, 0.0, 0.45);
  sim.controls.enableDamping = true;

  const keyLight = new THREE.DirectionalLight(0xffffff, 1.1);
  keyLight.position.set(2.2, -2.2, 4.0);
  keyLight.castShadow = true;
  sim.scene.add(keyLight);
  sim.scene.add(new THREE.AmbientLight(0x89a9cf, 0.45));

  const groundGeo = new THREE.PlaneGeometry(12, 12);
  const groundMat = new THREE.MeshStandardMaterial({ color: 0x33414f, roughness: 0.9, metalness: 0.05 });
  const ground = new THREE.Mesh(groundGeo, groundMat);
  ground.rotation.x = -Math.PI * 0.5;
  ground.receiveShadow = true;
  sim.scene.add(ground);

  const grid = new THREE.GridHelper(10, 40, 0x6ad6c9, 0x3a5067);
  grid.position.y = 0.001;
  sim.scene.add(grid);

  const baseGroup = new THREE.Group();
  const baseMesh = new THREE.Mesh(
    new THREE.BoxGeometry(0.32, 0.24, 0.1),
    new THREE.MeshStandardMaterial({ color: 0x2f8ded, roughness: 0.5, metalness: 0.25 }),
  );
  baseMesh.castShadow = true;
  baseGroup.add(baseMesh);

  const stickGroup = new THREE.Group();
  const stickMesh = new THREE.Mesh(
    new THREE.CylinderGeometry(0.045, 0.045, 0.5, 22),
    new THREE.MeshStandardMaterial({ color: 0xf2574d, roughness: 0.5, metalness: 0.12 }),
  );
  stickMesh.position.z = 0.25;
  stickMesh.rotation.x = Math.PI * 0.5;
  stickMesh.castShadow = true;
  stickGroup.add(stickMesh);

  const wheelGroup = new THREE.Group();
  const wheelMesh = new THREE.Mesh(
    new THREE.CylinderGeometry(0.2, 0.2, 0.06, 32),
    new THREE.MeshStandardMaterial({ color: 0xb99c68, roughness: 0.38, metalness: 0.48 }),
  );
  wheelMesh.rotation.z = Math.PI * 0.5;
  wheelMesh.castShadow = true;
  wheelGroup.add(wheelMesh);

  const payloadGroup = new THREE.Group();
  const payloadMesh = new THREE.Mesh(
    new THREE.BoxGeometry(0.12, 0.12, 0.06),
    new THREE.MeshStandardMaterial({ color: 0xffbe52, roughness: 0.42, metalness: 0.12 }),
  );
  payloadMesh.castShadow = true;
  payloadGroup.add(payloadMesh);

  sim.scene.add(baseGroup, stickGroup, wheelGroup, payloadGroup);
  sim.visuals = {
    base: baseGroup,
    stick: stickGroup,
    wheel: wheelGroup,
    payload: payloadGroup,
  };

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

async function rebuildSimulation(requestedMassKg) {
  if (!sim.mujoco || !sim.xmlTemplate) {
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

  const xmlText = sim.xmlTemplate.replace("__PAYLOAD_MASS__", sim.effectiveMassKg.toFixed(6));
  const modelPath = "/working/sidequest.xml";
  ensureFsDir("/working");
  try {
    sim.mujoco.FS.unlink(modelPath);
  } catch {}
  sim.mujoco.FS.writeFile(modelPath, xmlText);

  sim.model = sim.mujoco.MjModel.loadFromXML(modelPath);
  sim.data = new sim.mujoco.MjData(sim.model);
  sim.ids = resolveIds(sim.model);
  resetState();

  sim.failed = false;
  sim.failureReason = "";
  sim.comFailStreak = 0;
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

function resolveIds(model) {
  const mjtObj = sim.mujoco.mjtObj;
  const jidPitch = sim.mujoco.mj_name2id(model, mjtObj.mjOBJ_JOINT, "stick_pitch");
  const jidRoll = sim.mujoco.mj_name2id(model, mjtObj.mjOBJ_JOINT, "stick_roll");
  const jidRw = sim.mujoco.mj_name2id(model, mjtObj.mjOBJ_JOINT, "wheel_spin");
  const jidBaseX = sim.mujoco.mj_name2id(model, mjtObj.mjOBJ_JOINT, "base_x_slide");
  const jidBaseY = sim.mujoco.mj_name2id(model, mjtObj.mjOBJ_JOINT, "base_y_slide");

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
    aidRw: sim.mujoco.mj_name2id(model, mjtObj.mjOBJ_ACTUATOR, "wheel_spin"),
    aidBaseX: sim.mujoco.mj_name2id(model, mjtObj.mjOBJ_ACTUATOR, "base_x_force"),
    aidBaseY: sim.mujoco.mj_name2id(model, mjtObj.mjOBJ_ACTUATOR, "base_y_force"),
    bidBaseY: sim.mujoco.mj_name2id(model, mjtObj.mjOBJ_BODY, "base_y"),
    bidStick: sim.mujoco.mj_name2id(model, mjtObj.mjOBJ_BODY, "stick"),
    bidWheel: sim.mujoco.mj_name2id(model, mjtObj.mjOBJ_BODY, "wheel"),
    bidPayload: sim.mujoco.mj_name2id(model, mjtObj.mjOBJ_BODY, "payload"),
  };
}

function resetState() {
  sim.data.qpos.fill(0);
  sim.data.qvel.fill(0);
  sim.data.ctrl.fill(0);
  sim.data.qpos[sim.ids.qRoll] = 0.04;
  sim.mujoco.mj_forward(sim.model, sim.data);
}

function controllerStep() {
  const qpos = sim.data.qpos;
  const qvel = sim.data.qvel;
  const pitch = qpos[sim.ids.qPitch];
  const roll = qpos[sim.ids.qRoll];
  const pitchRate = qvel[sim.ids.vPitch];
  const rollRate = qvel[sim.ids.vRoll];
  const wheelRate = qvel[sim.ids.vRw];
  const baseX = qpos[sim.ids.qBaseX];
  const baseY = qpos[sim.ids.qBaseY];
  const baseXRate = qvel[sim.ids.vBaseX];
  const baseYRate = qvel[sim.ids.vBaseY];

  const uRw = clamp(
    -CTRL.rwKp * pitch - CTRL.rwKd * pitchRate - CTRL.rwWheelRateKd * wheelRate,
    -CTRL.rwMax,
    CTRL.rwMax,
  );
  const uBx = clamp(
    CTRL.bxKp * pitch +
      CTRL.bxKd * pitchRate -
      CTRL.baseDampK * baseXRate -
      CTRL.baseCenterK * baseX,
    -CTRL.baseMax,
    CTRL.baseMax,
  );
  const uBy = clamp(
    -CTRL.byKp * roll -
      CTRL.byKd * rollRate -
      CTRL.baseDampK * baseYRate -
      CTRL.baseCenterK * baseY,
    -CTRL.baseMax,
    CTRL.baseMax,
  );

  sim.data.ctrl[sim.ids.aidRw] = uRw;
  sim.data.ctrl[sim.ids.aidBaseX] = uBx;
  sim.data.ctrl[sim.ids.aidBaseY] = uBy;
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
  applyBodyPose(sim.visuals.base, sim.ids.bidBaseY);
  applyBodyPose(sim.visuals.stick, sim.ids.bidStick);
  applyBodyPose(sim.visuals.wheel, sim.ids.bidWheel);
  applyBodyPose(sim.visuals.payload, sim.ids.bidPayload);
}

function applyBodyPose(object3d, bodyId) {
  const base3 = 3 * bodyId;
  const base4 = 4 * bodyId;
  object3d.position.set(
    sim.data.xpos[base3],
    sim.data.xpos[base3 + 1],
    sim.data.xpos[base3 + 2],
  );
  object3d.quaternion.set(
    sim.data.xquat[base4 + 1],
    sim.data.xquat[base4 + 2],
    sim.data.xquat[base4 + 3],
    sim.data.xquat[base4],
  );
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
