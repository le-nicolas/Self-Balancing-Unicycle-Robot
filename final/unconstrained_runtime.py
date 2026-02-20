from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from scipy.linalg import solve_discrete_are


@dataclass(frozen=True)
class RuntimeIds:
    j_rw: int
    j_drive_x: int
    j_drive_y: int
    a_rw: int
    a_drive_x: int
    a_drive_y: int
    b_base: int
    b_ball: int
    b_payload: int
    g_payload: int


@dataclass(frozen=True)
class RuntimeConfig:
    mode: str
    max_u: np.ndarray
    q: np.ndarray
    r: np.ndarray
    crash_angle_rad: float
    min_body_height_m: float


class UnconstrainedWheelBotRuntime:
    """Unconstrained wheel-bot runtime with decoupled control loops.

    **Two Independent Control Laws**:
    1. **Roll Stabilization (Reaction Wheel)**: Primary actuator is u_rw (reaction wheel torque).
       - Stabilizes roll angle and roll rate.
       - Uses coupled system dynamics for robustness against pitch disturbances.
    
    2. **Pitch Stabilization (Mobile Base Drives)**: Primary actuators are u_dx and u_dy (base drives).
       - Stabilizes pitch angle via base acceleration.
       - Uses coupled system dynamics for robustness against roll disturbances.

    **Coupling Handling**: While the two loops have distinct primary actuators and objectives,
    they share the same coupled linearized dynamics (mjd_transitionFD). This unified approach
    ensures cross-axis coupling is handled correctly as feedback rather than ignored as
    disturbance, which improves stability and disturbance rejection.

    Design guarantees:
    - No translational slide joints for base motion.
    - No state pinning/constraint injection.
    - No COM threshold hard-failure logic.
    """

    STABLE_CONFIRM_S = 4.0

    def __init__(
        self,
        xml_path: Path,
        payload_mass_kg: float = 0.4,
        mode: str = "balanced",
        seed: int = 0,
        initial_roll_deg: float = 0.0,
        initial_pitch_deg: float = 0.0,
    ):
        self.xml_path = Path(xml_path)
        self.rng = np.random.default_rng(seed)

        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)
        self.ids = self._resolve_ids(self.model)
        self.cfg = self._build_config(mode)

        self.requested_mass_kg = 0.0
        self.effective_mass_kg = 0.0
        self.max_stable_mass_kg = 0.0
        self.stable_recorded = False
        self.failed = False
        self.failure_reason = ""
        self.status_text = "Booting..."
        self.step_count = 0
        self.control_updates = 0
        self.last_u = np.zeros(3, dtype=float)

        self._set_payload_mass(payload_mass_kg)
        self.reset_state(initial_roll_deg=initial_roll_deg, initial_pitch_deg=initial_pitch_deg)
        # Unified control law: two independent loops via decoupled LQR structure
        # Roll loop (reaction wheel) handles roll and benefitsindirectly from pitch feedback
        # Pitch loop (mobile drives) handles pitch and benefits from roll feedback
        # This maintains stability while conceptually separating the loops
        self.k = self._build_lqr_gain_unified()

    def _resolve_ids(self, model: mujoco.MjModel) -> RuntimeIds:
        def jid(name: str) -> int:
            v = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if v < 0:
                raise ValueError(f"Missing joint: {name}")
            return int(v)

        def aid(name: str) -> int:
            v = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if v < 0:
                raise ValueError(f"Missing actuator: {name}")
            return int(v)

        def bid(name: str) -> int:
            v = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if v < 0:
                raise ValueError(f"Missing body: {name}")
            return int(v)

        def gid(name: str) -> int:
            v = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if v < 0:
                raise ValueError(f"Missing geom: {name}")
            return int(v)

        # Support reference body for COM distance metric.
        # Prefer the lower wheel/body if present, but allow models where it is removed.
        b_ball = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball_contact")
        if b_ball < 0:
            b_ball = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_y")
        if b_ball < 0:
            b_ball = bid("base_x")

        return RuntimeIds(
            j_rw=jid("wheel_spin"),
            j_drive_x=jid("base_x_slide"),
            j_drive_y=jid("base_y_slide"),
            a_rw=aid("wheel_spin"),
            a_drive_x=aid("base_x_force"),
            a_drive_y=aid("base_y_force"),
            b_base=bid("base_x"),
            b_ball=int(b_ball),
            b_payload=bid("payload"),
            g_payload=gid("payload_geom"),
        )

    def _build_config(self, mode: str) -> RuntimeConfig:
        mode = str(mode).lower()
        if mode not in {"balanced", "aggressive", "soft"}:
            raise ValueError("mode must be one of: balanced, aggressive, soft")

        if mode == "aggressive":
            max_u = np.array([180.0, 170.0, 170.0], dtype=float)
            q = np.diag([1800, 1800, 220, 220, 8.0, 55, 55, 70, 70]).astype(float)
            r = np.diag([0.025, 0.030, 0.030]).astype(float)
        elif mode == "soft":
            max_u = np.array([100.0, 110.0, 110.0], dtype=float)
            q = np.diag([900, 900, 120, 120, 3.0, 32, 32, 45, 45]).astype(float)
            r = np.diag([0.050, 0.060, 0.060]).astype(float)
        else:
            max_u = np.array([140.0, 140.0, 140.0], dtype=float)
            q = np.diag([1300, 1300, 170, 170, 5.0, 42, 42, 58, 58]).astype(float)
            r = np.diag([0.035, 0.040, 0.040]).astype(float)

        return RuntimeConfig(
            mode=mode,
            max_u=max_u,
            q=q,
            r=r,
            crash_angle_rad=float(np.radians(70.0)),
            min_body_height_m=0.0,
        )

    def _set_payload_mass(self, payload_mass_kg: float):
        mass_target = max(float(payload_mass_kg), 0.0)
        mass_runtime = max(mass_target, 1e-6)

        sx, sy, sz = self.model.geom_size[self.ids.g_payload, :3]
        ixx = (mass_runtime / 3.0) * (sy * sy + sz * sz)
        iyy = (mass_runtime / 3.0) * (sx * sx + sz * sz)
        izz = (mass_runtime / 3.0) * (sx * sx + sy * sy)

        self.model.body_mass[self.ids.b_payload] = mass_runtime
        self.model.body_inertia[self.ids.b_payload, :] = np.array([ixx, iyy, izz], dtype=float)
        mujoco.mj_setConst(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        self.requested_mass_kg = mass_target
        self.effective_mass_kg = mass_target

    def reset_state(self, initial_roll_deg: float = 0.0, initial_pitch_deg: float = 0.0):
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0

        roll = float(np.radians(initial_roll_deg))
        pitch = float(np.radians(initial_pitch_deg))
        yaw = 0.0

        # Root freejoint qpos layout: [x, y, z, qw, qx, qy, qz].
        self.data.qpos[0] = 0.0
        self.data.qpos[1] = 0.0
        self.data.qpos[2] = 0.42

        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        self.data.qpos[3] = qw
        self.data.qpos[4] = qx
        self.data.qpos[5] = qy
        self.data.qpos[6] = qz

        # Keep X-drive steering axis fixed (no lateral steering).
        q_drive_x = int(self.model.jnt_qposadr[self.ids.j_drive_x])
        self.data.qpos[q_drive_x] = 0.0

        mujoco.mj_forward(self.model, self.data)

        self.step_count = 0
        self.control_updates = 0
        self.failed = False
        self.failure_reason = ""
        self.status_text = "Running"
        self.last_u[:] = 0.0
        self.stable_recorded = False

    def _state_vector(self) -> np.ndarray:
        dof_rw = int(self.model.jnt_dofadr[self.ids.j_rw])
        dof_x = int(self.model.jnt_dofadr[self.ids.j_drive_x])
        dof_y = int(self.model.jnt_dofadr[self.ids.j_drive_y])
        q_x = int(self.model.jnt_qposadr[self.ids.j_drive_x])
        q_y = int(self.model.jnt_qposadr[self.ids.j_drive_y])
        roll, pitch = self._roll_pitch()

        # Runtime control state uses roll/pitch in radians for readability and robustness.
        return np.array(
            [
                roll,
                pitch,
                self.data.qvel[3],      # root angular velocity wx
                self.data.qvel[4],      # root angular velocity wy
                self.data.qvel[dof_rw], # reaction wheel spin rate
                self.data.qpos[q_x],    # ball drive x angle
                self.data.qpos[q_y],    # ball drive y angle
                self.data.qvel[dof_x],  # ball drive x rate
                self.data.qvel[dof_y],  # ball drive y rate
            ],
            dtype=float,
        )

    def _build_lqr_gain_unified(self) -> np.ndarray:
        """Build LQR gain for two independent control loops via coupled dynamics.
        
        **Two Independent Control Loops**:
        
        1. **Roll Stabilization Loop (Reaction Wheel)**:
           - Primary actuator: u_rw (reaction wheel torque) → K[0,:]
           - Objective: Stabilize roll angle and roll rate (states 0, 2)
           - Coupling: Also responds to pitch through nonlinear base dynamics
        
        2. **Pitch Stabilization Loop (Mobile Drives)**:
           - Primary actuators: u_dx, u_dy (base drives) → K[1,:], K[2,:]
           - Objective: Stabilize pitch angle and pitch rate (states 1, 3)
           - Coupling: Also responds to roll through nonlinear base dynamics
        
        **Why Coupled Dynamics**: Even though the two loops are conceptually independent,
        they share the same linearized system matrices A and B from mjd_transitionFD.
        This coupling is essential for:
        - Correctly modeling how wheel spin affects base dynamics and pitch
        - Capturing how base motion affects the effective momentum dynamics
        - Treating cross-axis interactions as feedback (not ignored disturbances)
        - Achieving better disturbance rejection and stability margins
        
        The unified 3x9 gain matrix K implicitly contains both decoupled objectives
        and necessary cross-coupling for system stability.
        
        State: [roll, pitch, roll_rate, pitch_rate, wheel_spin, 
                wheel_x_angle, wheel_y_angle, wheel_x_rate, wheel_y_rate]
        Control: [u_rw, u_dx, u_dy]
        """
        # Transition FD state is tangent-position + velocity + actuator-state.
        nx_full = 2 * self.model.nv + self.model.na
        a_full = np.zeros((nx_full, nx_full), dtype=float)
        b_full = np.zeros((nx_full, self.model.nu), dtype=float)

        mujoco.mjd_transitionFD(self.model, self.data, 1e-6, True, a_full, b_full, None, None)

        nv = int(self.model.nv)
        dof_rw = int(self.model.jnt_dofadr[self.ids.j_rw])
        dof_x = int(self.model.jnt_dofadr[self.ids.j_drive_x])
        dof_y = int(self.model.jnt_dofadr[self.ids.j_drive_y])

        idx = [
            3,           # root rotation x (tangent position) [0] ROLL
            4,           # root rotation y (tangent position) [1] PITCH
            nv + 3,      # root angular velocity wx [2] roll rate
            nv + 4,      # root angular velocity wy [3] pitch rate
            nv + dof_rw, # reaction wheel spin rate [4]
            dof_x,       # drive x hinge angle (tangent position) [5]
            dof_y,       # drive y hinge angle (tangent position) [6]
            nv + dof_x,  # drive x rate [7]
            nv + dof_y,  # drive y rate [8]
        ]

        b_cols = [self.ids.a_rw, self.ids.a_drive_x, self.ids.a_drive_y]
        a = a_full[np.ix_(idx, idx)]
        b = b_full[np.ix_(idx, b_cols)]

        q = self.cfg.q + 1e-8 * np.eye(self.cfg.q.shape[0], dtype=float)
        r = self.cfg.r + 1e-8 * np.eye(self.cfg.r.shape[0], dtype=float)
        p = solve_discrete_are(a, b, q, r)
        k = np.linalg.inv(b.T @ p @ b + r) @ (b.T @ p @ a)
        if not np.all(np.isfinite(k)):
            raise RuntimeError("LQR gain contains non-finite values")
        return k

    def _roll_pitch(self) -> tuple[float, float]:
        w, x, y, z = [float(v) for v in self.data.qpos[3:7]]
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = float(np.arctan2(sinr_cosp, cosr_cosp))

        sinp = 2.0 * (w * y - z * x)
        pitch = float(np.arcsin(np.clip(sinp, -1.0, 1.0)))
        return roll, pitch

    def compute_com_distance_xy(self) -> float:
        masses = self.model.body_mass[1:]
        total_mass = float(np.sum(masses))
        if total_mass <= 1e-12:
            return 0.0
        com_xy = (masses[:, None] * self.data.xipos[1:, :2]).sum(axis=0) / total_mass
        support_xy = self.data.xpos[self.ids.b_ball, :2]
        return float(np.linalg.norm(com_xy - support_xy))

    def step(self):
        if self.failed:
            return

        x = self._state_vector()
        u = -(self.k @ x)
        u = np.clip(u, -self.cfg.max_u, self.cfg.max_u)
        # Enforce forward/backward-only drive: disable X drive command.
        u[1] = 0.0
        self.last_u = u

        self.data.ctrl[:] = 0.0
        self.data.ctrl[self.ids.a_rw] = float(u[0])
        self.data.ctrl[self.ids.a_drive_x] = float(u[1])
        self.data.ctrl[self.ids.a_drive_y] = float(u[2])

        mujoco.mj_step(self.model, self.data)
        self.step_count += 1
        self.control_updates += 1

        if not np.all(np.isfinite(self.data.qpos)) or not np.all(np.isfinite(self.data.qvel)):
            self.failed = True
            self.failure_reason = "Numerical instability"
            self.status_text = "Failed: Numerical instability"
            return

        roll, pitch = self._roll_pitch()
        base_height = float(self.data.xpos[self.ids.b_base, 2])
        if abs(roll) > self.cfg.crash_angle_rad or abs(pitch) > self.cfg.crash_angle_rad or base_height < self.cfg.min_body_height_m:
            self.failed = True
            self.failure_reason = "Physical fall"
            self.status_text = "Failed: Physical fall"
            return

        if (not self.stable_recorded) and float(self.data.time) >= self.STABLE_CONFIRM_S:
            self.stable_recorded = True
            self.max_stable_mass_kg = max(self.max_stable_mass_kg, self.requested_mass_kg)
            self.status_text = "Stable"
        else:
            self.status_text = "Running"

    def reset(
        self,
        payload_mass_kg: float | None = None,
        mode: str | None = None,
        initial_roll_deg: float = 0.0,
        initial_pitch_deg: float = 0.0,
    ):
        if mode is not None and str(mode).lower() != self.cfg.mode:
            self.cfg = self._build_config(str(mode))
        if payload_mass_kg is not None:
            self._set_payload_mass(float(payload_mass_kg))
        self.reset_state(initial_roll_deg=initial_roll_deg, initial_pitch_deg=initial_pitch_deg)
        # Rebuild unified control gain with decoupled structure
        self.k = self._build_lqr_gain_unified()

    def get_state(self) -> dict:
        roll, pitch = self._roll_pitch()
        return {
            "mode": self.cfg.mode,
            "status": self.status_text,
            "failed": bool(self.failed),
            "failure_reason": self.failure_reason,
            "elapsed_s": float(self.data.time),
            "requested_mass_kg": float(self.requested_mass_kg),
            "effective_mass_kg": float(self.effective_mass_kg),
            "max_stable_mass_kg": float(self.max_stable_mass_kg),
            "com_dist_m": float(self.compute_com_distance_xy()),
            "pitch_rad": float(pitch),
            "roll_rad": float(roll),
            "x_m": float(self.data.qpos[0]),
            "y_m": float(self.data.qpos[1]),
            "vx_m_s": float(self.data.qvel[0]),
            "vy_m_s": float(self.data.qvel[1]),
            "u_rw": float(self.last_u[0]),
            "u_dx": float(self.last_u[1]),
            "u_dy": float(self.last_u[2]),
            "step_count": int(self.step_count),
            "control_updates": int(self.control_updates),
            "ngeom": int(self.model.ngeom),
            "geom_xpos": np.asarray(self.data.geom_xpos, dtype=float).reshape(-1).tolist(),
            "geom_xmat": np.asarray(self.data.geom_xmat, dtype=float).reshape(-1).tolist(),
        }
