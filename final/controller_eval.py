import math
import csv
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import mujoco
import numpy as np
from scipy.linalg import LinAlgError, solve_discrete_are


@dataclass
class CandidateParams:
    r_du_rw: float
    r_du_bx: float
    r_du_by: float
    q_ang_scale: float
    q_rate_scale: float
    q_rw_scale: float
    q_base_scale: float
    q_vel_scale: float
    qu_scale: float
    ki_base: float
    u_bleed: float
    max_du_rw: float
    max_du_bx: float
    max_du_by: float


@dataclass
class EpisodeConfig:
    steps: int = 3000
    disturbance_magnitude_xy: float = 4.0
    disturbance_magnitude_z: float = 2.0
    disturbance_interval: int = 300
    init_angle_deg: float = 4.0
    init_base_pos_m: float = 0.15
    max_worst_tilt_deg: float = 20.0
    max_worst_base_m: float = 4.0
    max_mean_sat_rate_du: float = 0.98
    max_mean_sat_rate_abs: float = 0.90
    control_hz: float = 250.0
    control_delay_steps: int = 1
    hardware_realistic: bool = True
    imu_angle_noise_std_rad: float = 0.00436
    imu_rate_noise_std_rad_s: float = 0.02
    wheel_encoder_ticks_per_rev: int = 2048
    wheel_encoder_rate_noise_std_rad_s: float = 0.01
    preset: str = "default"
    stability_profile: str = "default"
    controller_family: str = "current"
    model_variant_id: str = "nominal"
    domain_profile_id: str = "default"
    hardware_replay: bool = False
    hardware_trace_path: str | None = None


class ControllerEvaluator:
    PITCH_EQ = 0.0
    ROLL_EQ = 0.0
    X_REF = 0.0
    Y_REF = 0.0
    INT_CLAMP = 2.0
    UPRIGHT_ANGLE_THRESH = np.radians(3.0)
    UPRIGHT_VEL_THRESH = 0.10
    UPRIGHT_POS_THRESH = 0.30
    CRASH_ANGLE_RAD = np.pi * 0.5

    MAX_U = np.array([80.0, 10.0, 10.0])
    QX = np.diag([95.0, 75.0, 40.0, 28.0, 0.8, 90.0, 90.0, 170.0, 170.0])
    QU = np.diag([5e-3, 0.35, 0.35])
    QN = np.diag([1e-4, 1e-4, 1e-4, 1e-4, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5])
    MAX_WHEEL_SPEED_RAD_S = 2035.75
    MAX_BASE_SPEED_M_S = 2.5
    WHEEL_TORQUE_DERATE_START = 0.75
    BASE_TORQUE_DERATE_START = 0.70
    BASE_FORCE_SOFT_LIMIT = 10.0
    BASE_DAMPING_GAIN = 2.4
    BASE_CENTERING_GAIN = 0.9
    BASE_TILT_DEADBAND_RAD = np.radians(0.4)
    BASE_TILT_FULL_AUTHORITY_RAD = np.radians(1.8)
    BASE_COMMAND_GAIN = 0.70
    BASE_CENTERING_POS_CLIP_M = 0.25
    BASE_SPEED_SOFT_LIMIT_FRAC = 0.55
    BASE_HOLD_RADIUS_M = 0.22
    BASE_REF_FOLLOW_RATE_HZ = 5.5
    BASE_REF_RECENTER_RATE_HZ = 0.90
    BASE_PITCH_KP = 84.0
    BASE_PITCH_KD = 18.0
    BASE_ROLL_KP = 34.0
    BASE_ROLL_KD = 8.0
    BASE_AUTHORITY_RATE_PER_S = 1.8
    BASE_COMMAND_LPF_HZ = 8.0
    UPRIGHT_BASE_DU_SCALE = 0.45
    WHEEL_MOMENTUM_THRESH_FRAC = 0.18
    WHEEL_MOMENTUM_K = 0.90
    WHEEL_MOMENTUM_UPRIGHT_K = 0.55
    HOLD_ENTER_ANGLE_RAD = np.radians(1.4)
    HOLD_EXIT_ANGLE_RAD = np.radians(2.4)
    HOLD_ENTER_RATE_RAD_S = 0.55
    HOLD_EXIT_RATE_RAD_S = 0.95
    WHEEL_SPIN_BUDGET_FRAC = 0.12
    WHEEL_SPIN_HARD_FRAC = 0.25
    WHEEL_SPIN_BUDGET_ABS_RAD_S = 160.0
    WHEEL_SPIN_HARD_ABS_RAD_S = 250.0
    HIGH_SPIN_EXIT_FRAC = 0.78
    HIGH_SPIN_COUNTER_MIN_FRAC = 0.20
    HIGH_SPIN_BASE_AUTHORITY_MIN = 0.60
    WHEEL_TO_BASE_BIAS_GAIN = 0.70
    RECOVERY_WHEEL_DESPIN_SCALE = 0.45
    HOLD_WHEEL_DESPIN_SCALE = 1.0
    UPRIGHT_BLEND_RISE = 0.10
    UPRIGHT_BLEND_FALL = 0.28
    # C maps full state to measured channels [pitch, roll, pitch_rate, roll_rate, wheel_rate].
    C_MEAS = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    @staticmethod
    def _fuzzy_roll_gain(roll: float, roll_rate: float, angle_scale: float, rate_scale: float) -> float:
        roll_n = abs(roll) / max(angle_scale, 1e-6)
        rate_n = abs(roll_rate) / max(rate_scale, 1e-6)
        level = float(np.clip(0.65 * roll_n + 0.35 * rate_n, 0.0, 1.0))
        return 0.35 + 0.95 * level

    def _episode_composite_score(self, ep: Dict[str, float], steps: int) -> float:
        survived = float(ep["survived"])
        max_tilt = max(float(ep["max_abs_pitch_deg"]), float(ep["max_abs_roll_deg"]))
        max_base = max(float(ep["max_abs_base_x_m"]), float(ep["max_abs_base_y_m"]))
        energy = float(ep["control_energy"])
        jerk = float(ep["mean_command_jerk"])
        sat_du = float(ep["sat_rate_du"])
        sat_abs = float(ep["sat_rate_abs"])
        wheel_hard_ratio = float(ep["wheel_over_hard"]) / max(float(steps), 1.0)
        wheel_budget_ratio = float(ep["wheel_over_budget"]) / max(float(steps), 1.0)
        return (
            100.0 * survived
            - 1.8 * max_tilt
            - 3.0 * max_base
            - 0.06 * energy
            - 0.2 * jerk
            - 20.0 * sat_du
            - 20.0 * sat_abs
            - 12.0 * wheel_hard_ratio
            - 3.0 * wheel_budget_ratio
        )

    @staticmethod
    def _variant_scales(model_variant_id: str) -> tuple[float, float, float]:
        vid = str(model_variant_id)
        # (A_scale, B_scale, disturbance_scale)
        return {
            "nominal": (1.0, 1.0, 1.0),
            "inertia_plus": (1.04, 0.90, 1.0),
            "inertia_minus": (0.96, 1.10, 1.0),
            "friction_low": (1.02, 0.95, 1.15),
            "friction_high": (0.98, 1.05, 0.85),
            "com_shift": (1.03, 0.93, 1.10),
        }.get(vid, (1.0, 1.0, 1.0))

    @staticmethod
    def _domain_noise_scales(domain_profile_id: str) -> tuple[float, float]:
        did = str(domain_profile_id)
        # (sensor_noise_scale, timing_jitter_frac)
        return {
            "default": (1.0, 0.0),
            "rand_light": (1.15, 0.05),
            "rand_medium": (1.35, 0.10),
            "rand_heavy": (1.65, 0.15),
        }.get(did, (1.0, 0.0))

    @staticmethod
    def _solve_discrete_are_robust(
        A: np.ndarray,
        B: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        label: str,
    ) -> np.ndarray:
        reg_steps = (0.0, 1e-12, 1e-10, 1e-8, 1e-6)
        eye = np.eye(R.shape[0], dtype=float)
        last_exc: Exception | None = None
        for eps in reg_steps:
            try:
                P = solve_discrete_are(A, B, Q, R + eps * eye)
                if np.all(np.isfinite(P)):
                    return 0.5 * (P + P.T)
            except Exception as exc:
                last_exc = exc
        raise RuntimeError(f"{label} DARE failed after regularization fallback: {last_exc}") from last_exc

    @staticmethod
    def _solve_linear_robust(gram: np.ndarray, rhs: np.ndarray, label: str) -> np.ndarray:
        try:
            return np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            sol, *_ = np.linalg.lstsq(gram, rhs, rcond=None)
            if not np.all(np.isfinite(sol)):
                raise RuntimeError(f"{label} linear solve returned non-finite result.")
            return sol

    def __init__(self, xml_path: Path):
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep

        self._resolve_ids()
        self.A, self.B = self._linearize()
        self.NX = self.A.shape[0]
        self.NU = self.B.shape[1]
        self._validate_xml_ctrlrange()

    def _resolve_ids(self):
        def jid(name: str) -> int:
            return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)

        def aid(name: str) -> int:
            return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

        self.jid_pitch = jid("stick_pitch")
        self.jid_roll = jid("stick_roll")
        self.jid_rw = jid("wheel_spin")
        self.jid_base_x = jid("base_x_slide")
        self.jid_base_y = jid("base_y_slide")

        self.q_pitch = self.model.jnt_qposadr[self.jid_pitch]
        self.q_roll = self.model.jnt_qposadr[self.jid_roll]
        self.q_base_x = self.model.jnt_qposadr[self.jid_base_x]
        self.q_base_y = self.model.jnt_qposadr[self.jid_base_y]

        self.v_pitch = self.model.jnt_dofadr[self.jid_pitch]
        self.v_roll = self.model.jnt_dofadr[self.jid_roll]
        self.v_rw = self.model.jnt_dofadr[self.jid_rw]
        self.v_base_x = self.model.jnt_dofadr[self.jid_base_x]
        self.v_base_y = self.model.jnt_dofadr[self.jid_base_y]

        self.aid_rw = aid("wheel_spin")
        self.aid_base_x = aid("base_x_force")
        self.aid_base_y = aid("base_y_force")
        self.stick_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "stick")

    def _linearize(self):
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        self.data.qpos[self.q_pitch] = self.PITCH_EQ
        self.data.qpos[self.q_roll] = self.ROLL_EQ
        mujoco.mj_forward(self.model, self.data)

        nx = self.model.nq + self.model.nv
        nu = self.model.nu
        A_full = np.zeros((nx, nx))
        B_full = np.zeros((nx, nu))
        mujoco.mjd_transitionFD(self.model, self.data, 1e-6, True, A_full, B_full, None, None)

        idx = [
            self.q_pitch,
            self.q_roll,
            self.model.nq + self.v_pitch,
            self.model.nq + self.v_roll,
            self.model.nq + self.v_rw,
            self.q_base_x,
            self.q_base_y,
            self.model.nq + self.v_base_x,
            self.model.nq + self.v_base_y,
        ]
        A = A_full[np.ix_(idx, idx)]
        B = B_full[np.ix_(idx, [self.aid_rw, self.aid_base_x, self.aid_base_y])]
        return A, B

    def _build_measurement_noise_cov(self, config: EpisodeConfig, wheel_lsb_rad_s: float) -> np.ndarray:
        angle_var = float(config.imu_angle_noise_std_rad**2)
        rate_var = float(config.imu_rate_noise_std_rad_s**2)
        # Uniform quantization noise variance + additive rate-noise variance.
        wheel_quant_var = float((wheel_lsb_rad_s**2) / 12.0)
        wheel_noise_var = float(config.wheel_encoder_rate_noise_std_rad_s**2)
        return np.diag([angle_var, angle_var, rate_var, rate_var, wheel_quant_var + wheel_noise_var])

    def _build_kalman_gain(self, config: EpisodeConfig, wheel_lsb_rad_s: float):
        R_meas = self._build_measurement_noise_cov(config, wheel_lsb_rad_s)
        Pk = self._solve_discrete_are_robust(self.A.T, self.C_MEAS.T, self.QN, R_meas, label="Kalman")
        gram = self.C_MEAS @ Pk @ self.C_MEAS.T + R_meas
        rhs = self.C_MEAS @ Pk.T
        return self._solve_linear_robust(gram, rhs, label="Kalman gain").T

    def _build_lqr_gain(self, params: CandidateParams):
        A_aug = np.block([[self.A, self.B], [np.zeros((3, 9)), np.eye(3)]])
        B_aug = np.vstack([self.B, np.eye(3)])

        qx = self.QX.copy()
        qx[0, 0] *= params.q_ang_scale
        qx[1, 1] *= params.q_ang_scale
        qx[2, 2] *= params.q_rate_scale
        qx[3, 3] *= params.q_rate_scale
        qx[4, 4] *= params.q_rw_scale
        qx[5, 5] *= params.q_base_scale
        qx[6, 6] *= params.q_base_scale
        qx[7, 7] *= params.q_vel_scale
        qx[8, 8] *= params.q_vel_scale
        qu = self.QU * params.qu_scale

        q_aug = np.block(
            [
                [qx, np.zeros((9, 3))],
                [np.zeros((3, 9)), qu],
            ]
        )
        r_du = np.diag([params.r_du_rw, params.r_du_bx, params.r_du_by])
        p_aug = self._solve_discrete_are_robust(A_aug, B_aug, q_aug, r_du, label="Controller")
        gram = B_aug.T @ p_aug @ B_aug + r_du
        rhs = B_aug.T @ p_aug @ A_aug
        return self._solve_linear_robust(gram, rhs, label="Controller gain")

    def _validate_xml_ctrlrange(self):
        act_ids = np.array([self.aid_rw, self.aid_base_x, self.aid_base_y], dtype=int)
        lo = self.model.actuator_ctrlrange[act_ids, 0]
        hi = self.model.actuator_ctrlrange[act_ids, 1]
        bad = (lo > -self.MAX_U) | (hi < self.MAX_U)
        if np.any(bad):
            raise ValueError("XML actuator ctrlrange is tighter than MAX_U.")

    def _reset_with_initial_state(self, rng: np.random.Generator, config: EpisodeConfig):
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0

        ang = math.radians(config.init_angle_deg)
        self.data.qpos[self.q_pitch] = rng.uniform(-ang, ang)
        self.data.qpos[self.q_roll] = rng.uniform(-ang, ang)
        self.data.qpos[self.q_base_x] = rng.uniform(-config.init_base_pos_m, config.init_base_pos_m)
        self.data.qpos[self.q_base_y] = rng.uniform(-config.init_base_pos_m, config.init_base_pos_m)

        mujoco.mj_forward(self.model, self.data)

    def _control_period_steps(self, config: EpisodeConfig) -> int:
        if not config.hardware_realistic:
            return 1
        denom = max(config.control_hz, 1e-9)
        return max(1, int(round(1.0 / (self.dt * denom))))

    def _wheel_rate_lsb(self, config: EpisodeConfig, control_dt: float) -> float:
        ticks = max(int(config.wheel_encoder_ticks_per_rev), 1)
        return (2.0 * np.pi) / (ticks * control_dt)

    def _build_measurement(
        self,
        x_true: np.ndarray,
        config: EpisodeConfig,
        wheel_lsb: float,
        rng: np.random.Generator,
        noise_scale: float = 1.0,
    ):
        pitch = x_true[0] + rng.normal(0.0, noise_scale * config.imu_angle_noise_std_rad)
        roll = x_true[1] + rng.normal(0.0, noise_scale * config.imu_angle_noise_std_rad)
        pitch_rate = x_true[2] + rng.normal(0.0, noise_scale * config.imu_rate_noise_std_rad_s)
        roll_rate = x_true[3] + rng.normal(0.0, noise_scale * config.imu_rate_noise_std_rad_s)
        wheel_quant = np.round(x_true[4] / wheel_lsb) * wheel_lsb
        wheel_rate = wheel_quant + rng.normal(0.0, noise_scale * config.wheel_encoder_rate_noise_std_rad_s)
        return np.array([pitch, roll, pitch_rate, roll_rate, wheel_rate], dtype=float)

    def simulate_episode(
        self,
        params: CandidateParams,
        episode_seed: int,
        config: EpisodeConfig,
    ) -> Dict[str, float]:
        rng = np.random.default_rng(episode_seed)
        self._reset_with_initial_state(rng, config)
        a_scale, b_scale, disturbance_scale = self._variant_scales(config.model_variant_id)
        noise_scale, timing_jitter_frac = self._domain_noise_scales(config.domain_profile_id)
        A_run = self.A * a_scale
        B_run = self.B * b_scale

        control_steps = self._control_period_steps(config)
        control_dt = control_steps * self.dt
        wheel_lsb = self._wheel_rate_lsb(config, control_dt)
        cfg_meas = EpisodeConfig(**{**config.__dict__})
        cfg_meas.imu_angle_noise_std_rad *= noise_scale
        cfg_meas.imu_rate_noise_std_rad_s *= noise_scale
        cfg_meas.wheel_encoder_rate_noise_std_rad_s *= noise_scale
        L = self._build_kalman_gain(cfg_meas, wheel_lsb)

        queue_len = 1 if not config.hardware_realistic else max(int(config.control_delay_steps), 0) + 1
        cmd_queue = deque([np.zeros(3, dtype=float) for _ in range(queue_len)], maxlen=queue_len)

        x_est = np.zeros(9, dtype=float)
        u_applied = np.zeros(3, dtype=float)
        u_eff_applied = np.zeros(3, dtype=float)
        base_int = np.zeros(2, dtype=float)
        base_ref = np.zeros(2, dtype=float)
        base_authority_state = 0.0
        u_base_smooth = np.zeros(2, dtype=float)
        balance_phase = "recovery"
        recovery_time_s = 0.0
        high_spin_active = False
        upright_blend = 0.0
        despin_gain = 0.25
        max_pitch = 0.0
        max_roll = 0.0
        max_base_x = 0.0
        max_base_y = 0.0
        drift_sq_acc = 0.0
        control_energy = 0.0
        command_jerk_acc = 0.0
        motion_activity_acc = 0.0
        sign_flip_count = 0
        pitch_rate_series = []
        roll_rate_series = []
        vx_series = []
        vy_series = []
        sat_hits = 0
        du_hits = 0
        control_updates = 0
        crash_step: Optional[int] = None
        phase_switch_count = 0
        hold_steps = 0
        wheel_over_budget_count = 0
        wheel_over_hard_count = 0
        high_spin_steps = 0

        family = str(getattr(config, "controller_family", "current"))
        low_spin_robust = (config.stability_profile == "low-spin-robust") and family in ("current", "hybrid_modern")
        legacy_family = family in ("legacy_wheel_pid", "legacy_wheel_lqr", "legacy_run_pd")
        paper_family = family == "paper_split_baseline"
        hybrid_family = family == "hybrid_modern"
        mpc_family = family == "baseline_mpc"
        robust_hinf_family = family == "baseline_robust_hinf_like"

        k_du = self._build_lqr_gain(params)
        a_w = A_run[np.ix_([0, 2, 4], [0, 2, 4])]
        b_w = B_run[np.ix_([0, 2, 4], [0])]
        q_w = np.diag([260.0, 35.0, 0.6])
        r_w = np.array([[0.08]])
        p_w = self._solve_discrete_are_robust(a_w, b_w, q_w, r_w, label="Paper pitch")
        k_paper_pitch = self._solve_linear_robust(b_w.T @ p_w @ b_w + r_w, b_w.T @ p_w @ a_w, label="Paper pitch K")
        q_mpc = np.diag([120.0, 85.0, 50.0, 40.0, 1.0, 100.0, 100.0, 200.0, 200.0])
        r_mpc = np.diag([0.35, 0.40, 0.40])
        gram_mpc = B_run.T @ q_mpc @ B_run + r_mpc
        rhs_mpc = B_run.T @ q_mpc @ A_run
        k_mpc = self._solve_linear_robust(gram_mpc, rhs_mpc, label="MPC one-step K")
        max_du = np.array([params.max_du_rw, params.max_du_bx, params.max_du_by], dtype=float)
        tilt_fail_rad = min(math.radians(config.max_worst_tilt_deg), self.CRASH_ANGLE_RAD)

        for step in range(1, config.steps + 1):
            x_true = np.array(
                [
                    self.data.qpos[self.q_pitch] - self.PITCH_EQ,
                    self.data.qpos[self.q_roll] - self.ROLL_EQ,
                    self.data.qvel[self.v_pitch],
                    self.data.qvel[self.v_roll],
                    self.data.qvel[self.v_rw],
                    self.data.qpos[self.q_base_x],
                    self.data.qpos[self.q_base_y],
                    self.data.qvel[self.v_base_x],
                    self.data.qvel[self.v_base_y],
                ],
                dtype=float,
            )

            # Continuous predictor at simulation rate.
            x_pred = A_run @ x_est + B_run @ u_eff_applied
            x_est = x_pred

            effective_control_steps = control_steps
            if timing_jitter_frac > 0.0:
                delta = 1 if rng.random() < timing_jitter_frac else 0
                if rng.random() < 0.5:
                    effective_control_steps = max(1, control_steps - delta)
                else:
                    effective_control_steps = control_steps + delta

            if step % effective_control_steps == 0:
                control_updates += 1
                y = self._build_measurement(x_true, config, wheel_lsb, rng, noise_scale=noise_scale)
                x_est = x_pred + L @ (y - self.C_MEAS @ x_pred)
                # Base states are unobserved by C_MEAS; anchor to avoid estimator drift.
                x_est[5] = x_true[5]
                x_est[6] = x_true[6]
                x_est[7] = x_true[7]
                x_est[8] = x_true[8]

                x_ctrl = x_est.copy()
                x_ctrl[5] -= self.X_REF
                x_ctrl[6] -= self.Y_REF

                base_int[0] = np.clip(base_int[0] + x_ctrl[5] * control_dt, -self.INT_CLAMP, self.INT_CLAMP)
                base_int[1] = np.clip(base_int[1] + x_ctrl[6] * control_dt, -self.INT_CLAMP, self.INT_CLAMP)

                if low_spin_robust:
                    angle_mag = max(abs(float(x_true[0])), abs(float(x_true[1])))
                    rate_mag = max(abs(float(x_true[2])), abs(float(x_true[3])))
                    prev_phase = balance_phase
                    if balance_phase == "recovery":
                        if angle_mag < self.HOLD_ENTER_ANGLE_RAD and rate_mag < self.HOLD_ENTER_RATE_RAD_S:
                            balance_phase = "hold"
                            recovery_time_s = 0.0
                    else:
                        if angle_mag > self.HOLD_EXIT_ANGLE_RAD or rate_mag > self.HOLD_EXIT_RATE_RAD_S:
                            balance_phase = "recovery"
                            recovery_time_s = 0.0
                    if balance_phase != prev_phase:
                        phase_switch_count += 1
                    if balance_phase == "recovery":
                        recovery_time_s += control_dt

                z = np.concatenate([x_ctrl, u_eff_applied])
                du_lqr = -k_du @ z

                rw_frac = abs(float(x_est[4])) / max(self.MAX_WHEEL_SPEED_RAD_S, 1e-6)
                rw_damp_gain = 0.18 + 0.60 * max(0.0, rw_frac - 0.35)
                if paper_family:
                    xw = np.array([x_est[0], x_est[2], x_est[4]], dtype=float)
                    u_pitch = float(-(k_paper_pitch @ xw)[0])
                    s_roll = float(x_est[1] + 0.42 * x_est[3])
                    kf = self._fuzzy_roll_gain(
                        float(x_est[1]),
                        float(x_est[3]),
                        self.HOLD_EXIT_ANGLE_RAD,
                        self.HOLD_EXIT_RATE_RAD_S,
                    )
                    u_roll_sm = float(-kf * np.tanh(s_roll / max(np.radians(0.5), 1e-3)) * 0.45 * self.MAX_U[0])
                    du_rw_cmd = float((u_pitch + u_roll_sm) - u_eff_applied[0])
                elif family == "legacy_wheel_pid":
                    u_rw_target = float(-(30.0 * x_est[0] + 8.0 * x_est[2] + 0.08 * x_est[4]))
                    du_rw_cmd = float(u_rw_target - u_eff_applied[0])
                elif family == "legacy_wheel_lqr":
                    xw = np.array([x_est[0], x_est[2], x_est[4]], dtype=float)
                    u_rw_target = float(-(k_paper_pitch @ xw)[0])
                    du_rw_cmd = float(u_rw_target - u_eff_applied[0])
                elif family == "legacy_run_pd":
                    u_rw_target = float(-(22.0 * x_est[0] + 6.0 * x_est[2]))
                    du_rw_cmd = float(u_rw_target - u_eff_applied[0])
                elif mpc_family:
                    u_target = -k_mpc @ x_ctrl
                    du_rw_cmd = float(u_target[0] - u_eff_applied[0])
                elif robust_hinf_family:
                    robust_term = float(
                        -0.40 * x_est[0]
                        - 0.18 * x_est[2]
                        - 0.08 * x_est[4]
                        - 0.06 * x_est[1]
                        - 0.03 * x_est[3]
                    )
                    du_rw_cmd = float(du_lqr[0] + robust_term - u_eff_applied[0] * 0.20)
                else:
                    du_rw_cmd = float(du_lqr[0] - rw_damp_gain * x_est[4])
                    if hybrid_family:
                        du_rw_cmd += float(
                            -(0.12 * self.BASE_PITCH_KP) * x_est[0]
                            - (0.10 * self.BASE_PITCH_KD) * x_est[2]
                            + (0.06 * self.BASE_ROLL_KP) * x_est[1]
                            + (0.06 * self.BASE_ROLL_KD) * x_est[3]
                        )
                du_rw = float(np.clip(du_rw_cmd, -max_du[0], max_du[0]))
                u_rw_unc = float(u_eff_applied[0] + du_rw)
                u_rw_cmd = float(np.clip(u_rw_unc, -self.MAX_U[0], self.MAX_U[0]))
                wheel_speed_abs_est = abs(float(x_est[4]))
                wheel_derate_start_speed = self.WHEEL_TORQUE_DERATE_START * self.MAX_WHEEL_SPEED_RAD_S
                if wheel_speed_abs_est > wheel_derate_start_speed:
                    span = max(self.MAX_WHEEL_SPEED_RAD_S - wheel_derate_start_speed, 1e-6)
                    rw_scale = max(0.0, 1.0 - (wheel_speed_abs_est - wheel_derate_start_speed) / span)
                    rw_cap = self.MAX_U[0] * rw_scale
                    u_rw_cmd = float(np.clip(u_rw_cmd, -rw_cap, rw_cap))

                if low_spin_robust:
                    hard_frac = self.WHEEL_SPIN_HARD_FRAC
                    budget_speed = min(
                        self.WHEEL_SPIN_BUDGET_FRAC * self.MAX_WHEEL_SPEED_RAD_S,
                        self.WHEEL_SPIN_BUDGET_ABS_RAD_S,
                    )
                    hard_speed = min(
                        hard_frac * self.MAX_WHEEL_SPEED_RAD_S,
                        self.WHEEL_SPIN_HARD_ABS_RAD_S,
                    )
                    if high_spin_active:
                        high_spin_exit_speed = self.HIGH_SPIN_EXIT_FRAC * hard_speed
                        if wheel_speed_abs_est < high_spin_exit_speed:
                            high_spin_active = False
                    elif wheel_speed_abs_est > hard_speed:
                        high_spin_active = True
                    momentum_speed = min(self.WHEEL_MOMENTUM_THRESH_FRAC * self.MAX_WHEEL_SPEED_RAD_S, budget_speed)
                    if wheel_speed_abs_est > momentum_speed:
                        pre_span = max(budget_speed - momentum_speed, 1e-6)
                        pre_over = float(np.clip((wheel_speed_abs_est - momentum_speed) / pre_span, 0.0, 1.0))
                        u_rw_cmd += float(
                            np.clip(
                                -np.sign(x_est[4]) * 0.35 * self.WHEEL_MOMENTUM_K * pre_over * self.MAX_U[0],
                                -0.30 * self.MAX_U[0],
                                0.30 * self.MAX_U[0],
                            )
                        )

                    if wheel_speed_abs_est > budget_speed:
                        wheel_over_budget_count += 1
                        speed_span = max(hard_speed - budget_speed, 1e-6)
                        over = np.clip((wheel_speed_abs_est - budget_speed) / speed_span, 0.0, 1.5)
                        u_rw_cmd += float(
                            np.clip(
                                -np.sign(x_est[4]) * self.WHEEL_MOMENTUM_K * over * self.MAX_U[0],
                                -0.65 * self.MAX_U[0],
                                0.65 * self.MAX_U[0],
                            )
                        )
                        if (wheel_speed_abs_est <= hard_speed) and (not high_spin_active):
                            rw_cap_scale = max(0.55, 1.0 - 0.45 * float(over))
                        else:
                            wheel_over_hard_count += 1
                            rw_cap_scale = 0.35
                            tilt_mag = max(abs(float(x_est[0])), abs(float(x_est[1])))
                            if balance_phase == "recovery" and recovery_time_s < 0.12 and tilt_mag > self.HOLD_EXIT_ANGLE_RAD:
                                rw_cap_scale = max(rw_cap_scale, 0.55)
                            over_hard = float(np.clip((wheel_speed_abs_est - hard_speed) / max(hard_speed, 1e-6), 0.0, 1.0))
                            emergency_counter = -np.sign(x_est[4]) * (0.60 + 0.35 * over_hard) * self.MAX_U[0]
                            if balance_phase != "recovery" or recovery_time_s >= 0.12 or tilt_mag <= self.HOLD_EXIT_ANGLE_RAD:
                                if np.sign(u_rw_cmd) == np.sign(x_est[4]):
                                    u_rw_cmd = float(emergency_counter)
                            if np.sign(u_rw_cmd) != np.sign(x_est[4]):
                                min_counter = self.HIGH_SPIN_COUNTER_MIN_FRAC * self.MAX_U[0]
                                u_rw_cmd = float(np.sign(u_rw_cmd) * max(abs(u_rw_cmd), min_counter))
                            if high_spin_active:
                                u_rw_cmd = float(0.25 * u_rw_cmd + 0.75 * emergency_counter)
                        u_rw_cmd = float(np.clip(u_rw_cmd, -rw_cap_scale * self.MAX_U[0], rw_cap_scale * self.MAX_U[0]))

                tilt_span = max(self.BASE_TILT_FULL_AUTHORITY_RAD - self.BASE_TILT_DEADBAND_RAD, 1e-6)
                tilt_mag = max(abs(x_est[0]), abs(x_est[1]))
                base_authority = float(np.clip((tilt_mag - self.BASE_TILT_DEADBAND_RAD) / tilt_span, 0.0, 1.0))
                if tilt_mag > 1.5 * self.BASE_TILT_FULL_AUTHORITY_RAD:
                    base_authority *= 0.55
                if low_spin_robust:
                    if high_spin_active:
                        base_authority = max(base_authority, self.HIGH_SPIN_BASE_AUTHORITY_MIN)
                    max_auth_delta = self.BASE_AUTHORITY_RATE_PER_S * control_dt
                    base_authority_state += float(np.clip(base_authority - base_authority_state, -max_auth_delta, max_auth_delta))
                    base_authority = float(np.clip(base_authority_state, 0.0, 1.0))

                follow_alpha = float(np.clip(self.BASE_REF_FOLLOW_RATE_HZ * control_dt, 0.0, 1.0))
                recenter_alpha = float(np.clip(self.BASE_REF_RECENTER_RATE_HZ * control_dt, 0.0, 1.0))
                base_disp = float(np.hypot(x_est[5], x_est[6]))
                if base_authority > 0.35 and base_disp < self.BASE_HOLD_RADIUS_M:
                    base_ref[0] += follow_alpha * (x_est[5] - base_ref[0])
                    base_ref[1] += follow_alpha * (x_est[6] - base_ref[1])
                else:
                    base_ref[0] += recenter_alpha * (0.0 - base_ref[0])
                    base_ref[1] += recenter_alpha * (0.0 - base_ref[1])

                base_x_err = float(np.clip(x_est[5] - base_ref[0], -self.BASE_CENTERING_POS_CLIP_M, self.BASE_CENTERING_POS_CLIP_M))
                base_y_err = float(np.clip(x_est[6] - base_ref[1], -self.BASE_CENTERING_POS_CLIP_M, self.BASE_CENTERING_POS_CLIP_M))
                hold_x = -self.BASE_DAMPING_GAIN * x_est[7] - self.BASE_CENTERING_GAIN * base_x_err
                hold_y = -self.BASE_DAMPING_GAIN * x_est[8] - self.BASE_CENTERING_GAIN * base_y_err
                balance_x = self.BASE_COMMAND_GAIN * (self.BASE_PITCH_KP * x_est[0] + self.BASE_PITCH_KD * x_est[2])
                balance_y = -self.BASE_COMMAND_GAIN * (self.BASE_ROLL_KP * x_est[1] + self.BASE_ROLL_KD * x_est[3])
                if hybrid_family:
                    balance_x += float(-0.08 * self.BASE_ROLL_KP * x_est[1] - 0.05 * self.BASE_ROLL_KD * x_est[3])
                    balance_y += float(0.08 * self.BASE_PITCH_KP * x_est[0] + 0.05 * self.BASE_PITCH_KD * x_est[2])
                if paper_family:
                    s_roll = float(x_est[1] + 0.42 * x_est[3])
                    kf = self._fuzzy_roll_gain(
                        float(x_est[1]),
                        float(x_est[3]),
                        self.HOLD_EXIT_ANGLE_RAD,
                        self.HOLD_EXIT_RATE_RAD_S,
                    )
                    balance_y += float(-0.25 * kf * self.MAX_U[2] * np.tanh(s_roll / max(np.radians(0.5), 1e-3)))
                base_target_x = (1.0 - base_authority) * hold_x + base_authority * balance_x
                base_target_y = (1.0 - base_authority) * hold_y + base_authority * balance_y
                base_target_x += -params.ki_base * base_int[0]
                base_target_y += -params.ki_base * base_int[1]
                if legacy_family:
                    base_target_x = 0.0
                    base_target_y = 0.0
                if mpc_family:
                    u_target = -k_mpc @ x_ctrl
                    base_target_x = float(u_target[1])
                    base_target_y = float(u_target[2])
                if robust_hinf_family:
                    base_target_x += float(-0.12 * x_est[7] - 0.06 * x_est[5] - 0.06 * x_est[1])
                    base_target_y += float(-0.12 * x_est[8] - 0.06 * x_est[6] + 0.06 * x_est[0])
                if low_spin_robust:
                    budget_speed = min(
                        self.WHEEL_SPIN_BUDGET_FRAC * self.MAX_WHEEL_SPEED_RAD_S,
                        self.WHEEL_SPIN_BUDGET_ABS_RAD_S,
                    )
                    hard_speed = min(
                        self.WHEEL_SPIN_HARD_FRAC * self.MAX_WHEEL_SPEED_RAD_S,
                        self.WHEEL_SPIN_HARD_ABS_RAD_S,
                    )
                    if wheel_speed_abs_est > budget_speed:
                        over_budget = float(
                            np.clip(
                                (wheel_speed_abs_est - budget_speed) / max(hard_speed - budget_speed, 1e-6),
                                0.0,
                                1.0,
                            )
                        )
                        extra_bias = 1.25 if high_spin_active else 1.0
                        base_target_x += -np.sign(x_est[4]) * self.WHEEL_TO_BASE_BIAS_GAIN * extra_bias * over_budget

                du_base_cmd = np.array([base_target_x, base_target_y]) - u_eff_applied[1:]
                base_du_limit = max_du[1:].copy()
                if low_spin_robust:
                    near_upright_for_base = (
                        abs(x_true[0]) < self.UPRIGHT_ANGLE_THRESH
                        and abs(x_true[1]) < self.UPRIGHT_ANGLE_THRESH
                        and abs(x_true[2]) < self.UPRIGHT_VEL_THRESH
                        and abs(x_true[3]) < self.UPRIGHT_VEL_THRESH
                    )
                    if near_upright_for_base:
                        base_du_limit *= self.UPRIGHT_BASE_DU_SCALE
                du_base = np.clip(du_base_cmd, -base_du_limit, base_du_limit)
                u_base_unc = u_eff_applied[1:] + du_base
                u_base_cmd = np.clip(u_base_unc, -self.MAX_U[1:], self.MAX_U[1:])
                if low_spin_robust:
                    base_lpf_alpha = float(np.clip(self.BASE_COMMAND_LPF_HZ * control_dt, 0.0, 1.0))
                    u_base_smooth += base_lpf_alpha * (u_base_cmd - u_base_smooth)
                    u_base_cmd = u_base_smooth.copy()

                du_hits += int((abs(du_rw_cmd) > max_du[0]) or np.any(np.abs(du_base_cmd) > base_du_limit))
                sat_hits += int((abs(u_rw_unc) > self.MAX_U[0]) or np.any(np.abs(u_base_unc) > self.MAX_U[1:]))
                u_cmd = np.array([u_rw_cmd, u_base_cmd[0], u_base_cmd[1]], dtype=float)

                near_upright = (
                    abs(x_true[0]) < self.UPRIGHT_ANGLE_THRESH
                    and abs(x_true[1]) < self.UPRIGHT_ANGLE_THRESH
                    and abs(x_true[2]) < self.UPRIGHT_VEL_THRESH
                    and abs(x_true[3]) < self.UPRIGHT_VEL_THRESH
                    and abs(x_true[5]) < self.UPRIGHT_POS_THRESH
                    and abs(x_true[6]) < self.UPRIGHT_POS_THRESH
                )
                if near_upright:
                    upright_target = 1.0
                else:
                    upright_target = 0.0
                if low_spin_robust:
                    quasi_upright = (
                        abs(x_true[0]) < 1.8 * self.UPRIGHT_ANGLE_THRESH
                        and abs(x_true[1]) < 1.8 * self.UPRIGHT_ANGLE_THRESH
                        and abs(x_true[2]) < 1.8 * self.UPRIGHT_VEL_THRESH
                        and abs(x_true[3]) < 1.8 * self.UPRIGHT_VEL_THRESH
                    )
                    if quasi_upright and not near_upright:
                        upright_target = 0.35
                blend_alpha = self.UPRIGHT_BLEND_RISE if (low_spin_robust and upright_target > upright_blend) else (
                    self.UPRIGHT_BLEND_FALL if low_spin_robust else 0.20
                )
                upright_blend += blend_alpha * (upright_target - upright_blend)
                if upright_blend > 1e-6:
                    bleed_scale = 1.0 - upright_blend * (1.0 - params.u_bleed)
                    if low_spin_robust and high_spin_active:
                        u_cmd[1:] *= bleed_scale
                    else:
                        u_cmd *= bleed_scale
                    phase_scale = 1.0
                    if low_spin_robust:
                        phase_scale = self.HOLD_WHEEL_DESPIN_SCALE if balance_phase == "hold" else self.RECOVERY_WHEEL_DESPIN_SCALE
                    u_cmd[0] += upright_blend * float(
                        np.clip(-phase_scale * despin_gain * x_est[4], -0.50 * self.MAX_U[0], 0.50 * self.MAX_U[0])
                    )
                    u_cmd[np.abs(u_cmd) < 1e-3] = 0.0

                command_jerk_acc += float(np.sum(np.abs(u_cmd - u_applied)))
                sign_flip_count += int(np.sum((u_cmd * u_applied) < 0.0))

                if config.hardware_realistic:
                    cmd_queue.append(u_cmd.copy())
                    u_applied = cmd_queue.popleft()
                else:
                    u_applied = u_cmd
            if low_spin_robust and balance_phase == "hold":
                hold_steps += 1
            if low_spin_robust and high_spin_active:
                high_spin_steps += 1

            self.data.ctrl[:] = 0.0
            wheel_speed = float(self.data.qvel[self.v_rw])
            wheel_speed_abs = abs(wheel_speed)
            wheel_limit = self.MAX_U[0]
            wheel_derate_start_speed = self.WHEEL_TORQUE_DERATE_START * self.MAX_WHEEL_SPEED_RAD_S
            if wheel_speed_abs > wheel_derate_start_speed:
                span = max(self.MAX_WHEEL_SPEED_RAD_S - wheel_derate_start_speed, 1e-6)
                wheel_scale = max(0.0, 1.0 - (wheel_speed_abs - wheel_derate_start_speed) / span)
                wheel_limit *= wheel_scale
            wheel_cmd = float(np.clip(u_applied[0], -wheel_limit, wheel_limit))
            hard_speed = min(self.WHEEL_SPIN_HARD_FRAC * self.MAX_WHEEL_SPEED_RAD_S, self.WHEEL_SPIN_HARD_ABS_RAD_S)
            if wheel_speed_abs >= hard_speed and np.sign(wheel_cmd) == np.sign(wheel_speed):
                wheel_cmd *= 0.03
            if wheel_speed_abs >= hard_speed and abs(wheel_speed) > 1e-9:
                over_hard = float(np.clip((wheel_speed_abs - hard_speed) / max(hard_speed, 1e-6), 0.0, 1.0))
                min_counter = (0.55 + 0.45 * over_hard) * self.HIGH_SPIN_COUNTER_MIN_FRAC * wheel_limit
                if abs(wheel_cmd) < 1e-9:
                    wheel_cmd = -np.sign(wheel_speed) * min_counter
                elif np.sign(wheel_cmd) != np.sign(wheel_speed):
                    wheel_cmd = float(np.sign(wheel_cmd) * max(abs(wheel_cmd), min_counter))
            if wheel_speed_abs >= self.MAX_WHEEL_SPEED_RAD_S and np.sign(wheel_cmd) == np.sign(wheel_speed):
                wheel_cmd = 0.0
            self.data.ctrl[self.aid_rw] = wheel_cmd

            base_x_speed = float(self.data.qvel[self.v_base_x])
            base_y_speed = float(self.data.qvel[self.v_base_y])
            base_derate_start = self.BASE_TORQUE_DERATE_START * self.MAX_BASE_SPEED_M_S
            bx_scale = 1.0
            by_scale = 1.0
            if abs(base_x_speed) > base_derate_start:
                base_margin = max(self.MAX_BASE_SPEED_M_S - base_derate_start, 1e-6)
                bx_scale = max(0.0, 1.0 - (abs(base_x_speed) - base_derate_start) / base_margin)
            if abs(base_y_speed) > base_derate_start:
                base_margin = max(self.MAX_BASE_SPEED_M_S - base_derate_start, 1e-6)
                by_scale = max(0.0, 1.0 - (abs(base_y_speed) - base_derate_start) / base_margin)
            base_x_cmd = float(u_applied[1] * bx_scale)
            base_y_cmd = float(u_applied[2] * by_scale)

            soft_speed = self.BASE_SPEED_SOFT_LIMIT_FRAC * self.MAX_BASE_SPEED_M_S
            if abs(base_x_speed) > soft_speed and np.sign(base_x_cmd) == np.sign(base_x_speed):
                span = max(self.MAX_BASE_SPEED_M_S - soft_speed, 1e-6)
                scale = max(0.0, 1.0 - (abs(base_x_speed) - soft_speed) / span)
                base_x_cmd *= scale
            if abs(base_y_speed) > soft_speed and np.sign(base_y_cmd) == np.sign(base_y_speed):
                span = max(self.MAX_BASE_SPEED_M_S - soft_speed, 1e-6)
                scale = max(0.0, 1.0 - (abs(base_y_speed) - soft_speed) / span)
                base_y_cmd *= scale

            base_x = float(self.data.qpos[self.q_base_x])
            base_y = float(self.data.qpos[self.q_base_y])
            if abs(base_x) > self.BASE_HOLD_RADIUS_M and np.sign(base_x_cmd) == np.sign(base_x):
                base_x_cmd *= 0.4
            if abs(base_y) > self.BASE_HOLD_RADIUS_M and np.sign(base_y_cmd) == np.sign(base_y):
                base_y_cmd *= 0.4

            base_x_cmd = float(np.clip(base_x_cmd, -self.BASE_FORCE_SOFT_LIMIT, self.BASE_FORCE_SOFT_LIMIT))
            base_y_cmd = float(np.clip(base_y_cmd, -self.BASE_FORCE_SOFT_LIMIT, self.BASE_FORCE_SOFT_LIMIT))
            self.data.ctrl[self.aid_base_x] = base_x_cmd
            self.data.ctrl[self.aid_base_y] = base_y_cmd
            u_eff_applied[:] = [wheel_cmd, base_x_cmd, base_y_cmd]

            if step % config.disturbance_interval == 0:
                force = np.array(
                    [
                        rng.uniform(-config.disturbance_magnitude_xy, config.disturbance_magnitude_xy) * disturbance_scale,
                        rng.uniform(-config.disturbance_magnitude_xy, config.disturbance_magnitude_xy) * disturbance_scale,
                        rng.uniform(-config.disturbance_magnitude_z, config.disturbance_magnitude_z) * disturbance_scale,
                    ]
                )
                self.data.xfrc_applied[self.stick_body_id, :3] = force
            else:
                self.data.xfrc_applied[self.stick_body_id, :3] = 0.0

            mujoco.mj_step(self.model, self.data)

            pitch = float(self.data.qpos[self.q_pitch])
            roll = float(self.data.qpos[self.q_roll])
            bx = float(self.data.qpos[self.q_base_x])
            by = float(self.data.qpos[self.q_base_y])

            max_pitch = max(max_pitch, abs(pitch))
            max_roll = max(max_roll, abs(roll))
            max_base_x = max(max_base_x, abs(bx))
            max_base_y = max(max_base_y, abs(by))
            drift_sq_acc += bx * bx + by * by
            control_energy += float(np.dot(u_applied, u_applied))
            motion_activity_acc += (
                abs(float(self.data.qvel[self.v_pitch]))
                + abs(float(self.data.qvel[self.v_roll]))
                + 0.25 * abs(float(self.data.qvel[self.v_base_x]))
                + 0.25 * abs(float(self.data.qvel[self.v_base_y]))
            )
            pitch_rate_series.append(float(self.data.qvel[self.v_pitch]))
            roll_rate_series.append(float(self.data.qvel[self.v_roll]))
            vx_series.append(float(self.data.qvel[self.v_base_x]))
            vy_series.append(float(self.data.qvel[self.v_base_y]))

            if abs(pitch) >= tilt_fail_rad or abs(roll) >= tilt_fail_rad:
                crash_step = step
                break

        steps_run = crash_step if crash_step is not None else config.steps
        updates_run = max(control_updates, 1)
        if crash_step is not None:
            max_pitch = min(max_pitch, tilt_fail_rad)
            max_roll = min(max_roll, tilt_fail_rad)
        osc_band_energy = self._oscillation_band_energy(pitch_rate_series, roll_rate_series, vx_series, vy_series)
        hw_consistency = 1.0
        hw_traj_nrmse = np.nan
        if config.hardware_replay and config.hardware_trace_path:
            hw_consistency, hw_traj_nrmse = self._hardware_trace_consistency(
                trace_path=config.hardware_trace_path,
                pred_pitch_series=[float(v) for v in pitch_rate_series],
                pred_roll_series=[float(v) for v in roll_rate_series],
                pred_vx_series=[float(v) for v in vx_series],
                pred_vy_series=[float(v) for v in vy_series],
            )
        return {
            "survived": 1.0 if crash_step is None else 0.0,
            "crash_count": 0.0 if crash_step is None else 1.0,
            "crash_step": float(crash_step) if crash_step is not None else np.nan,
            "max_abs_pitch_deg": float(np.degrees(max_pitch)),
            "max_abs_roll_deg": float(np.degrees(max_roll)),
            "max_abs_base_x_m": max_base_x,
            "max_abs_base_y_m": max_base_y,
            "rms_base_drift_m": float(np.sqrt(drift_sq_acc / max(steps_run, 1))),
            "control_energy": control_energy / max(steps_run, 1),
            "mean_command_jerk": command_jerk_acc / updates_run,
            "mean_motion_activity": motion_activity_acc / max(steps_run, 1),
            "ctrl_sign_flip_rate": sign_flip_count / updates_run,
            "osc_band_energy": osc_band_energy,
            "sat_rate_abs": sat_hits / updates_run,
            "sat_rate_du": du_hits / updates_run,
            "phase_switch_count": float(phase_switch_count),
            "hold_phase_ratio": float(hold_steps / max(steps_run, 1)),
            "wheel_over_budget": float(wheel_over_budget_count),
            "wheel_over_hard": float(wheel_over_hard_count),
            "high_spin_active_ratio": float(high_spin_steps / max(steps_run, 1)),
            "sim_real_consistency": float(hw_consistency),
            "sim_real_traj_nrmse": float(hw_traj_nrmse),
        }

    def evaluate_candidate(
        self,
        params: CandidateParams,
        episode_seeds: List[int],
        config: EpisodeConfig,
    ) -> Dict[str, float]:
        per_episode = [self.simulate_episode(params, s, config) for s in episode_seeds]
        episode_score_list = [self._episode_composite_score(m, config.steps) for m in per_episode]

        survival_rate = float(np.mean([m["survived"] for m in per_episode]))
        worst_pitch = float(np.max([m["max_abs_pitch_deg"] for m in per_episode]))
        worst_roll = float(np.max([m["max_abs_roll_deg"] for m in per_episode]))
        worst_tilt = max(worst_pitch, worst_roll)
        worst_base_x = float(np.max([m["max_abs_base_x_m"] for m in per_episode]))
        worst_base_y = float(np.max([m["max_abs_base_y_m"] for m in per_episode]))
        mean_rms_drift = float(np.mean([m["rms_base_drift_m"] for m in per_episode]))
        mean_energy = float(np.mean([m["control_energy"] for m in per_episode]))
        mean_jerk = float(np.mean([m["mean_command_jerk"] for m in per_episode]))
        mean_activity = float(np.mean([m["mean_motion_activity"] for m in per_episode]))
        mean_flip_rate = float(np.mean([m["ctrl_sign_flip_rate"] for m in per_episode]))
        mean_osc_band = float(np.mean([m["osc_band_energy"] for m in per_episode]))
        mean_sat_abs = float(np.mean([m["sat_rate_abs"] for m in per_episode]))
        mean_sat_du = float(np.mean([m["sat_rate_du"] for m in per_episode]))
        phase_switch_count_mean = float(np.mean([m["phase_switch_count"] for m in per_episode]))
        hold_phase_ratio_mean = float(np.mean([m["hold_phase_ratio"] for m in per_episode]))
        wheel_over_budget_mean = float(np.mean([m["wheel_over_budget"] for m in per_episode]))
        wheel_over_hard_mean = float(np.mean([m["wheel_over_hard"] for m in per_episode]))
        high_spin_active_ratio_mean = float(np.mean([m.get("high_spin_active_ratio", 0.0) for m in per_episode]))
        sim_real_consistency_mean = float(np.mean([m.get("sim_real_consistency", 0.0) for m in per_episode]))
        sim_real_vals = np.asarray([m.get("sim_real_traj_nrmse", np.nan) for m in per_episode], dtype=float)
        sim_real_traj_nrmse_mean = float(np.nanmean(sim_real_vals)) if np.any(np.isfinite(sim_real_vals)) else np.nan
        crash_count_total = float(np.sum([m["crash_count"] for m in per_episode]))
        crash_rate = 1.0 - survival_rate
        score_composite = float(np.mean(episode_score_list))
        score_p5 = float(np.quantile(np.asarray(episode_score_list, dtype=float), 0.05)) if episode_score_list else np.nan
        score_p1 = float(np.quantile(np.asarray(episode_score_list, dtype=float), 0.01)) if episode_score_list else np.nan

        crash_steps = [m["crash_step"] for m in per_episode if not np.isnan(m["crash_step"])]
        worst_crash_step = float(np.min(crash_steps)) if crash_steps else np.nan
        worst_base = max(worst_base_x, worst_base_y)
        ok_survival = bool(survival_rate == 1.0)
        ok_tilt = bool(worst_tilt < config.max_worst_tilt_deg)
        ok_base = bool(worst_base < config.max_worst_base_m)
        ok_sat_du = bool(mean_sat_du <= config.max_mean_sat_rate_du)
        ok_sat_abs = bool(mean_sat_abs <= config.max_mean_sat_rate_abs)
        accepted_gate = bool(ok_survival and ok_tilt and ok_base and ok_sat_du and ok_sat_abs)

        failure_tags: List[str] = []
        if not ok_survival:
            failure_tags.append("gate_survival")
        if not ok_tilt:
            failure_tags.append("gate_tilt")
        if not ok_base:
            failure_tags.append("gate_base_bound")
        if not ok_sat_du:
            failure_tags.append("gate_sat_du")
        if not ok_sat_abs:
            failure_tags.append("gate_sat_abs")
        if config.hardware_replay and sim_real_consistency_mean < 0.55:
            failure_tags.append("gate_sim_real")
        failure_reason = "+".join(failure_tags)
        if accepted_gate and score_composite >= 90.0:
            confidence_tier = "best_in_class_candidate"
        elif accepted_gate and score_composite >= 75.0:
            confidence_tier = "strong"
        else:
            confidence_tier = "exploratory"

        return {
            "survival_rate": survival_rate,
            "score_composite": score_composite,
            "score_p5": score_p5,
            "score_p1": score_p1,
            "episode_score_list": [float(v) for v in episode_score_list],
            "accepted_gate": accepted_gate,
            "worst_crash_step": worst_crash_step,
            "worst_pitch_deg": worst_pitch,
            "worst_roll_deg": worst_roll,
            "worst_base_x_m": worst_base_x,
            "worst_base_y_m": worst_base_y,
            "mean_rms_base_drift_m": mean_rms_drift,
            "mean_control_energy": mean_energy,
            "mean_command_jerk": mean_jerk,
            "mean_motion_activity": mean_activity,
            "mean_ctrl_sign_flip_rate": mean_flip_rate,
            "mean_osc_band_energy": mean_osc_band,
            "mean_sat_rate_abs": mean_sat_abs,
            "mean_sat_rate_du": mean_sat_du,
            "phase_switch_count_mean": phase_switch_count_mean,
            "hold_phase_ratio_mean": hold_phase_ratio_mean,
            "wheel_over_budget_mean": wheel_over_budget_mean,
            "wheel_over_hard_mean": wheel_over_hard_mean,
            "high_spin_active_ratio_mean": high_spin_active_ratio_mean,
            "sim_real_consistency_mean": sim_real_consistency_mean,
            "sim_real_traj_nrmse_mean": sim_real_traj_nrmse_mean,
            "crash_count_total": crash_count_total,
            "crash_rate": crash_rate,
            "failure_reason": failure_reason,
            "confidence_tier": confidence_tier,
            "hardware_realistic": bool(config.hardware_realistic),
            "control_hz": float(config.control_hz),
            "control_delay_steps": int(config.control_delay_steps),
            "preset": str(config.preset),
            "stability_profile": str(config.stability_profile),
            "controller_family": str(config.controller_family),
            "model_variant_id": str(config.model_variant_id),
            "domain_profile_id": str(config.domain_profile_id),
            "hardware_replay": bool(config.hardware_replay),
        }

    def _oscillation_band_energy(
        self,
        pitch_rate: List[float],
        roll_rate: List[float],
        vx: List[float],
        vy: List[float],
        f_low: float = 2.0,
        f_high: float = 8.0,
    ) -> float:
        def band_power(x: np.ndarray) -> float:
            n = x.size
            if n < 16:
                return 0.0
            x = x - np.mean(x)
            freq = np.fft.rfftfreq(n, d=self.dt)
            psd = np.abs(np.fft.rfft(x)) ** 2 / n
            mask = (freq >= f_low) & (freq <= f_high)
            return float(np.sum(psd[mask]))

        p = band_power(np.asarray(pitch_rate))
        r = band_power(np.asarray(roll_rate))
        bx = band_power(np.asarray(vx))
        by = band_power(np.asarray(vy))
        return p + r + 0.25 * bx + 0.25 * by

    def _hardware_trace_consistency(
        self,
        trace_path: str,
        pred_pitch_series: List[float],
        pred_roll_series: List[float],
        pred_vx_series: List[float],
        pred_vy_series: List[float],
    ) -> tuple[float, float]:
        path = Path(trace_path)
        if (not path.exists()) or (not path.is_file()):
            return 0.0, np.nan
        obs_pitch = []
        obs_roll = []
        obs_vx = []
        obs_vy = []
        try:
            with path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    obs_pitch.append(float(row.get("pitch_rate", "nan")))
                    obs_roll.append(float(row.get("roll_rate", "nan")))
                    obs_vx.append(float(row.get("base_vx", "nan")))
                    obs_vy.append(float(row.get("base_vy", "nan")))
        except Exception:
            return 0.0, np.nan

        n = min(len(obs_pitch), len(pred_pitch_series), len(pred_roll_series), len(pred_vx_series), len(pred_vy_series))
        if n < 16:
            return 0.0, np.nan
        obs = np.vstack(
            [
                np.asarray(obs_pitch[:n], dtype=float),
                np.asarray(obs_roll[:n], dtype=float),
                np.asarray(obs_vx[:n], dtype=float),
                np.asarray(obs_vy[:n], dtype=float),
            ]
        )
        pred = np.vstack(
            [
                np.asarray(pred_pitch_series[:n], dtype=float),
                np.asarray(pred_roll_series[:n], dtype=float),
                np.asarray(pred_vx_series[:n], dtype=float),
                np.asarray(pred_vy_series[:n], dtype=float),
            ]
        )
        finite = np.all(np.isfinite(obs), axis=0) & np.all(np.isfinite(pred), axis=0)
        if np.count_nonzero(finite) < 16:
            return 0.0, np.nan
        obs = obs[:, finite]
        pred = pred[:, finite]
        mse = float(np.mean((pred - obs) ** 2))
        var = float(np.var(obs))
        nrmse = float(np.sqrt(mse / max(var, 1e-9)))
        consistency = float(np.clip(1.0 - nrmse, 0.0, 1.0))
        return consistency, nrmse


def safe_evaluate_candidate(
    evaluator: ControllerEvaluator,
    params: CandidateParams,
    episode_seeds: List[int],
    config: EpisodeConfig,
) -> Dict[str, float]:
    try:
        return evaluator.evaluate_candidate(params, episode_seeds, config)
    except (LinAlgError, RuntimeError, ValueError, FloatingPointError):
        return {
            "survival_rate": 0.0,
            "score_composite": -1e9,
            "score_p5": -1e9,
            "score_p1": -1e9,
            "episode_score_list": [],
            "crash_rate": 1.0,
            "crash_count_total": float(len(episode_seeds)),
            "accepted_gate": False,
            "worst_crash_step": np.nan,
            "worst_pitch_deg": np.nan,
            "worst_roll_deg": np.nan,
            "worst_base_x_m": np.nan,
            "worst_base_y_m": np.nan,
            "mean_rms_base_drift_m": np.nan,
            "mean_control_energy": np.nan,
            "mean_command_jerk": np.nan,
            "mean_motion_activity": np.nan,
            "mean_ctrl_sign_flip_rate": np.nan,
            "mean_osc_band_energy": np.nan,
            "mean_sat_rate_abs": np.nan,
            "mean_sat_rate_du": np.nan,
            "phase_switch_count_mean": 0.0,
            "hold_phase_ratio_mean": 0.0,
            "wheel_over_budget_mean": 0.0,
            "wheel_over_hard_mean": 0.0,
            "high_spin_active_ratio_mean": 0.0,
            "sim_real_consistency_mean": 0.0,
            "sim_real_traj_nrmse_mean": np.nan,
            "failure_reason": "riccati_failure",
            "confidence_tier": "exploratory",
            "hardware_realistic": bool(config.hardware_realistic),
            "control_hz": float(config.control_hz),
            "control_delay_steps": int(config.control_delay_steps),
            "preset": str(config.preset),
            "stability_profile": str(config.stability_profile),
            "controller_family": str(config.controller_family),
            "model_variant_id": str(config.model_variant_id),
            "domain_profile_id": str(config.domain_profile_id),
            "hardware_replay": bool(config.hardware_replay),
        }
