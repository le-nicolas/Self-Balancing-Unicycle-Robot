import math
import csv
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import mujoco
import numpy as np
from scipy.linalg import LinAlgError, solve_discrete_are
import runtime_config as runtime_cfg
from mpc_controller import MPCController


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
    init_pitch_deg_override: float | None = None
    init_roll_deg_override: float | None = None
    init_base_x_m_override: float | None = None
    init_base_y_m_override: float | None = None
    linearize_pitch_deg: float = 0.0
    linearize_roll_deg: float = 0.0
    max_worst_tilt_deg: float = 20.0
    max_worst_base_m: float = 4.0
    max_mean_sat_rate_du: float = 0.98
    max_mean_sat_rate_abs: float = 0.90
    crash_divergence_gate: bool = True
    crash_recovery_steps: int = 500
    rw_emergency_du: bool = True
    rw_emergency_pitch_deg: float = 15.0
    rw_emergency_du_scale: float = 1.5
    hold_base_x_centering_gain: float = 0.0
    mode: str = "smooth"
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
    dob_cutoff_hz: float = 5.0
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

    @staticmethod
    def _lift_discrete_dynamics(A: np.ndarray, B: np.ndarray, steps: int) -> tuple[np.ndarray, np.ndarray]:
        n = max(int(steps), 1)
        if n == 1:
            return A.copy(), B.copy()
        A_lift = np.linalg.matrix_power(A, n)
        acc = np.zeros_like(A)
        Ak = np.eye(A.shape[0], dtype=float)
        for _ in range(n):
            acc += Ak
            Ak = A @ Ak
        B_lift = acc @ B
        return A_lift, B_lift

    def __init__(self, xml_path: Path):
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep

        self._runtime_cfg_cache: dict[tuple[object, ...], runtime_cfg.RuntimeConfig] = {}
        self._linearization_cache: dict[tuple[float, float], tuple[np.ndarray, np.ndarray]] = {}
        self._sync_runtime_tuning(self._resolve_runtime_cfg(EpisodeConfig()))
        self._resolve_ids()
        self.A, self.B = self._linearize()
        self.NX = self.A.shape[0]
        self.NU = self.B.shape[1]
        self._validate_xml_ctrlrange()

    def _resolve_runtime_cfg(self, config: EpisodeConfig) -> runtime_cfg.RuntimeConfig:
        key = (
            str(config.mode),
            str(config.preset),
            str(config.stability_profile),
            bool(config.hardware_realistic),
            float(config.control_hz),
            int(config.control_delay_steps),
            int(config.wheel_encoder_ticks_per_rev),
            round(float(config.imu_angle_noise_std_rad), 12),
            round(float(config.imu_rate_noise_std_rad_s), 12),
            round(float(config.wheel_encoder_rate_noise_std_rad_s), 12),
            str(config.controller_family),
            bool(config.crash_divergence_gate),
            int(config.crash_recovery_steps),
            bool(config.rw_emergency_du),
            round(float(config.rw_emergency_pitch_deg), 8),
            round(float(config.rw_emergency_du_scale), 8),
            round(float(config.hold_base_x_centering_gain), 8),
            round(float(config.dob_cutoff_hz), 8),
            round(float(config.linearize_pitch_deg), 8),
            round(float(config.linearize_roll_deg), 8),
        )
        cached = self._runtime_cfg_cache.get(key)
        if cached is not None:
            return cached

        controller_family = str(config.controller_family)
        if controller_family not in {"current", "current_dob", "hybrid_modern", "paper_split_baseline"}:
            controller_family = "current"

        argv = [
            "--mode",
            str(config.mode),
            "--preset",
            str(config.preset),
            "--stability-profile",
            str(config.stability_profile),
            "--control-hz",
            f"{float(config.control_hz)}",
            "--control-delay-steps",
            str(int(config.control_delay_steps)),
            "--wheel-encoder-ticks",
            str(int(config.wheel_encoder_ticks_per_rev)),
            "--imu-angle-noise-deg",
            f"{float(np.degrees(config.imu_angle_noise_std_rad))}",
            "--imu-rate-noise",
            f"{float(config.imu_rate_noise_std_rad_s)}",
            "--wheel-rate-noise",
            f"{float(config.wheel_encoder_rate_noise_std_rad_s)}",
            "--crash-recovery-steps",
            str(int(config.crash_recovery_steps)),
            "--rw-emergency-pitch-deg",
            f"{float(config.rw_emergency_pitch_deg)}",
            "--rw-emergency-du-scale",
            f"{float(config.rw_emergency_du_scale)}",
            "--hold-base-x-centering-gain",
            f"{float(config.hold_base_x_centering_gain)}",
            "--linearize-pitch-deg",
            f"{float(config.linearize_pitch_deg)}",
            "--linearize-roll-deg",
            f"{float(config.linearize_roll_deg)}",
            "--controller-family",
            controller_family,
        ]
        argv.append("--crash-gate-divergence" if bool(config.crash_divergence_gate) else "--no-crash-gate-divergence")
        argv.append("--rw-emergency-du" if bool(config.rw_emergency_du) else "--no-rw-emergency-du")
        if controller_family == "current_dob":
            argv.append("--enable-dob")
            dob_cutoff_hz = float(max(config.dob_cutoff_hz, 0.0))
            if dob_cutoff_hz > 0.0:
                argv.extend(["--dob-cutoff-hz", f"{dob_cutoff_hz}"])
        if not config.hardware_realistic:
            argv.append("--legacy-model")

        parsed = runtime_cfg.parse_args(argv)
        resolved = runtime_cfg.build_config(parsed)
        self._runtime_cfg_cache[key] = resolved
        return resolved

    def _sync_runtime_tuning(self, cfg: runtime_cfg.RuntimeConfig) -> None:
        self.X_REF = float(cfg.x_ref)
        self.Y_REF = float(cfg.y_ref)
        self.LINEARIZE_PITCH_RAD = float(getattr(cfg, "linearize_pitch_rad", 0.0))
        self.LINEARIZE_ROLL_RAD = float(getattr(cfg, "linearize_roll_rad", 0.0))
        self.INT_CLAMP = float(cfg.int_clamp)
        self.UPRIGHT_ANGLE_THRESH = float(cfg.upright_angle_thresh)
        self.UPRIGHT_VEL_THRESH = float(cfg.upright_vel_thresh)
        self.UPRIGHT_POS_THRESH = float(cfg.upright_pos_thresh)
        self.CRASH_ANGLE_RAD = float(cfg.crash_angle_rad)
        self.CRASH_DIVERGENCE_GATE_ENABLED = bool(cfg.crash_divergence_gate_enabled)
        self.CRASH_RECOVERY_WINDOW_STEPS = int(max(cfg.crash_recovery_window_steps, 0))
        self.MAX_U = np.asarray(cfg.max_u, dtype=float).copy()
        self.RW_EMERGENCY_DU_ENABLED = bool(cfg.rw_emergency_du_enabled)
        self.RW_EMERGENCY_PITCH_RAD = float(max(cfg.rw_emergency_pitch_rad, 0.0))
        self.RW_EMERGENCY_DU_SCALE = float(max(cfg.rw_emergency_du_scale, 1.0))
        self.MAX_WHEEL_SPEED_RAD_S = float(cfg.max_wheel_speed_rad_s)
        self.MAX_BASE_SPEED_M_S = float(cfg.max_base_speed_m_s)
        self.WHEEL_TORQUE_DERATE_START = float(cfg.wheel_torque_derate_start)
        self.BASE_TORQUE_DERATE_START = float(cfg.base_torque_derate_start)
        self.BASE_FORCE_SOFT_LIMIT = float(cfg.base_force_soft_limit)
        self.BASE_DAMPING_GAIN = float(cfg.base_damping_gain)
        self.BASE_CENTERING_GAIN = float(cfg.base_centering_gain)
        self.HOLD_BASE_X_CENTERING_GAIN = float(max(getattr(cfg, "hold_base_x_centering_gain", 0.0), 0.0))
        self.BASE_TILT_DEADBAND_RAD = float(cfg.base_tilt_deadband_rad)
        self.BASE_TILT_FULL_AUTHORITY_RAD = float(cfg.base_tilt_full_authority_rad)
        self.BASE_COMMAND_GAIN = float(cfg.base_command_gain)
        self.BASE_CENTERING_POS_CLIP_M = float(cfg.base_centering_pos_clip_m)
        self.BASE_SPEED_SOFT_LIMIT_FRAC = float(cfg.base_speed_soft_limit_frac)
        self.BASE_HOLD_RADIUS_M = float(cfg.base_hold_radius_m)
        self.BASE_REF_FOLLOW_RATE_HZ = float(cfg.base_ref_follow_rate_hz)
        self.BASE_REF_RECENTER_RATE_HZ = float(cfg.base_ref_recenter_rate_hz)
        self.BASE_PITCH_KP = float(cfg.base_pitch_kp)
        self.BASE_PITCH_KD = float(cfg.base_pitch_kd)
        self.BASE_ROLL_KP = float(cfg.base_roll_kp)
        self.BASE_ROLL_KD = float(cfg.base_roll_kd)
        self.BASE_AUTHORITY_RATE_PER_S = float(cfg.base_authority_rate_per_s)
        self.BASE_COMMAND_LPF_HZ = float(cfg.base_command_lpf_hz)
        self.UPRIGHT_BASE_DU_SCALE = float(cfg.upright_base_du_scale)
        self.WHEEL_MOMENTUM_THRESH_FRAC = float(cfg.wheel_momentum_thresh_frac)
        self.WHEEL_MOMENTUM_K = float(cfg.wheel_momentum_k)
        self.WHEEL_MOMENTUM_UPRIGHT_K = float(cfg.wheel_momentum_upright_k)
        self.HOLD_ENTER_ANGLE_RAD = float(cfg.hold_enter_angle_rad)
        self.HOLD_EXIT_ANGLE_RAD = float(cfg.hold_exit_angle_rad)
        self.HOLD_ENTER_RATE_RAD_S = float(cfg.hold_enter_rate_rad_s)
        self.HOLD_EXIT_RATE_RAD_S = float(cfg.hold_exit_rate_rad_s)
        self.WHEEL_SPIN_BUDGET_FRAC = float(cfg.wheel_spin_budget_frac)
        self.WHEEL_SPIN_HARD_FRAC = float(cfg.wheel_spin_hard_frac)
        self.WHEEL_SPIN_BUDGET_ABS_RAD_S = float(cfg.wheel_spin_budget_abs_rad_s)
        self.WHEEL_SPIN_HARD_ABS_RAD_S = float(cfg.wheel_spin_hard_abs_rad_s)
        self.HIGH_SPIN_EXIT_FRAC = float(cfg.high_spin_exit_frac)
        self.HIGH_SPIN_COUNTER_MIN_FRAC = float(cfg.high_spin_counter_min_frac)
        self.HIGH_SPIN_BASE_AUTHORITY_MIN = float(cfg.high_spin_base_authority_min)
        self.WHEEL_TO_BASE_BIAS_GAIN = float(cfg.wheel_to_base_bias_gain)
        self.RECOVERY_WHEEL_DESPIN_SCALE = float(cfg.recovery_wheel_despin_scale)
        self.HOLD_WHEEL_DESPIN_SCALE = float(cfg.hold_wheel_despin_scale)
        self.UPRIGHT_BLEND_RISE = float(cfg.upright_blend_rise)
        self.UPRIGHT_BLEND_FALL = float(cfg.upright_blend_fall)
        self.lock_root_attitude = bool(getattr(cfg, "lock_root_attitude", True))
        self.base_integrator_enabled = bool(cfg.base_integrator_enabled)

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
        self.jid_base_free = jid("base_free")
        qadr_base_free = int(self.model.jnt_qposadr[self.jid_base_free])
        dadr_base_free = int(self.model.jnt_dofadr[self.jid_base_free])

        self.q_pitch = self.model.jnt_qposadr[self.jid_pitch]
        self.q_roll = self.model.jnt_qposadr[self.jid_roll]
        self.q_base_x = self.model.jnt_qposadr[self.jid_base_x]
        self.q_base_y = self.model.jnt_qposadr[self.jid_base_y]
        self.q_base_quat_w = qadr_base_free + 3
        self.q_base_quat_x = qadr_base_free + 4
        self.q_base_quat_y = qadr_base_free + 5
        self.q_base_quat_z = qadr_base_free + 6

        self.v_pitch = self.model.jnt_dofadr[self.jid_pitch]
        self.v_roll = self.model.jnt_dofadr[self.jid_roll]
        self.v_rw = self.model.jnt_dofadr[self.jid_rw]
        self.v_base_x = self.model.jnt_dofadr[self.jid_base_x]
        self.v_base_y = self.model.jnt_dofadr[self.jid_base_y]
        self.v_base_ang_x = dadr_base_free + 3
        self.v_base_ang_y = dadr_base_free + 4
        self.v_base_ang_z = dadr_base_free + 5

        self.aid_rw = aid("wheel_spin")
        self.aid_base_x = aid("base_x_force")
        self.aid_base_y = aid("base_y_force")
        self.stick_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "stick")

    def _enforce_planar_root_attitude(self, do_forward: bool = True) -> None:
        # Keep the free-joint attitude upright to remove unmodeled tumble drift.
        self.data.qpos[self.q_base_quat_w] = 1.0
        self.data.qpos[self.q_base_quat_x] = 0.0
        self.data.qpos[self.q_base_quat_y] = 0.0
        self.data.qpos[self.q_base_quat_z] = 0.0
        self.data.qvel[self.v_base_ang_x] = 0.0
        self.data.qvel[self.v_base_ang_y] = 0.0
        self.data.qvel[self.v_base_ang_z] = 0.0
        if do_forward:
            mujoco.mj_forward(self.model, self.data)

    def _linearize(self, pitch_rad: float | None = None, roll_rad: float | None = None):
        self.data.qpos[:] = self.model.qpos0
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        self.data.qpos[self.q_pitch] = self.PITCH_EQ if pitch_rad is None else float(pitch_rad)
        self.data.qpos[self.q_roll] = self.ROLL_EQ if roll_rad is None else float(roll_rad)
        if self.lock_root_attitude:
            self._enforce_planar_root_attitude(do_forward=False)
        mujoco.mj_forward(self.model, self.data)

        nx = 2 * self.model.nv + self.model.na
        nu = self.model.nu
        A_full = np.zeros((nx, nx))
        B_full = np.zeros((nx, nu))
        mujoco.mjd_transitionFD(self.model, self.data, 1e-6, True, A_full, B_full, None, None)

        # mjd_transitionFD uses tangent-space position coordinates (size nv), not qpos (size nq).
        idx = [
            self.v_pitch,
            self.v_roll,
            self.model.nv + self.v_pitch,
            self.model.nv + self.v_roll,
            self.model.nv + self.v_rw,
            self.v_base_x,
            self.v_base_y,
            self.model.nv + self.v_base_x,
            self.model.nv + self.v_base_y,
        ]
        A = A_full[np.ix_(idx, idx)]
        B = B_full[np.ix_(idx, [self.aid_rw, self.aid_base_x, self.aid_base_y])]
        return A, B

    def _resolve_linearization(self, pitch_rad: float, roll_rad: float) -> tuple[np.ndarray, np.ndarray]:
        key = (round(float(pitch_rad), 8), round(float(roll_rad), 8))
        cached = self._linearization_cache.get(key)
        if cached is not None:
            return cached
        a_lin, b_lin = self._linearize(pitch_rad=float(pitch_rad), roll_rad=float(roll_rad))
        self._linearization_cache[key] = (a_lin, b_lin)
        return a_lin, b_lin

    def _build_measurement_matrix(self, cfg: runtime_cfg.RuntimeConfig) -> np.ndarray:
        rows = [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ]
        if cfg.base_state_from_sensors:
            rows.extend(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                ]
            )
        return np.asarray(rows, dtype=float)

    def _build_measurement_noise_cov(self, cfg: runtime_cfg.RuntimeConfig, wheel_lsb_rad_s: float) -> np.ndarray:
        angle_var = float(cfg.imu_angle_noise_std_rad**2)
        rate_var = float(cfg.imu_rate_noise_std_rad_s**2)
        # Uniform quantization noise variance + additive rate-noise variance.
        wheel_quant_var = float((wheel_lsb_rad_s**2) / 12.0)
        wheel_noise_var = float(cfg.wheel_encoder_rate_noise_std_rad_s**2)
        variances = [angle_var, angle_var, rate_var, rate_var, wheel_quant_var + wheel_noise_var]
        if cfg.base_state_from_sensors:
            variances.extend(
                [
                    float(cfg.base_encoder_pos_noise_std_m**2),
                    float(cfg.base_encoder_pos_noise_std_m**2),
                    float(cfg.base_encoder_vel_noise_std_m_s**2),
                    float(cfg.base_encoder_vel_noise_std_m_s**2),
                ]
            )
        return np.diag(variances)

    def _build_kalman_gain(
        self,
        cfg: runtime_cfg.RuntimeConfig,
        wheel_lsb_rad_s: float,
        C_meas: np.ndarray,
        A_lin: np.ndarray,
    ):
        R_meas = self._build_measurement_noise_cov(cfg, wheel_lsb_rad_s)
        Pk = self._solve_discrete_are_robust(A_lin.T, C_meas.T, self.QN, R_meas, label="Kalman")
        gram = C_meas @ Pk @ C_meas.T + R_meas
        rhs = C_meas @ Pk.T
        return self._solve_linear_robust(gram, rhs, label="Kalman gain").T

    def _build_lqr_gain(self, params: CandidateParams, A_lin: np.ndarray, B_lin: np.ndarray):
        A_aug = np.block([[A_lin, B_lin], [np.zeros((3, 9)), np.eye(3)]])
        B_aug = np.vstack([B_lin, np.eye(3)])

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
        self.data.qpos[:] = self.model.qpos0
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0

        ang = math.radians(config.init_angle_deg)
        if config.init_pitch_deg_override is None:
            self.data.qpos[self.q_pitch] = rng.uniform(-ang, ang)
        else:
            self.data.qpos[self.q_pitch] = float(np.radians(config.init_pitch_deg_override))
        if config.init_roll_deg_override is None:
            self.data.qpos[self.q_roll] = rng.uniform(-ang, ang)
        else:
            self.data.qpos[self.q_roll] = float(np.radians(config.init_roll_deg_override))
        if config.init_base_x_m_override is None:
            self.data.qpos[self.q_base_x] = rng.uniform(-config.init_base_pos_m, config.init_base_pos_m)
        else:
            self.data.qpos[self.q_base_x] = float(config.init_base_x_m_override)
        if config.init_base_y_m_override is None:
            self.data.qpos[self.q_base_y] = rng.uniform(-config.init_base_pos_m, config.init_base_pos_m)
        else:
            self.data.qpos[self.q_base_y] = float(config.init_base_y_m_override)

        if self.lock_root_attitude:
            self._enforce_planar_root_attitude(do_forward=False)
        mujoco.mj_forward(self.model, self.data)

    def _build_measurement(
        self,
        x_true: np.ndarray,
        cfg: runtime_cfg.RuntimeConfig,
        wheel_lsb: float,
        rng: np.random.Generator,
        noise_scale: float = 1.0,
    ):
        pitch = x_true[0] + rng.normal(0.0, noise_scale * cfg.imu_angle_noise_std_rad)
        roll = x_true[1] + rng.normal(0.0, noise_scale * cfg.imu_angle_noise_std_rad)
        pitch_rate = x_true[2] + rng.normal(0.0, noise_scale * cfg.imu_rate_noise_std_rad_s)
        roll_rate = x_true[3] + rng.normal(0.0, noise_scale * cfg.imu_rate_noise_std_rad_s)
        wheel_quant = np.round(x_true[4] / wheel_lsb) * wheel_lsb
        wheel_rate = wheel_quant + rng.normal(0.0, noise_scale * cfg.wheel_encoder_rate_noise_std_rad_s)
        y = [pitch, roll, pitch_rate, roll_rate, wheel_rate]
        if cfg.base_state_from_sensors:
            y.extend(
                [
                    x_true[5] + rng.normal(0.0, noise_scale * cfg.base_encoder_pos_noise_std_m),
                    x_true[6] + rng.normal(0.0, noise_scale * cfg.base_encoder_pos_noise_std_m),
                    x_true[7] + rng.normal(0.0, noise_scale * cfg.base_encoder_vel_noise_std_m_s),
                    x_true[8] + rng.normal(0.0, noise_scale * cfg.base_encoder_vel_noise_std_m_s),
                ]
            )
        return np.asarray(y, dtype=float)

    def simulate_episode(
        self,
        params: CandidateParams,
        episode_seed: int,
        config: EpisodeConfig,
        collect_disturbance_events: bool = False,
        collect_pitch_phase_trace: bool = False,
        collect_rw_du_clip_sign_stats: bool = False,
        collect_filter_error_stats: bool = False,
    ) -> Dict[str, float]:
        rng = np.random.default_rng(episode_seed)
        runtime_cfg_for_episode = self._resolve_runtime_cfg(config)
        self._sync_runtime_tuning(runtime_cfg_for_episode)
        A_base, B_base = self._resolve_linearization(self.LINEARIZE_PITCH_RAD, self.LINEARIZE_ROLL_RAD)
        self._reset_with_initial_state(rng, config)
        a_scale, b_scale, disturbance_scale = self._variant_scales(config.model_variant_id)
        noise_scale, timing_jitter_frac = self._domain_noise_scales(config.domain_profile_id)
        A_run_step = A_base * a_scale
        B_run_step = B_base * b_scale

        if not runtime_cfg_for_episode.hardware_realistic:
            control_steps = 1
        else:
            denom_hz = max(float(runtime_cfg_for_episode.control_hz), 1e-9)
            control_steps = max(1, int(round(1.0 / (self.dt * denom_hz))))
        control_dt = control_steps * self.dt
        A_run, B_run = self._lift_discrete_dynamics(A_run_step, B_run_step, control_steps)
        B_run_pinv = np.linalg.pinv(B_run)
        wheel_ticks = max(int(runtime_cfg_for_episode.wheel_encoder_ticks_per_rev), 1)
        wheel_lsb = (2.0 * np.pi) / (wheel_ticks * control_dt)
        C_meas = self._build_measurement_matrix(runtime_cfg_for_episode)
        L = self._build_kalman_gain(runtime_cfg_for_episode, wheel_lsb, C_meas, A_lin=A_run)

        queue_len = (
            1 if not runtime_cfg_for_episode.hardware_realistic else max(int(runtime_cfg_for_episode.control_delay_steps), 0) + 1
        )
        cmd_queue = deque([np.zeros(3, dtype=float) for _ in range(queue_len)], maxlen=queue_len)

        x_est = np.zeros(9, dtype=float)
        u_applied = np.zeros(3, dtype=float)
        u_eff_applied = np.zeros(3, dtype=float)
        base_int = np.zeros(2, dtype=float)
        base_ref = np.zeros(2, dtype=float)
        base_authority_state = 0.0
        u_base_smooth = np.zeros(2, dtype=float)
        wheel_momentum_bias_int = 0.0
        balance_phase = "recovery"
        recovery_time_s = 0.0
        high_spin_active = False
        dob_hat = np.zeros(3, dtype=float)
        dob_raw = np.zeros(3, dtype=float)
        dob_prev_x_est: np.ndarray | None = None
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
        crash_reason = ""
        crash_pitch_rate_rad_s = np.nan
        crash_roll_rate_rad_s = np.nan
        crash_wheel_rate_rad_s = np.nan
        pitch_recovery_deadline_step: Optional[int] = None
        phase_switch_count = 0
        hold_steps = 0
        wheel_over_budget_count = 0
        wheel_over_hard_count = 0
        high_spin_steps = 0
        disturbance_events: List[Dict[str, float]] = []
        pitch_phase_trace: List[Dict[str, float | str]] = []
        phase_switch_events: List[Dict[str, float | str]] = []
        rw_du_clip_pos = 0
        rw_du_clip_neg = 0
        rw_du_clip_zero = 0
        rw_du_total_samples = 0
        filter_err_pitch_samples: List[float] = []
        filter_err_pitch_ctrl_samples: List[float] = []
        true_pitch_samples: List[float] = []

        family = str(getattr(config, "controller_family", "current"))
        low_spin_robust = (config.stability_profile == "low-spin-robust") and family in ("current", "current_dob", "hybrid_modern")
        use_hold_momentum_bias = family in ("current", "current_dob", "hybrid_modern")
        legacy_family = family in ("legacy_wheel_pid", "legacy_wheel_lqr", "legacy_run_pd")
        paper_family = family == "paper_split_baseline"
        hybrid_family = family == "hybrid_modern"
        mpc_family = family == "baseline_mpc"
        robust_hinf_family = family == "baseline_robust_hinf_like"

        k_du = self._build_lqr_gain(params, A_lin=A_run, B_lin=B_run)
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
        mpc_controller = None
        if mpc_family:
            mpc_q_diag = np.array(
                [
                    runtime_cfg_for_episode.mpc_q_angles,
                    runtime_cfg_for_episode.mpc_q_angles,
                    runtime_cfg_for_episode.mpc_q_rates,
                    runtime_cfg_for_episode.mpc_q_rates,
                    1.0,
                    runtime_cfg_for_episode.mpc_q_position,
                    runtime_cfg_for_episode.mpc_q_position,
                    runtime_cfg_for_episode.mpc_q_rates,
                    runtime_cfg_for_episode.mpc_q_rates,
                ],
                dtype=float,
            )
            mpc_r_diag = np.array(
                [
                    runtime_cfg_for_episode.mpc_r_control,
                    runtime_cfg_for_episode.mpc_r_control,
                    runtime_cfg_for_episode.mpc_r_control,
                ],
                dtype=float,
            )
            mpc_controller = MPCController(
                A=A_run,
                B=B_run,
                horizon=int(runtime_cfg_for_episode.mpc_horizon),
                q_diag=mpc_q_diag,
                r_diag=mpc_r_diag,
                terminal_weight=float(runtime_cfg_for_episode.mpc_terminal_weight),
                u_max=self.MAX_U.copy(),
                com_radius_m=float(runtime_cfg_for_episode.mpc_com_constraint_radius_m),
                angle_max_rad=float(runtime_cfg_for_episode.crash_angle_rad),
                verbose=bool(runtime_cfg_for_episode.mpc_verbose),
            )
        max_du = np.array([params.max_du_rw, params.max_du_bx, params.max_du_by], dtype=float)
        tilt_fail_rad = min(math.radians(config.max_worst_tilt_deg), self.CRASH_ANGLE_RAD)

        for step in range(1, config.steps + 1):
            if self.lock_root_attitude:
                self._enforce_planar_root_attitude()
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
            x_pred = A_run_step @ x_est + B_run_step @ u_eff_applied
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
                y = self._build_measurement(x_true, runtime_cfg_for_episode, wheel_lsb, rng, noise_scale=noise_scale)
                x_est = x_pred + L @ (y - C_meas @ x_pred)
                if not runtime_cfg_for_episode.base_state_from_sensors:
                    # Legacy observation model: anchor unmeasured base states.
                    x_est[5] = x_true[5]
                    x_est[6] = x_true[6]
                    x_est[7] = x_true[7]
                    x_est[8] = x_true[8]

                x_ctrl = x_est.copy()
                x_ctrl[5] -= self.X_REF
                x_ctrl[6] -= self.Y_REF

                if self.base_integrator_enabled:
                    base_int[0] = np.clip(base_int[0] + x_ctrl[5] * control_dt, -self.INT_CLAMP, self.INT_CLAMP)
                    base_int[1] = np.clip(base_int[1] + x_ctrl[6] * control_dt, -self.INT_CLAMP, self.INT_CLAMP)
                else:
                    base_int[:] = 0.0

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
                    if collect_pitch_phase_trace:
                        phase_switch_events.append(
                            {
                                "step": float(step),
                                "time_s": float(step * self.dt),
                                "from_phase": str(prev_phase),
                                "to_phase": str(balance_phase),
                                "pitch_rad": float(x_true[0]),
                                "pitch_deg": float(np.degrees(float(x_true[0]))),
                                "pitch_rate_rad_s": float(x_true[2]),
                            }
                        )
                if balance_phase == "recovery":
                        recovery_time_s += control_dt

                if runtime_cfg_for_episode.dob_enabled:
                    if dob_prev_x_est is None:
                        dob_raw[:] = 0.0
                    else:
                        x_model = A_run @ dob_prev_x_est + B_run @ u_eff_applied
                        state_residual = x_est - x_model
                        dob_raw = B_run_pinv @ state_residual
                        if (dob_raw.shape[0] != 3) or (not np.all(np.isfinite(dob_raw))):
                            dob_hat[:] = 0.0
                            dob_raw[:] = 0.0
                        else:
                            gain = float(max(runtime_cfg_for_episode.dob_gain, 0.0))
                            alpha = float(1.0 - np.exp(-gain * max(control_dt, 0.0)))
                            alpha = float(np.clip(alpha, 0.0, 1.0))
                            dob_hat = (1.0 - alpha) * dob_hat + alpha * dob_raw
                            leak = float(np.clip(runtime_cfg_for_episode.dob_leak_per_s * control_dt, 0.0, 1.0))
                            dob_hat *= (1.0 - leak)
                            dob_hat = np.clip(dob_hat, -runtime_cfg_for_episode.dob_max_abs_u, runtime_cfg_for_episode.dob_max_abs_u)
                            if not np.all(np.isfinite(dob_hat)):
                                dob_hat[:] = 0.0
                                dob_raw[:] = 0.0
                else:
                    dob_hat[:] = 0.0
                    dob_raw[:] = 0.0

                z = np.concatenate([x_ctrl, u_eff_applied])
                du_lqr = -k_du @ z
                u_target_mpc = None
                if mpc_family:
                    if mpc_controller is not None:
                        x_ref_mpc = np.zeros_like(x_ctrl)
                        rate_clip = float(max(runtime_cfg_for_episode.mpc_target_rate_clip_rad_s, 0.0))
                        run_gain = float(max(runtime_cfg_for_episode.mpc_target_rate_gain, 0.0))
                        term_gain = float(max(runtime_cfg_for_episode.mpc_terminal_rate_gain, 0.0))
                        if rate_clip > 0.0:
                            x_ref_mpc[2] = float(np.clip(-run_gain * x_ctrl[0], -rate_clip, rate_clip))
                            x_ref_mpc[3] = float(np.clip(-run_gain * x_ctrl[1], -rate_clip, rate_clip))
                        x_ref_terminal_mpc = x_ref_mpc.copy()
                        if rate_clip > 0.0:
                            x_ref_terminal_mpc[2] = float(np.clip(-term_gain * x_ctrl[0], -rate_clip, rate_clip))
                            x_ref_terminal_mpc[3] = float(np.clip(-term_gain * x_ctrl[1], -rate_clip, rate_clip))
                        u_mpc, mpc_info = mpc_controller.solve(
                            x_ctrl,
                            x_ref=x_ref_mpc,
                            x_ref_terminal=x_ref_terminal_mpc,
                        )
                        if bool(mpc_info.get("success", False)):
                            u_target_mpc = np.asarray(u_mpc, dtype=float)
                    if u_target_mpc is None:
                        # Keep deterministic fallback for rare infeasible/failed solves.
                        u_target_mpc = np.asarray(-k_mpc @ x_ctrl, dtype=float)

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
                    du_rw_cmd = float(u_target_mpc[0] - u_eff_applied[0])
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
                rw_du_limit = float(max_du[0])
                if self.RW_EMERGENCY_DU_ENABLED and balance_phase != "hold":
                    if abs(float(x_est[0])) >= self.RW_EMERGENCY_PITCH_RAD:
                        rw_du_limit *= self.RW_EMERGENCY_DU_SCALE
                if collect_rw_du_clip_sign_stats:
                    rw_du_total_samples += 1
                    if abs(du_rw_cmd) > rw_du_limit:
                        if du_rw_cmd > 0.0:
                            rw_du_clip_pos += 1
                        elif du_rw_cmd < 0.0:
                            rw_du_clip_neg += 1
                        else:
                            rw_du_clip_zero += 1
                du_rw = float(np.clip(du_rw_cmd, -rw_du_limit, rw_du_limit))
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
                if balance_phase == "hold" and self.HOLD_BASE_X_CENTERING_GAIN > 0.0:
                    hold_center_err_x = float(np.clip(x_est[5], -self.BASE_CENTERING_POS_CLIP_M, self.BASE_CENTERING_POS_CLIP_M))
                    base_target_x += float(-self.HOLD_BASE_X_CENTERING_GAIN * hold_center_err_x)
                if self.base_integrator_enabled:
                    base_target_x += -params.ki_base * base_int[0]
                    base_target_y += -params.ki_base * base_int[1]
                if legacy_family:
                    base_target_x = 0.0
                    base_target_y = 0.0
                if mpc_family:
                    base_target_x = float(u_target_mpc[1])
                    base_target_y = float(u_target_mpc[2])
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
                if use_hold_momentum_bias and (not mpc_family):
                    speed_ref = min(
                        self.WHEEL_SPIN_BUDGET_FRAC * self.MAX_WHEEL_SPEED_RAD_S,
                        self.WHEEL_SPIN_BUDGET_ABS_RAD_S,
                    )
                    speed_ref = max(0.35 * speed_ref, 1e-3)
                    if balance_phase == "hold":
                        wheel_norm = float(np.clip(float(x_est[4]) / speed_ref, -2.0, 2.0))
                        wheel_momentum_bias_int += wheel_norm * control_dt
                        wheel_momentum_bias_int = float(np.clip(wheel_momentum_bias_int, -2.0, 2.0))
                    else:
                        leak = float(np.clip(0.65 * control_dt, 0.0, 1.0))
                        wheel_momentum_bias_int = float((1.0 - leak) * wheel_momentum_bias_int)
                    hold_bias_term = float(
                        np.clip(-0.32 * self.WHEEL_TO_BASE_BIAS_GAIN * wheel_momentum_bias_int, -0.25, 0.25)
                    )
                    base_target_x += hold_bias_term

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

                du_hits += int((abs(du_rw_cmd) > rw_du_limit) or np.any(np.abs(du_base_cmd) > base_du_limit))
                sat_hits += int((abs(u_rw_unc) > self.MAX_U[0]) or np.any(np.abs(u_base_unc) > self.MAX_U[1:]))
                u_cmd = np.array([u_rw_cmd, u_base_cmd[0], u_base_cmd[1]], dtype=float)
                if runtime_cfg_for_episode.dob_enabled:
                    u_cmd_dob = u_cmd - dob_hat
                    sat_hits += int(np.any(np.abs(u_cmd_dob) > self.MAX_U))
                    u_cmd = np.clip(u_cmd_dob, -self.MAX_U, self.MAX_U)

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
                dob_prev_x_est = x_est.copy()
            if collect_filter_error_stats:
                filter_err_pitch_samples.append(float(x_est[0] - x_true[0]))
                true_pitch_samples.append(float(x_true[0]))
                if step % effective_control_steps == 0:
                    filter_err_pitch_ctrl_samples.append(float(x_est[0] - x_true[0]))
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
                if collect_disturbance_events:
                    disturbance_events.append(
                        {
                            "step": float(step),
                            "wheel_speed": float(self.data.qvel[self.v_rw]),
                            "base_x_pos": float(self.data.qpos[self.q_base_x]),
                            "pitch": float(self.data.qpos[self.q_pitch]),
                            "pitch_rate": float(self.data.qvel[self.v_pitch]),
                            "force_x": float(force[0]),
                            "force_y": float(force[1]),
                            "force_z": float(force[2]),
                        }
                    )
                self.data.xfrc_applied[self.stick_body_id, :3] = force
            else:
                self.data.xfrc_applied[self.stick_body_id, :3] = 0.0

            mujoco.mj_step(self.model, self.data)
            if self.lock_root_attitude:
                self._enforce_planar_root_attitude()

            pitch = float(self.data.qpos[self.q_pitch])
            roll = float(self.data.qpos[self.q_roll])
            bx = float(self.data.qpos[self.q_base_x])
            by = float(self.data.qpos[self.q_base_y])
            if collect_pitch_phase_trace:
                pitch_phase_trace.append(
                    {
                        "step": float(step),
                        "time_s": float(step * self.dt),
                        "pitch_rad": pitch,
                        "pitch_deg": float(np.degrees(pitch)),
                        "pitch_rate_rad_s": float(self.data.qvel[self.v_pitch]),
                        "phase": str(balance_phase),
                    }
                )

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

            pitch_rate_now = float(self.data.qvel[self.v_pitch])
            roll_rate_now = float(self.data.qvel[self.v_roll])
            wheel_rate_now = float(self.data.qvel[self.v_rw])
            pitch_failed = abs(pitch) >= tilt_fail_rad
            roll_failed = abs(roll) >= tilt_fail_rad

            pitch_crash_confirmed = False
            pitch_crash_reason = "pitch_tilt"
            if pitch_failed:
                pitch_diverging = (abs(pitch_rate_now) > 1e-9) and (np.sign(pitch_rate_now) == np.sign(pitch))
                if (not self.CRASH_DIVERGENCE_GATE_ENABLED) or pitch_diverging:
                    pitch_crash_confirmed = True
                    pitch_crash_reason = "pitch_tilt_diverging" if self.CRASH_DIVERGENCE_GATE_ENABLED else "pitch_tilt"
                    pitch_recovery_deadline_step = None
                elif self.CRASH_RECOVERY_WINDOW_STEPS <= 0:
                    pitch_crash_confirmed = True
                    pitch_crash_reason = "pitch_tilt_recovery_disabled"
                    pitch_recovery_deadline_step = None
                else:
                    if pitch_recovery_deadline_step is None:
                        pitch_recovery_deadline_step = step + self.CRASH_RECOVERY_WINDOW_STEPS
                    if step >= pitch_recovery_deadline_step:
                        pitch_crash_confirmed = True
                        pitch_crash_reason = "pitch_tilt_recovery_timeout"
                        pitch_recovery_deadline_step = None
            else:
                pitch_recovery_deadline_step = None

            if pitch_crash_confirmed or roll_failed:
                crash_step = step
                crash_reason = pitch_crash_reason if pitch_crash_confirmed else "roll_tilt"
                crash_pitch_rate_rad_s = pitch_rate_now
                crash_roll_rate_rad_s = roll_rate_now
                crash_wheel_rate_rad_s = wheel_rate_now
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
        crash_preceding_step = None
        if crash_step is not None and disturbance_events:
            crash_preceding_candidates = [int(ev["step"]) for ev in disturbance_events if int(ev["step"]) <= int(crash_step)]
            if crash_preceding_candidates:
                crash_preceding_step = max(crash_preceding_candidates)
        survivor_reference_step = None
        if crash_step is None and disturbance_events:
            survivor_reference_step = max(int(ev["step"]) for ev in disturbance_events)
        for ev in disturbance_events:
            ev_step = int(ev["step"])
            ev["episode_seed"] = float(episode_seed)
            ev["episode_crashed"] = 1.0 if crash_step is not None else 0.0
            ev["crash_step"] = float(crash_step) if crash_step is not None else np.nan
            ev["is_crash_preceding"] = 1.0 if (crash_preceding_step is not None and ev_step == crash_preceding_step) else 0.0
            ev["is_survivor_reference"] = (
                1.0 if (survivor_reference_step is not None and ev_step == survivor_reference_step) else 0.0
            )

        result = {
            "survived": 1.0 if crash_step is None else 0.0,
            "crash_count": 0.0 if crash_step is None else 1.0,
            "crash_step": float(crash_step) if crash_step is not None else np.nan,
            "crash_reason": crash_reason,
            "crash_pitch_rate_rad_s": crash_pitch_rate_rad_s,
            "crash_roll_rate_rad_s": crash_roll_rate_rad_s,
            "crash_wheel_rate_rad_s": crash_wheel_rate_rad_s,
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
        if collect_disturbance_events:
            result["disturbance_events"] = disturbance_events
        if collect_pitch_phase_trace:
            result["pitch_phase_trace"] = pitch_phase_trace
            result["phase_switch_events"] = phase_switch_events
        if collect_rw_du_clip_sign_stats:
            total_clips = rw_du_clip_pos + rw_du_clip_neg + rw_du_clip_zero
            result["rw_du_clip_pos"] = float(rw_du_clip_pos)
            result["rw_du_clip_neg"] = float(rw_du_clip_neg)
            result["rw_du_clip_zero"] = float(rw_du_clip_zero)
            result["rw_du_clip_total"] = float(total_clips)
            result["rw_du_clip_rate"] = float(total_clips / max(rw_du_total_samples, 1))
            result["rw_du_clip_pos_frac"] = float(rw_du_clip_pos / max(total_clips, 1))
            result["rw_du_clip_neg_frac"] = float(rw_du_clip_neg / max(total_clips, 1))
        if collect_filter_error_stats:
            arr = np.asarray(filter_err_pitch_samples, dtype=float)
            carr = np.asarray(filter_err_pitch_ctrl_samples, dtype=float)
            tarr = np.asarray(true_pitch_samples, dtype=float)
            if arr.size > 0:
                result["filter_err_pitch_mean_rad"] = float(np.mean(arr))
                result["filter_err_pitch_mean_deg"] = float(np.degrees(np.mean(arr)))
                result["filter_err_pitch_std_deg"] = float(np.degrees(np.std(arr)))
                result["filter_err_pitch_median_deg"] = float(np.degrees(np.median(arr)))
                result["filter_err_pitch_p05_deg"] = float(np.degrees(np.percentile(arr, 5)))
                result["filter_err_pitch_p95_deg"] = float(np.degrees(np.percentile(arr, 95)))
                result["filter_err_pitch_neg_frac"] = float(np.mean(arr < 0.0))
                result["filter_err_pitch_pos_frac"] = float(np.mean(arr > 0.0))
                result["filter_err_pitch_samples"] = float(arr.size)
                if tarr.size == arr.size and tarr.size > 2:
                    slope = float(np.polyfit(tarr, arr, 1)[0])
                    result["filter_err_vs_true_pitch_slope_deg_per_deg"] = float(slope)
                    # High-angle bucket: top quartile of true pitch magnitude.
                    q = float(np.quantile(np.abs(tarr), 0.75))
                    hi_mask = np.abs(tarr) >= q
                    if np.any(hi_mask):
                        result["filter_err_high_angle_mean_deg"] = float(np.degrees(np.mean(arr[hi_mask])))
                    lo_mask = np.abs(tarr) < q
                    if np.any(lo_mask):
                        result["filter_err_low_angle_mean_deg"] = float(np.degrees(np.mean(arr[lo_mask])))
            if carr.size > 0:
                result["filter_err_pitch_ctrl_mean_deg"] = float(np.degrees(np.mean(carr)))
                result["filter_err_pitch_ctrl_neg_frac"] = float(np.mean(carr < 0.0))
                result["filter_err_pitch_ctrl_pos_frac"] = float(np.mean(carr > 0.0))
                result["filter_err_pitch_ctrl_samples"] = float(carr.size)
        return result

    def evaluate_candidate(
        self,
        params: CandidateParams,
        episode_seeds: List[int],
        config: EpisodeConfig,
    ) -> Dict[str, float]:
        runtime_cfg_for_episode = self._resolve_runtime_cfg(config)
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
        crash_pitch_rate_vals = np.asarray([m.get("crash_pitch_rate_rad_s", np.nan) for m in per_episode], dtype=float)
        crash_roll_rate_vals = np.asarray([m.get("crash_roll_rate_rad_s", np.nan) for m in per_episode], dtype=float)
        crash_wheel_rate_vals = np.asarray([m.get("crash_wheel_rate_rad_s", np.nan) for m in per_episode], dtype=float)
        crash_pitch_rate_vals = crash_pitch_rate_vals[np.isfinite(crash_pitch_rate_vals)]
        crash_roll_rate_vals = crash_roll_rate_vals[np.isfinite(crash_roll_rate_vals)]
        crash_wheel_rate_vals = crash_wheel_rate_vals[np.isfinite(crash_wheel_rate_vals)]
        crash_pitch_rate_mean = float(np.mean(crash_pitch_rate_vals)) if crash_pitch_rate_vals.size else np.nan
        crash_pitch_rate_abs_mean = float(np.mean(np.abs(crash_pitch_rate_vals))) if crash_pitch_rate_vals.size else np.nan
        crash_pitch_rate_pos_frac = float(np.mean(crash_pitch_rate_vals > 0.0)) if crash_pitch_rate_vals.size else np.nan
        crash_pitch_rate_neg_frac = float(np.mean(crash_pitch_rate_vals < 0.0)) if crash_pitch_rate_vals.size else np.nan
        crash_roll_rate_mean = float(np.mean(crash_roll_rate_vals)) if crash_roll_rate_vals.size else np.nan
        crash_wheel_rate_abs_mean = (
            float(np.mean(np.abs(crash_wheel_rate_vals))) if crash_wheel_rate_vals.size else np.nan
        )

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
            "crash_pitch_rate_mean_rad_s": crash_pitch_rate_mean,
            "crash_pitch_rate_abs_mean_rad_s": crash_pitch_rate_abs_mean,
            "crash_pitch_rate_pos_frac": crash_pitch_rate_pos_frac,
            "crash_pitch_rate_neg_frac": crash_pitch_rate_neg_frac,
            "crash_roll_rate_mean_rad_s": crash_roll_rate_mean,
            "crash_wheel_rate_abs_mean_rad_s": crash_wheel_rate_abs_mean,
            "failure_reason": failure_reason,
            "confidence_tier": confidence_tier,
            "hardware_realistic": bool(runtime_cfg_for_episode.hardware_realistic),
            "control_hz": float(runtime_cfg_for_episode.control_hz),
            "control_delay_steps": int(runtime_cfg_for_episode.control_delay_steps),
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
            "crash_pitch_rate_mean_rad_s": np.nan,
            "crash_pitch_rate_abs_mean_rad_s": np.nan,
            "crash_pitch_rate_pos_frac": np.nan,
            "crash_pitch_rate_neg_frac": np.nan,
            "crash_roll_rate_mean_rad_s": np.nan,
            "crash_wheel_rate_abs_mean_rad_s": np.nan,
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
