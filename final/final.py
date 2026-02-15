import argparse
import csv
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from scipy.linalg import solve_discrete_are

"""
Teaching-oriented MuJoCo balancing controller.

Reader Guide
------------
Big picture:
- `final.xml` defines rigid bodies, joints, actuator names/ranges, and scene geometry.
- `final.py` reads that model, linearizes it around upright equilibrium, then runs:
  1) state estimation (Kalman update from noisy sensors),
  2) control (delta-u LQR + safety shaping),
  3) actuator limits and MuJoCo stepping in a viewer loop.

How XML affects Python:
- Joint/actuator/body names in XML are hard-linked in `lookup_model_ids`.
- XML actuator `ctrlrange` is checked against runtime command limits to prevent mismatch.
- Dynamics and geometry in XML determine the A/B matrices obtained by finite-difference
  linearization (`mjd_transitionFD`), so geometry/inertia edits directly change control behavior.

How Python affects XML:
- Python does not rewrite XML at runtime; it only loads and validates it.
- CLI/runtime config can alter simulation behavior (noise, delay, limits), but not mesh/layout.

Run on your laptop/PC:
- Install deps: `pip install -r requirements.txt`
- Start viewer: `python final/final.py --mode smooth`
- Try robust profile: `python final/final.py --mode robust --stability-profile low-spin-robust`
- Hardware-like simulation: `python final/final.py --real-hardware`
"""


@dataclass(frozen=True)
class RuntimeConfig:
    controller_family: str
    log_control_terms: bool
    control_terms_csv: str | None
    trace_events_csv: str | None
    preset: str
    stability_profile: str
    stable_demo_profile: bool
    low_spin_robust_profile: bool
    smooth_viewer: bool
    real_hardware_profile: bool
    hardware_safe: bool
    easy_mode: bool
    stop_on_crash: bool
    wheel_only: bool
    wheel_only_forced: bool
    allow_base_motion_requested: bool
    real_hardware_base_unlocked: bool
    allow_base_motion: bool
    seed: int
    hardware_realistic: bool
    control_hz: float
    control_delay_steps: int
    wheel_encoder_ticks_per_rev: int
    imu_angle_noise_std_rad: float
    imu_rate_noise_std_rad_s: float
    wheel_encoder_rate_noise_std_rad_s: float
    base_encoder_pos_noise_std_m: float
    base_encoder_vel_noise_std_m_s: float
    base_state_from_sensors: bool
    max_u: np.ndarray
    max_du: np.ndarray
    disturbance_magnitude: float
    disturbance_interval: int
    qx: np.ndarray
    qu: np.ndarray
    r_du: np.ndarray
    ki_base: float
    base_integrator_enabled: bool
    u_bleed: float
    crash_angle_rad: float
    x_ref: float
    y_ref: float
    int_clamp: float
    upright_angle_thresh: float
    upright_vel_thresh: float
    upright_pos_thresh: float
    max_wheel_speed_rad_s: float
    max_pitch_roll_rate_rad_s: float
    max_base_speed_m_s: float
    wheel_torque_derate_start: float
    wheel_momentum_thresh_frac: float
    wheel_momentum_k: float
    wheel_momentum_upright_k: float
    hold_enter_angle_rad: float
    hold_exit_angle_rad: float
    hold_enter_rate_rad_s: float
    hold_exit_rate_rad_s: float
    wheel_spin_budget_frac: float
    wheel_spin_hard_frac: float
    wheel_spin_budget_abs_rad_s: float
    wheel_spin_hard_abs_rad_s: float
    high_spin_exit_frac: float
    high_spin_counter_min_frac: float
    high_spin_base_authority_min: float
    wheel_to_base_bias_gain: float
    recovery_wheel_despin_scale: float
    hold_wheel_despin_scale: float
    upright_blend_rise: float
    upright_blend_fall: float
    base_torque_derate_start: float
    wheel_torque_limit_nm: float
    enforce_wheel_motor_limit: bool
    wheel_motor_kv_rpm_per_v: float
    wheel_motor_resistance_ohm: float
    wheel_current_limit_a: float
    bus_voltage_v: float
    wheel_gear_ratio: float
    drive_efficiency: float
    base_force_soft_limit: float
    base_damping_gain: float
    base_centering_gain: float
    base_tilt_deadband_rad: float
    base_tilt_full_authority_rad: float
    base_command_gain: float
    base_centering_pos_clip_m: float
    base_speed_soft_limit_frac: float
    base_hold_radius_m: float
    base_ref_follow_rate_hz: float
    base_ref_recenter_rate_hz: float
    base_authority_rate_per_s: float
    base_command_lpf_hz: float
    upright_base_du_scale: float
    base_pitch_kp: float
    base_pitch_kd: float
    base_roll_kp: float
    base_roll_kd: float
    wheel_only_pitch_kp: float
    wheel_only_pitch_kd: float
    wheel_only_pitch_ki: float
    wheel_only_wheel_rate_kd: float
    wheel_only_max_u: float
    wheel_only_max_du: float
    wheel_only_int_clamp: float


@dataclass(frozen=True)
class ModelIds:
    q_pitch: int
    q_roll: int
    q_base_x: int
    q_base_y: int
    v_pitch: int
    v_roll: int
    v_rw: int
    v_base_x: int
    v_base_y: int
    aid_rw: int
    aid_base_x: int
    aid_base_y: int
    base_x_body_id: int
    base_y_body_id: int
    stick_body_id: int


def lookup_model_ids(model: mujoco.MjModel) -> ModelIds:
    def jid(name):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)

    def aid(name):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

    jid_pitch = jid("stick_pitch")
    jid_roll = jid("stick_roll")
    jid_rw = jid("wheel_spin")
    jid_base_x = jid("base_x_slide")
    jid_base_y = jid("base_y_slide")

    return ModelIds(
        q_pitch=model.jnt_qposadr[jid_pitch],
        q_roll=model.jnt_qposadr[jid_roll],
        q_base_x=model.jnt_qposadr[jid_base_x],
        q_base_y=model.jnt_qposadr[jid_base_y],
        v_pitch=model.jnt_dofadr[jid_pitch],
        v_roll=model.jnt_dofadr[jid_roll],
        v_rw=model.jnt_dofadr[jid_rw],
        v_base_x=model.jnt_dofadr[jid_base_x],
        v_base_y=model.jnt_dofadr[jid_base_y],
        aid_rw=aid("wheel_spin"),
        aid_base_x=aid("base_x_force"),
        aid_base_y=aid("base_y_force"),
        base_x_body_id=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_x"),
        base_y_body_id=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_y"),
        stick_body_id=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "stick"),
    )


def enforce_wheel_only_constraints(model, data, ids: ModelIds):
    """Pin base translation + roll so wheel-only mode stays single-axis."""
    data.qpos[ids.q_base_x] = 0.0
    data.qpos[ids.q_base_y] = 0.0
    data.qpos[ids.q_roll] = 0.0
    data.qvel[ids.v_base_x] = 0.0
    data.qvel[ids.v_base_y] = 0.0
    data.qvel[ids.v_roll] = 0.0
    mujoco.mj_forward(model, data)


def get_true_state(data, ids: ModelIds) -> np.ndarray:
    """State vector used by estimator/controller."""
    return np.array(
        [
            data.qpos[ids.q_pitch],
            data.qpos[ids.q_roll],
            data.qvel[ids.v_pitch],
            data.qvel[ids.v_roll],
            data.qvel[ids.v_rw],
            data.qpos[ids.q_base_x],
            data.qpos[ids.q_base_y],
            data.qvel[ids.v_base_x],
            data.qvel[ids.v_base_y],
        ],
        dtype=float,
    )


def reset_controller_buffers(nx: int, nu: int, queue_len: int):
    x_est = np.zeros(nx)
    u_applied = np.zeros(nu)
    u_eff_applied = np.zeros(nu)
    base_int = np.zeros(2)
    wheel_pitch_int = 0.0
    base_ref = np.zeros(2)
    base_authority_state = 0.0
    u_base_smooth = np.zeros(2)
    balance_phase = "recovery"
    recovery_time_s = 0.0
    high_spin_active = False
    cmd_queue = deque([np.zeros(nu, dtype=float) for _ in range(queue_len)], maxlen=queue_len)
    return (
        x_est,
        u_applied,
        u_eff_applied,
        base_int,
        wheel_pitch_int,
        base_ref,
        base_authority_state,
        u_base_smooth,
        balance_phase,
        recovery_time_s,
        high_spin_active,
        cmd_queue,
    )


def wheel_command_with_limits(cfg: RuntimeConfig, wheel_speed: float, wheel_cmd_requested: float) -> float:
    """Runtime wheel torque clamp: motor current/voltage + speed derating."""
    wheel_speed_abs = abs(wheel_speed)
    kv_rad_per_s_per_v = cfg.wheel_motor_kv_rpm_per_v * (2.0 * np.pi / 60.0)
    ke_v_per_rad_s = 1.0 / max(kv_rad_per_s_per_v, 1e-9)
    motor_speed = wheel_speed_abs * cfg.wheel_gear_ratio
    v_eff = cfg.bus_voltage_v * cfg.drive_efficiency
    back_emf_v = ke_v_per_rad_s * motor_speed
    headroom_v = max(v_eff - back_emf_v, 0.0)
    i_voltage_limited = headroom_v / max(cfg.wheel_motor_resistance_ohm, 1e-9)
    i_available = min(cfg.wheel_current_limit_a, i_voltage_limited)
    wheel_dynamic_limit = ke_v_per_rad_s * i_available * cfg.wheel_gear_ratio * cfg.drive_efficiency

    if cfg.enforce_wheel_motor_limit:
        wheel_limit = min(cfg.wheel_torque_limit_nm, wheel_dynamic_limit)
    else:
        wheel_limit = cfg.max_u[0]

    wheel_derate_start_speed = cfg.wheel_torque_derate_start * cfg.max_wheel_speed_rad_s
    if wheel_speed_abs > wheel_derate_start_speed:
        span = max(cfg.max_wheel_speed_rad_s - wheel_derate_start_speed, 1e-6)
        wheel_scale = max(0.0, 1.0 - (wheel_speed_abs - wheel_derate_start_speed) / span)
        wheel_limit *= wheel_scale

    wheel_cmd = float(np.clip(wheel_cmd_requested, -wheel_limit, wheel_limit))
    hard_speed = min(cfg.wheel_spin_hard_frac * cfg.max_wheel_speed_rad_s, cfg.wheel_spin_hard_abs_rad_s)
    if wheel_speed_abs >= hard_speed and np.sign(wheel_cmd) == np.sign(wheel_speed):
        # Emergency same-direction suppression before absolute speed limit.
        wheel_cmd *= 0.03
    if wheel_speed_abs >= hard_speed and abs(wheel_speed) > 1e-9:
        # Actuator-level desaturation floor: enforce some opposite torque near hard speed.
        over_hard = float(np.clip((wheel_speed_abs - hard_speed) / max(hard_speed, 1e-6), 0.0, 1.0))
        min_counter = (0.55 + 0.45 * over_hard) * cfg.high_spin_counter_min_frac * wheel_limit
        if abs(wheel_cmd) < 1e-9:
            wheel_cmd = -np.sign(wheel_speed) * min_counter
        elif np.sign(wheel_cmd) != np.sign(wheel_speed):
            wheel_cmd = float(np.sign(wheel_cmd) * max(abs(wheel_cmd), min_counter))
    if wheel_speed_abs >= cfg.max_wheel_speed_rad_s and np.sign(wheel_cmd) == np.sign(wheel_speed):
        wheel_cmd = 0.0
    return wheel_cmd


def base_commands_with_limits(
    cfg: RuntimeConfig,
    base_x_speed: float,
    base_y_speed: float,
    base_x: float,
    base_y: float,
    base_x_request: float,
    base_y_request: float,
):
    """Runtime base force clamp: speed derating + anti-runaway + force clamp."""
    base_derate_start = cfg.base_torque_derate_start * cfg.max_base_speed_m_s
    bx_scale = 1.0
    by_scale = 1.0

    if abs(base_x_speed) > base_derate_start:
        base_margin = max(cfg.max_base_speed_m_s - base_derate_start, 1e-6)
        bx_scale = max(0.0, 1.0 - (abs(base_x_speed) - base_derate_start) / base_margin)
    if abs(base_y_speed) > base_derate_start:
        base_margin = max(cfg.max_base_speed_m_s - base_derate_start, 1e-6)
        by_scale = max(0.0, 1.0 - (abs(base_y_speed) - base_derate_start) / base_margin)

    base_x_cmd = float(base_x_request * bx_scale)
    base_y_cmd = float(base_y_request * by_scale)

    soft_speed = cfg.base_speed_soft_limit_frac * cfg.max_base_speed_m_s
    if abs(base_x_speed) > soft_speed and np.sign(base_x_cmd) == np.sign(base_x_speed):
        span = max(cfg.max_base_speed_m_s - soft_speed, 1e-6)
        base_x_cmd *= max(0.0, 1.0 - (abs(base_x_speed) - soft_speed) / span)
    if abs(base_y_speed) > soft_speed and np.sign(base_y_cmd) == np.sign(base_y_speed):
        span = max(cfg.max_base_speed_m_s - soft_speed, 1e-6)
        base_y_cmd *= max(0.0, 1.0 - (abs(base_y_speed) - soft_speed) / span)

    if abs(base_x) > cfg.base_hold_radius_m and np.sign(base_x_cmd) == np.sign(base_x):
        base_x_cmd *= 0.4
    if abs(base_y) > cfg.base_hold_radius_m and np.sign(base_y_cmd) == np.sign(base_y):
        base_y_cmd *= 0.4

    base_x_cmd = float(np.clip(base_x_cmd, -cfg.base_force_soft_limit, cfg.base_force_soft_limit))
    base_y_cmd = float(np.clip(base_y_cmd, -cfg.base_force_soft_limit, cfg.base_force_soft_limit))
    return base_x_cmd, base_y_cmd


def parse_args():
    parser = argparse.ArgumentParser(description="MuJoCo wheel-on-stick viewer controller.")
    parser.add_argument(
        "--controller-family",
        choices=[
            "current",
            "hybrid_modern",
            "paper_split_baseline",
        ],
        default="current",
        help="Controller family selector. 'current' preserves existing behavior.",
    )
    parser.add_argument(
        "--log-control-terms",
        action="store_true",
        help="Log per-update interpretable control terms (headless-friendly CSV).",
    )
    parser.add_argument(
        "--control-terms-csv",
        type=str,
        default=None,
        help="Optional CSV output path for --log-control-terms.",
    )
    parser.add_argument(
        "--trace-events-csv",
        type=str,
        default=None,
        help="Optional runtime trace/event CSV path for replay alignment.",
    )
    parser.add_argument(
        "--preset",
        choices=["default", "stable-demo"],
        default="default",
        help="Controller tuning preset. 'stable-demo' prioritizes smoothness and anti-runaway behavior.",
    )
    parser.add_argument(
        "--stability-profile",
        choices=["default", "low-spin-robust"],
        default="default",
        help="Optional stability profile. 'low-spin-robust' adds hysteresis and wheel-spin budget control.",
    )
    parser.add_argument("--mode", choices=["smooth", "robust"], default="smooth")
    parser.add_argument(
        "--real-hardware",
        action="store_true",
        help="Extra-conservative bring-up profile for physical hardware (forces strict limits and stop-on-crash).",
    )
    parser.add_argument("--hardware-safe", action="store_true", help="Use conservative real-hardware startup limits.")
    parser.add_argument("--easy-mode", action="store_true")
    parser.add_argument("--stop-on-crash", action="store_true")
    parser.add_argument(
        "--wheel-only",
        action="store_true",
        help="Disable base x/y actuation. Keeps only reaction-wheel control.",
    )
    parser.add_argument(
        "--allow-base-motion",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable base x/y actuation (enabled by default). Use --no-allow-base-motion to force wheel-only.",
    )
    parser.add_argument(
        "--unlock-base",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow base x/y actuation in --real-hardware mode (enabled by default).",
    )
    parser.add_argument(
        "--enable-base-integrator",
        action="store_true",
        help="Enable base x/y integral action. Disabled by default to avoid drift with unobserved base states.",
    )
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--initial-y-tilt-deg",
        type=float,
        default=0.0,
        help="Initial stick tilt in degrees in Y direction (implemented as roll for base_y response).",
    )
    parser.add_argument("--crash-angle-deg", type=float, default=25.0)
    parser.add_argument("--disturbance-mag", type=float, default=None)
    parser.add_argument("--disturbance-interval", type=int, default=None)
    parser.add_argument("--control-hz", type=float, default=250.0)
    parser.add_argument("--control-delay-steps", type=int, default=1)
    parser.add_argument("--wheel-encoder-ticks", type=int, default=2048)
    parser.add_argument("--imu-angle-noise-deg", type=float, default=0.25)
    parser.add_argument("--imu-rate-noise", type=float, default=0.02)
    parser.add_argument("--wheel-rate-noise", type=float, default=0.01)
    parser.add_argument(
        "--base-pos-noise",
        type=float,
        default=0.0015,
        help="Base position sensor noise std-dev (m).",
    )
    parser.add_argument(
        "--base-vel-noise",
        type=float,
        default=0.03,
        help="Base velocity estimate noise std-dev (m/s).",
    )
    parser.add_argument(
        "--planar-perturb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep mouse perturbation forces in the X/Y plane by zeroing Z-force (enabled by default).",
    )
    parser.add_argument(
        "--drag-assist",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Temporarily reduce controller effort while mouse perturbation is active (enabled by default).",
    )
    parser.add_argument(
        "--push-body",
        choices=["stick", "base_y", "base_x"],
        default="stick",
        help="Body used for scripted push force.",
    )
    parser.add_argument("--push-x", type=float, default=0.0, help="Scripted push force in X (N).")
    parser.add_argument("--push-y", type=float, default=0.0, help="Scripted push force in Y (N).")
    parser.add_argument("--push-start-s", type=float, default=1.0, help="Scripted push start time (s).")
    parser.add_argument("--push-duration-s", type=float, default=0.0, help="Scripted push duration (s).")
    parser.add_argument("--legacy-model", action="store_true", help="Disable hardware-realistic timing/noise model.")
    parser.add_argument(
        "--max-wheel-speed",
        type=float,
        default=None,
        help="Reaction wheel speed limit (rad/s). If omitted, derived from motor KV and bus voltage.",
    )
    parser.add_argument("--max-tilt-rate", type=float, default=16.0, help="Pitch/roll rate limit (rad/s).")
    parser.add_argument("--max-base-speed", type=float, default=2.5, help="Base axis speed limit (m/s).")
    parser.add_argument(
        "--wheel-derate-start",
        type=float,
        default=0.75,
        help="Start torque derating at this fraction of wheel speed limit.",
    )
    parser.add_argument(
        "--base-derate-start",
        type=float,
        default=0.70,
        help="Start base motor torque derating at this fraction of base speed limit.",
    )
    parser.add_argument(
        "--wheel-torque-limit",
        type=float,
        default=None,
        help="Max wheel torque (Nm). If omitted, derived from motor constants and current limit.",
    )
    parser.add_argument("--wheel-kv-rpm-per-v", type=float, default=900.0, help="Wheel motor KV (RPM/V).")
    parser.add_argument(
        "--wheel-resistance-ohm", type=float, default=0.22, help="Wheel motor phase-to-phase resistance (ohm)."
    )
    parser.add_argument("--wheel-current-limit-a", type=float, default=25.0, help="Wheel current limit (A).")
    parser.add_argument("--bus-voltage-v", type=float, default=24.0, help="Battery/driver bus voltage (V).")
    parser.add_argument("--wheel-gear-ratio", type=float, default=1.0, help="Motor-to-wheel gear ratio.")
    parser.add_argument(
        "--drive-efficiency", type=float, default=0.90, help="Electrical+mechanical drive efficiency (0..1)."
    )
    return parser.parse_args()


def build_config(args) -> RuntimeConfig:
    preset = str(getattr(args, "preset", "default"))
    stability_profile = str(getattr(args, "stability_profile", "default"))
    stable_demo_profile = preset == "stable-demo"
    low_spin_robust_profile = stability_profile == "low-spin-robust"
    smooth_viewer = args.mode == "smooth"
    real_hardware_profile = bool(getattr(args, "real_hardware", False))
    hardware_safe = bool(getattr(args, "hardware_safe", False)) or real_hardware_profile

    kv_rpm_per_v = max(float(args.wheel_kv_rpm_per_v), 1e-6)
    kv_rad_per_s_per_v = kv_rpm_per_v * (2.0 * np.pi / 60.0)
    ke_v_per_rad_s = 1.0 / kv_rad_per_s_per_v
    kt_nm_per_a = ke_v_per_rad_s
    r_ohm = max(float(args.wheel_resistance_ohm), 1e-6)
    i_lim = max(float(args.wheel_current_limit_a), 0.0)
    v_bus = max(float(args.bus_voltage_v), 0.0)
    gear = max(float(args.wheel_gear_ratio), 1e-6)
    eta = float(np.clip(args.drive_efficiency, 0.05, 1.0))
    v_eff = v_bus * eta

    derived_wheel_speed = (kv_rad_per_s_per_v * v_eff) / gear
    derived_wheel_torque = kt_nm_per_a * i_lim * gear * eta

    max_wheel_speed_rad_s = (
        float(args.max_wheel_speed) if args.max_wheel_speed is not None else float(max(derived_wheel_speed, 1e-3))
    )
    wheel_torque_limit_nm = (
        float(args.wheel_torque_limit) if args.wheel_torque_limit is not None else float(max(derived_wheel_torque, 1e-4))
    )

    if smooth_viewer:
        qx = np.diag(
            [
                95.0 * 1.8881272496398052,
                75.0 * 1.8881272496398052,
                40.0 * 4.365896739242128,
                28.0 * 4.365896739242128,
                0.8 * 4.67293811764239,
                90.0 * 6.5,
                90.0 * 6.5,
                170.0 * 2.4172709641758128,
                170.0 * 2.4172709641758128,
            ]
        )
        qu = np.diag([5e-3 * 3.867780939456718, 0.35 * 3.867780939456718, 0.35 * 3.867780939456718])
        r_du = np.diag([0.1378734731394442, 1.9942819169979045, 5.076875253951513])
        max_u = np.array([80.0, 10.0, 10.0])
        max_du = np.array([18.210732648355684, 5.886188088635976, 15.0])
        disturbance_magnitude = 0.0
        disturbance_interval = 600
        # Only active when --enable-base-integrator is set.
        ki_base = 0.22991644116687834
        u_bleed = 0.9310825140555963
        base_force_soft_limit = 10.0
        base_damping_gain = 2.4
        base_centering_gain = 0.9
        base_tilt_deadband_deg = 0.4
        base_tilt_full_authority_deg = 1.8
        base_command_gain = 0.70
        base_centering_pos_clip_m = 0.25
        base_speed_soft_limit_frac = 0.55
        base_hold_radius_m = 0.22
        base_ref_follow_rate_hz = 5.5
        base_ref_recenter_rate_hz = 0.90
        base_authority_rate_per_s = 1.8
        base_command_lpf_hz = 7.0
        upright_base_du_scale = 0.30
        base_pitch_kp = 84.0
        base_pitch_kd = 18.0
        base_roll_kp = 34.0
        base_roll_kd = 8.0
        wheel_only_pitch_kp = 180.0
        wheel_only_pitch_kd = 34.0
        wheel_only_pitch_ki = 20.0
        wheel_only_wheel_rate_kd = 0.22
        wheel_only_max_u = 40.0
        wheel_only_max_du = 8.0
    else:
        qx = np.diag([120.0, 90.0, 70.0, 50.0, 1.0, 220.0, 220.0, 520.0, 520.0])
        qu = np.diag([1e-3, 0.08, 0.08])
        r_du = np.diag([3.0, 0.8, 0.8])
        max_u = np.array([100.0, 14.0, 14.0])
        max_du = np.array([10.0, 0.45, 0.45])
        disturbance_magnitude = 0.0
        disturbance_interval = 300
        # Only active when --enable-base-integrator is set.
        ki_base = 1.6
        u_bleed = 0.94
        base_force_soft_limit = 12.0
        base_damping_gain = 2.2
        base_centering_gain = 0.8
        base_tilt_deadband_deg = 0.5
        base_tilt_full_authority_deg = 2.0
        base_command_gain = 0.60
        base_centering_pos_clip_m = 0.20
        base_speed_soft_limit_frac = 0.70
        base_hold_radius_m = 0.20
        base_ref_follow_rate_hz = 9.0
        base_ref_recenter_rate_hz = 0.40
        base_authority_rate_per_s = 2.4
        base_command_lpf_hz = 9.0
        upright_base_du_scale = 0.40
        base_pitch_kp = 78.0
        base_pitch_kd = 16.0
        base_roll_kp = 30.0
        base_roll_kd = 7.0
        wheel_only_pitch_kp = 220.0
        wheel_only_pitch_kd = 40.0
        wheel_only_pitch_ki = 24.0
        wheel_only_wheel_rate_kd = 0.25
        wheel_only_max_u = 50.0
        wheel_only_max_du = 8.0

    # Default momentum-management tuning (non-hardware profiles may override below).
    wheel_momentum_thresh_frac = 0.35
    wheel_momentum_k = 0.40
    wheel_momentum_upright_k = 0.22
    hold_enter_angle_rad = float(np.radians(1.4))
    hold_exit_angle_rad = float(np.radians(2.4))
    hold_enter_rate_rad_s = 0.55
    hold_exit_rate_rad_s = 0.95
    wheel_spin_budget_frac = 0.18
    wheel_spin_hard_frac = 0.42
    wheel_spin_budget_abs_rad_s = 160.0
    wheel_spin_hard_abs_rad_s = 250.0
    high_spin_exit_frac = 0.82
    high_spin_counter_min_frac = 0.12
    high_spin_base_authority_min = 0.45
    wheel_to_base_bias_gain = 0.35
    recovery_wheel_despin_scale = 0.45
    hold_wheel_despin_scale = 1.0
    upright_blend_rise = 0.10
    upright_blend_fall = 0.28

    if stable_demo_profile:
        # Stable demo preset (crash-resistant): keep smoothness, but avoid over-damped recovery.
        qx[4, 4] *= 2.0
        qu[0, 0] *= 1.25
        r_du[0, 0] *= 1.25
        base_tilt_deadband_deg = 0.25
        base_tilt_full_authority_deg = 1.2
        base_authority_rate_per_s = 1.8
        base_command_lpf_hz = 8.0
        upright_base_du_scale = 0.45
        wheel_momentum_thresh_frac = 0.24
        wheel_momentum_k = 0.55
        wheel_momentum_upright_k = 0.30

    if low_spin_robust_profile:
        # Benchmark-injected tuning from final/results/benchmark_20260215_031944.csv (trial_023).
        qx[0, 0] *= 1.7037747217016754
        qx[1, 1] *= 1.7037747217016754
        qx[2, 2] *= 2.9574744481445046
        qx[3, 3] *= 2.9574744481445046
        qx[4, 4] *= 4.410911939796731
        qx[5, 5] *= 0.6781132435145498
        qx[6, 6] *= 0.6781132435145498
        qx[7, 7] *= 1.1422579118979612
        qx[8, 8] *= 1.1422579118979612
        qu *= 1.7831550306923665
        r_du = np.diag([0.12711198214543537, 12.684229592421493, 1.9634054183227847])
        max_du = np.array([68.02730076004603, 10.955727511513118, 1.620934713344921], dtype=float)
        ki_base = 0.18327288746013967
        u_bleed = 0.9033485723919594
        # Stricter anti-runaway overrides for very high wheel-speed failures.
        wheel_momentum_thresh_frac = 0.18
        wheel_momentum_k = 0.90
        wheel_momentum_upright_k = 0.55
        wheel_spin_budget_frac = 0.12
        wheel_spin_hard_frac = 0.25
        wheel_spin_budget_abs_rad_s = 160.0
        wheel_spin_hard_abs_rad_s = 250.0
        high_spin_exit_frac = 0.78
        high_spin_counter_min_frac = 0.20
        high_spin_base_authority_min = 0.60
        wheel_to_base_bias_gain = 0.70

    # Disturbance injection is disabled to avoid non-deterministic external pushes.
    disturbance_magnitude = 0.0

    # Apply hardware torque ceiling for the wheel actuator.
    enforce_wheel_motor_limit = hardware_safe or (args.wheel_torque_limit is not None)
    if enforce_wheel_motor_limit:
        max_u[0] = min(max_u[0], wheel_torque_limit_nm)

    if hardware_safe:
        # Conservative startup profile for real hardware bring-up.
        max_u[0] = min(max_u[0], 0.05)
        max_u[1:] = np.minimum(max_u[1:], np.array([0.8, 0.8]))
        max_du = np.minimum(max_du, np.array([0.25, 0.04, 0.04]))
        ki_base = 0.0
        u_bleed = min(u_bleed, 0.90)
        disturbance_magnitude = 0.0
        max_wheel_speed_rad_s = min(max_wheel_speed_rad_s, 70.0)
        crash_angle_deg = min(float(args.crash_angle_deg), 8.0)
        max_tilt_rate = min(float(args.max_tilt_rate), 2.5)
        max_base_speed = min(float(args.max_base_speed), 0.05)
        wheel_derate_start = min(float(args.wheel_derate_start), 0.40)
        wheel_momentum_thresh_frac = 0.22
        wheel_momentum_k = 0.65
        wheel_momentum_upright_k = 0.30
        base_derate_start = min(float(args.base_derate_start), 0.10)
        base_force_soft_limit = min(base_force_soft_limit, 0.35)
        base_damping_gain = max(base_damping_gain, 3.0)
        base_centering_gain = max(base_centering_gain, 1.2)
        base_tilt_deadband_deg = max(base_tilt_deadband_deg, 2.0)
        base_tilt_full_authority_deg = max(base_tilt_full_authority_deg, 7.0)
        base_command_gain = min(base_command_gain, 0.10)
        base_centering_pos_clip_m = min(base_centering_pos_clip_m, 0.10)
        base_speed_soft_limit_frac = min(base_speed_soft_limit_frac, 0.35)
        base_hold_radius_m = min(base_hold_radius_m, 0.08)
        base_ref_follow_rate_hz = max(base_ref_follow_rate_hz, 12.0)
        base_ref_recenter_rate_hz = min(base_ref_recenter_rate_hz, 0.20)
        base_authority_rate_per_s = min(base_authority_rate_per_s, 1.2)
        base_command_lpf_hz = min(base_command_lpf_hz, 4.0)
        upright_base_du_scale = min(upright_base_du_scale, 0.20)
        base_pitch_kp = min(base_pitch_kp, 8.0)
        base_pitch_kd = min(base_pitch_kd, 2.0)
        base_roll_kp = min(base_roll_kp, 8.0)
        base_roll_kd = min(base_roll_kd, 2.0)
        wheel_only_pitch_kp = min(wheel_only_pitch_kp, 40.0)
        wheel_only_pitch_kd = min(wheel_only_pitch_kd, 8.0)
        wheel_only_pitch_ki = min(wheel_only_pitch_ki, 8.0)
        wheel_only_wheel_rate_kd = min(wheel_only_wheel_rate_kd, 0.10)
        wheel_only_max_u = min(wheel_only_max_u, 6.0)
        wheel_only_max_du = min(wheel_only_max_du, 0.5)
    else:
        crash_angle_deg = float(args.crash_angle_deg)
        max_tilt_rate = float(args.max_tilt_rate)
        max_base_speed = float(args.max_base_speed)
        wheel_derate_start = float(args.wheel_derate_start)
        base_derate_start = float(args.base_derate_start)

    if real_hardware_profile:
        # Physical bring-up: reduce authority and required reaction speed even further.
        max_u[0] = min(max_u[0], 0.03)
        max_u[1:] = np.minimum(max_u[1:], np.array([0.20, 0.20]))
        max_du = np.minimum(max_du, np.array([0.12, 0.02, 0.02]))
        max_wheel_speed_rad_s = min(max_wheel_speed_rad_s, 45.0)
        crash_angle_deg = min(crash_angle_deg, 6.0)
        max_tilt_rate = min(max_tilt_rate, 1.5)
        max_base_speed = min(max_base_speed, 0.02)
        wheel_derate_start = min(wheel_derate_start, 0.30)
        wheel_momentum_thresh_frac = min(wheel_momentum_thresh_frac, 0.18)
        wheel_momentum_k = max(wheel_momentum_k, 0.70)
        wheel_momentum_upright_k = max(wheel_momentum_upright_k, 0.35)
        base_derate_start = min(base_derate_start, 0.08)
        base_force_soft_limit = min(base_force_soft_limit, 0.10)
        base_authority_rate_per_s = min(base_authority_rate_per_s, 0.8)
        base_command_lpf_hz = min(base_command_lpf_hz, 3.0)
        upright_base_du_scale = min(upright_base_du_scale, 0.15)
        wheel_only_pitch_kp = min(wheel_only_pitch_kp, 25.0)
        wheel_only_pitch_kd = min(wheel_only_pitch_kd, 5.0)
        wheel_only_pitch_ki = min(wheel_only_pitch_ki, 4.0)
        wheel_only_wheel_rate_kd = min(wheel_only_wheel_rate_kd, 0.05)
        wheel_only_max_u = min(wheel_only_max_u, 2.5)
        wheel_only_max_du = min(wheel_only_max_du, 0.15)

    # Runtime mode resolution: preserve CLI intent with staged real-hardware safety.
    allow_base_motion_requested = bool(args.allow_base_motion)
    wheel_only_requested = bool(args.wheel_only)
    real_hardware_base_unlocked = bool(args.unlock_base)
    if real_hardware_profile:
        allow_base_motion = allow_base_motion_requested and real_hardware_base_unlocked
    else:
        allow_base_motion = allow_base_motion_requested
    effective_wheel_only = wheel_only_requested or (not allow_base_motion)
    wheel_only_forced = (not wheel_only_requested) and effective_wheel_only
    stop_on_crash = bool(args.stop_on_crash) or real_hardware_profile
    control_hz = float(args.control_hz)
    if stable_demo_profile and (not real_hardware_profile):
        # Stable demo: reduce latency and keep control updates responsive.
        control_hz = max(control_hz, 300.0)
    if real_hardware_profile:
        control_hz = min(control_hz, 200.0)
    control_delay_steps = max(int(args.control_delay_steps), 0)
    if stable_demo_profile and (not real_hardware_profile):
        control_delay_steps = 0
    if real_hardware_profile:
        control_delay_steps = max(control_delay_steps, 1)

    return RuntimeConfig(
        controller_family=str(getattr(args, "controller_family", "current")),
        log_control_terms=bool(getattr(args, "log_control_terms", False)),
        control_terms_csv=getattr(args, "control_terms_csv", None),
        trace_events_csv=getattr(args, "trace_events_csv", None),
        preset=preset,
        stability_profile=stability_profile,
        stable_demo_profile=stable_demo_profile,
        low_spin_robust_profile=low_spin_robust_profile,
        smooth_viewer=smooth_viewer,
        real_hardware_profile=real_hardware_profile,
        hardware_safe=hardware_safe,
        easy_mode=bool(args.easy_mode),
        stop_on_crash=stop_on_crash,
        wheel_only=effective_wheel_only,
        wheel_only_forced=wheel_only_forced,
        allow_base_motion_requested=allow_base_motion_requested,
        real_hardware_base_unlocked=real_hardware_base_unlocked,
        allow_base_motion=allow_base_motion,
        seed=int(args.seed),
        hardware_realistic=(not args.legacy_model) or real_hardware_profile,
        control_hz=control_hz,
        control_delay_steps=control_delay_steps,
        wheel_encoder_ticks_per_rev=max(int(args.wheel_encoder_ticks), 1),
        imu_angle_noise_std_rad=float(np.radians(args.imu_angle_noise_deg)),
        imu_rate_noise_std_rad_s=float(args.imu_rate_noise),
        wheel_encoder_rate_noise_std_rad_s=float(args.wheel_rate_noise),
        base_encoder_pos_noise_std_m=float(max(args.base_pos_noise, 0.0)),
        base_encoder_vel_noise_std_m_s=float(max(args.base_vel_noise, 0.0)),
        base_state_from_sensors=bool((not args.legacy_model) or real_hardware_profile),
        max_u=max_u,
        max_du=max_du,
        disturbance_magnitude=disturbance_magnitude,
        disturbance_interval=disturbance_interval,
        qx=qx,
        qu=qu,
        r_du=r_du,
        ki_base=ki_base,
        base_integrator_enabled=bool(args.enable_base_integrator),
        u_bleed=u_bleed,
        crash_angle_rad=np.radians(crash_angle_deg),
        x_ref=0.0,
        y_ref=0.0,
        int_clamp=2.0,
        upright_angle_thresh=np.radians(3.0),
        upright_vel_thresh=0.10,
        upright_pos_thresh=0.30,
        max_wheel_speed_rad_s=max_wheel_speed_rad_s,
        max_pitch_roll_rate_rad_s=max_tilt_rate,
        max_base_speed_m_s=max_base_speed,
        wheel_torque_derate_start=float(np.clip(wheel_derate_start, 0.0, 0.99)),
        wheel_momentum_thresh_frac=float(np.clip(wheel_momentum_thresh_frac, 0.0, 0.95)),
        wheel_momentum_k=float(max(wheel_momentum_k, 0.0)),
        wheel_momentum_upright_k=float(max(wheel_momentum_upright_k, 0.0)),
        hold_enter_angle_rad=hold_enter_angle_rad,
        hold_exit_angle_rad=hold_exit_angle_rad,
        hold_enter_rate_rad_s=hold_enter_rate_rad_s,
        hold_exit_rate_rad_s=hold_exit_rate_rad_s,
        wheel_spin_budget_frac=float(np.clip(wheel_spin_budget_frac, 0.0, 0.95)),
        wheel_spin_hard_frac=float(np.clip(wheel_spin_hard_frac, 0.05, 0.99)),
        wheel_spin_budget_abs_rad_s=float(max(wheel_spin_budget_abs_rad_s, 1.0)),
        wheel_spin_hard_abs_rad_s=float(max(wheel_spin_hard_abs_rad_s, 2.0)),
        high_spin_exit_frac=float(np.clip(high_spin_exit_frac, 0.40, 0.98)),
        high_spin_counter_min_frac=float(np.clip(high_spin_counter_min_frac, 0.0, 0.95)),
        high_spin_base_authority_min=float(np.clip(high_spin_base_authority_min, 0.0, 1.0)),
        wheel_to_base_bias_gain=float(max(wheel_to_base_bias_gain, 0.0)),
        recovery_wheel_despin_scale=float(max(recovery_wheel_despin_scale, 0.0)),
        hold_wheel_despin_scale=float(max(hold_wheel_despin_scale, 0.0)),
        upright_blend_rise=float(np.clip(upright_blend_rise, 0.0, 1.0)),
        upright_blend_fall=float(np.clip(upright_blend_fall, 0.0, 1.0)),
        base_torque_derate_start=float(np.clip(base_derate_start, 0.0, 0.99)),
        wheel_torque_limit_nm=wheel_torque_limit_nm,
        enforce_wheel_motor_limit=enforce_wheel_motor_limit,
        wheel_motor_kv_rpm_per_v=kv_rpm_per_v,
        wheel_motor_resistance_ohm=r_ohm,
        wheel_current_limit_a=i_lim,
        bus_voltage_v=v_bus,
        wheel_gear_ratio=gear,
        drive_efficiency=eta,
        base_force_soft_limit=base_force_soft_limit,
        base_damping_gain=base_damping_gain,
        base_centering_gain=base_centering_gain,
        base_tilt_deadband_rad=float(np.radians(base_tilt_deadband_deg)),
        base_tilt_full_authority_rad=float(np.radians(base_tilt_full_authority_deg)),
        base_command_gain=base_command_gain,
        base_centering_pos_clip_m=base_centering_pos_clip_m,
        base_speed_soft_limit_frac=float(np.clip(base_speed_soft_limit_frac, 0.05, 0.95)),
        base_hold_radius_m=base_hold_radius_m,
        base_ref_follow_rate_hz=base_ref_follow_rate_hz,
        base_ref_recenter_rate_hz=base_ref_recenter_rate_hz,
        base_authority_rate_per_s=base_authority_rate_per_s,
        base_command_lpf_hz=base_command_lpf_hz,
        upright_base_du_scale=upright_base_du_scale,
        base_pitch_kp=base_pitch_kp,
        base_pitch_kd=base_pitch_kd,
        base_roll_kp=base_roll_kp,
        base_roll_kd=base_roll_kd,
        wheel_only_pitch_kp=wheel_only_pitch_kp,
        wheel_only_pitch_kd=wheel_only_pitch_kd,
        wheel_only_pitch_ki=wheel_only_pitch_ki,
        wheel_only_wheel_rate_kd=wheel_only_wheel_rate_kd,
        wheel_only_max_u=wheel_only_max_u,
        wheel_only_max_du=wheel_only_max_du,
        wheel_only_int_clamp=0.35,
    )


def reset_state(model, data, q_pitch, q_roll, pitch_eq=0.0, roll_eq=0.0):
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    data.qpos[q_pitch] = pitch_eq
    data.qpos[q_roll] = roll_eq
    mujoco.mj_forward(model, data)


def build_partial_measurement_matrix(cfg: RuntimeConfig):
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
    return np.array(rows, dtype=float)


def build_measurement_noise_cov(cfg: RuntimeConfig, wheel_lsb: float) -> np.ndarray:
    variances = [
        cfg.imu_angle_noise_std_rad**2,
        cfg.imu_angle_noise_std_rad**2,
        cfg.imu_rate_noise_std_rad_s**2,
        cfg.imu_rate_noise_std_rad_s**2,
        (wheel_lsb**2) / 12.0 + cfg.wheel_encoder_rate_noise_std_rad_s**2,
    ]
    if cfg.base_state_from_sensors:
        variances.extend(
            [
                cfg.base_encoder_pos_noise_std_m**2,
                cfg.base_encoder_pos_noise_std_m**2,
                cfg.base_encoder_vel_noise_std_m_s**2,
                cfg.base_encoder_vel_noise_std_m_s**2,
            ]
        )
    return np.diag(variances)


def build_kalman_gain(A: np.ndarray, Qn: np.ndarray, C: np.ndarray, R: np.ndarray):
    Pk = solve_discrete_are(A.T, C.T, Qn, R)
    return Pk @ C.T @ np.linalg.inv(C @ Pk @ C.T + R)


def estimator_measurement_update(
    cfg: RuntimeConfig,
    x_true: np.ndarray,
    x_pred: np.ndarray,
    C: np.ndarray,
    L: np.ndarray,
    rng: np.random.Generator,
    wheel_lsb: float,
) -> np.ndarray:
    """One Kalman correction step from noisy/quantized sensors."""
    wheel_quant = np.round(x_true[4] / wheel_lsb) * wheel_lsb
    y = np.array(
        [
            x_true[0] + rng.normal(0.0, cfg.imu_angle_noise_std_rad),
            x_true[1] + rng.normal(0.0, cfg.imu_angle_noise_std_rad),
            x_true[2] + rng.normal(0.0, cfg.imu_rate_noise_std_rad_s),
            x_true[3] + rng.normal(0.0, cfg.imu_rate_noise_std_rad_s),
            wheel_quant + rng.normal(0.0, cfg.wheel_encoder_rate_noise_std_rad_s),
        ],
        dtype=float,
    )
    if cfg.base_state_from_sensors:
        y = np.concatenate(
            [
                y,
                np.array(
                    [
                        x_true[5] + rng.normal(0.0, cfg.base_encoder_pos_noise_std_m),
                        x_true[6] + rng.normal(0.0, cfg.base_encoder_pos_noise_std_m),
                        x_true[7] + rng.normal(0.0, cfg.base_encoder_vel_noise_std_m_s),
                        x_true[8] + rng.normal(0.0, cfg.base_encoder_vel_noise_std_m_s),
                    ],
                    dtype=float,
                ),
            ]
        )
    x_est = x_pred + L @ (y - C @ x_pred)

    if not cfg.base_state_from_sensors:
        # Legacy teaching model: keep unobserved base states tied to truth.
        x_est[5] = x_true[5]
        x_est[6] = x_true[6]
        x_est[7] = x_true[7]
        x_est[8] = x_true[8]
    return x_est


def _init_control_terms() -> dict[str, np.ndarray]:
    return {
        "term_lqr_core": np.zeros(3, dtype=float),
        "term_roll_stability": np.zeros(3, dtype=float),
        "term_pitch_stability": np.zeros(3, dtype=float),
        "term_despin": np.zeros(3, dtype=float),
        "term_base_hold": np.zeros(3, dtype=float),
        "term_safety_shaping": np.zeros(3, dtype=float),
    }


def _fuzzy_roll_gain(cfg: RuntimeConfig, roll: float, roll_rate: float) -> float:
    roll_n = abs(roll) / max(cfg.hold_exit_angle_rad, 1e-6)
    rate_n = abs(roll_rate) / max(cfg.hold_exit_rate_rad_s, 1e-6)
    level = float(np.clip(0.65 * roll_n + 0.35 * rate_n, 0.0, 1.0))
    return 0.35 + 0.95 * level


def compute_control_command(
    cfg: RuntimeConfig,
    x_est: np.ndarray,
    x_true: np.ndarray,
    u_eff_applied: np.ndarray,
    base_int: np.ndarray,
    base_ref: np.ndarray,
    base_authority_state: float,
    u_base_smooth: np.ndarray,
    wheel_pitch_int: float,
    balance_phase: str,
    recovery_time_s: float,
    high_spin_active: bool,
    control_dt: float,
    K_du: np.ndarray,
    K_wheel_only: np.ndarray | None,
    K_paper_pitch: np.ndarray | None,
    du_hits: np.ndarray,
    sat_hits: np.ndarray,
):
    """Controller core: delta-u LQR + wheel-only mode + base policy + safety shaping."""
    wheel_over_budget = False
    wheel_over_hard = False
    terms = _init_control_terms()
    x_ctrl = x_est.copy()
    x_ctrl[5] -= cfg.x_ref
    x_ctrl[6] -= cfg.y_ref

    if cfg.base_integrator_enabled:
        base_int[0] = np.clip(base_int[0] + x_ctrl[5] * control_dt, -cfg.int_clamp, cfg.int_clamp)
        base_int[1] = np.clip(base_int[1] + x_ctrl[6] * control_dt, -cfg.int_clamp, cfg.int_clamp)
    else:
        base_int[:] = 0.0

    # Use effective applied command in delta-u state to avoid windup when
    # saturation/delay/motor limits differ from requested command.
    z = np.concatenate([x_ctrl, u_eff_applied])
    du_lqr = -K_du @ z
    terms["term_lqr_core"] = np.array([du_lqr[0], du_lqr[1], du_lqr[2]], dtype=float)

    # Literature-style benchmark comparator:
    # pitch = LQR channel, roll = sliding mode with fuzzy gain.
    if cfg.controller_family == "paper_split_baseline":
        xw = np.array([x_est[0], x_est[2], x_est[4]], dtype=float)
        if K_paper_pitch is None:
            u_pitch = float(-0.35 * x_est[0] - 0.11 * x_est[2] - 0.03 * x_est[4])
        else:
            u_pitch = float(-(K_paper_pitch @ xw)[0])
        lam_roll = 0.42
        phi = max(np.radians(0.5), 1e-3)
        s_roll = float(x_est[1] + lam_roll * x_est[3])
        k_fuzzy = _fuzzy_roll_gain(cfg, float(x_est[1]), float(x_est[3]))
        u_roll_sm = float(-k_fuzzy * np.tanh(s_roll / phi) * 0.45 * cfg.max_u[0])

        terms["term_pitch_stability"][0] = u_pitch
        terms["term_roll_stability"][0] = u_roll_sm
        u_rw_target = u_pitch + u_roll_sm
        du_rw_cmd = float(u_rw_target - u_eff_applied[0])
        rw_du_limit = cfg.max_du[0]
        rw_u_limit = cfg.max_u[0]
        du_hits[0] += int(abs(du_rw_cmd) > rw_du_limit)
        du_rw = float(np.clip(du_rw_cmd, -rw_du_limit, rw_du_limit))
        u_rw_unc = float(u_eff_applied[0] + du_rw)
        sat_hits[0] += int(abs(u_rw_unc) > rw_u_limit)
        u_rw_cmd = float(np.clip(u_rw_unc, -rw_u_limit, rw_u_limit))

        hold_x = -cfg.base_damping_gain * x_est[7] - cfg.base_centering_gain * x_est[5]
        hold_y = -cfg.base_damping_gain * x_est[8] - cfg.base_centering_gain * x_est[6]
        terms["term_base_hold"][1:] = np.array([hold_x, hold_y], dtype=float)
        if cfg.allow_base_motion:
            balance_x = cfg.base_command_gain * (cfg.base_pitch_kp * x_est[0] + cfg.base_pitch_kd * x_est[2])
            balance_y = -cfg.base_command_gain * (cfg.base_roll_kp * x_est[1] + cfg.base_roll_kd * x_est[3])
            # Roll sliding compensation influences base y to emulate split-channel behavior.
            balance_y += float(-0.25 * np.sign(s_roll) * cfg.max_u[2] * np.tanh(abs(s_roll) / phi))
            terms["term_pitch_stability"][1] = balance_x
            terms["term_roll_stability"][2] = balance_y
            base_target = np.array([hold_x + balance_x, hold_y + balance_y], dtype=float)
            du_base_cmd = base_target - u_eff_applied[1:]
            du_hits[1:] += (np.abs(du_base_cmd) > cfg.max_du[1:]).astype(int)
            du_base = np.clip(du_base_cmd, -cfg.max_du[1:], cfg.max_du[1:])
            u_base_unc = u_eff_applied[1:] + du_base
            sat_hits[1:] += (np.abs(u_base_unc) > cfg.max_u[1:]).astype(int)
            u_base_cmd = np.clip(u_base_unc, -cfg.max_u[1:], cfg.max_u[1:])
        else:
            u_base_cmd = np.zeros(2, dtype=float)
            base_int[:] = 0.0
            base_ref[:] = 0.0
            base_authority_state = 0.0
            u_base_smooth[:] = 0.0

        u_cmd = np.array([u_rw_cmd, u_base_cmd[0], u_base_cmd[1]], dtype=float)
        if cfg.hardware_safe:
            terms["term_safety_shaping"][1:] += u_cmd[1:] * (-0.75)
            u_cmd[1:] = np.clip(0.25 * u_cmd[1:], -0.35, 0.35)
        return (
            u_cmd,
            base_int,
            base_ref,
            base_authority_state,
            u_base_smooth,
            wheel_pitch_int,
            rw_u_limit,
            wheel_over_budget,
            wheel_over_hard,
            high_spin_active,
            terms,
        )

    if cfg.wheel_only:
        xw = np.array([x_est[0], x_est[2], x_est[4]], dtype=float)
        u_rw_target = float(-(K_wheel_only @ xw)[0])
        wheel_pitch_int = float(
            np.clip(
                wheel_pitch_int + x_est[0] * control_dt,
                -cfg.wheel_only_int_clamp,
                cfg.wheel_only_int_clamp,
            )
        )
        u_rw_target += -cfg.wheel_only_pitch_ki * wheel_pitch_int
        u_rw_target += -cfg.wheel_only_wheel_rate_kd * x_est[4]
        du_rw_cmd = float(u_rw_target - u_eff_applied[0])
        rw_du_limit = cfg.wheel_only_max_du
        rw_u_limit = cfg.wheel_only_max_u
        base_int[:] = 0.0
        base_ref[:] = 0.0
    else:
        wheel_pitch_int = 0.0
        rw_frac = abs(float(x_est[4])) / max(cfg.max_wheel_speed_rad_s, 1e-6)
        rw_damp_gain = 0.18 + 0.60 * max(0.0, rw_frac - 0.35)
        du_rw_cmd = float(du_lqr[0] - rw_damp_gain * x_est[4])
        terms["term_despin"][0] += float(-rw_damp_gain * x_est[4])
        if cfg.controller_family == "hybrid_modern":
            pitch_stab = float(-(0.12 * cfg.base_pitch_kp) * x_est[0] - (0.10 * cfg.base_pitch_kd) * x_est[2])
            roll_stab = float((0.06 * cfg.base_roll_kp) * x_est[1] + (0.06 * cfg.base_roll_kd) * x_est[3])
            du_rw_cmd += pitch_stab + roll_stab
            terms["term_pitch_stability"][0] += pitch_stab
            terms["term_roll_stability"][0] += roll_stab
        rw_du_limit = cfg.max_du[0]
        rw_u_limit = cfg.max_u[0]

    du_hits[0] += int(abs(du_rw_cmd) > rw_du_limit)
    du_rw = float(np.clip(du_rw_cmd, -rw_du_limit, rw_du_limit))
    u_rw_unc = float(u_eff_applied[0] + du_rw)
    sat_hits[0] += int(abs(u_rw_unc) > rw_u_limit)
    u_rw_cmd = float(np.clip(u_rw_unc, -rw_u_limit, rw_u_limit))

    wheel_speed_abs_est = abs(float(x_est[4]))
    wheel_derate_start_speed = cfg.wheel_torque_derate_start * cfg.max_wheel_speed_rad_s
    if wheel_speed_abs_est > wheel_derate_start_speed:
        span = max(cfg.max_wheel_speed_rad_s - wheel_derate_start_speed, 1e-6)
        rw_scale = max(0.0, 1.0 - (wheel_speed_abs_est - wheel_derate_start_speed) / span)
        rw_cap = rw_u_limit * rw_scale
        terms["term_safety_shaping"][0] += float(np.clip(u_rw_cmd, -rw_u_limit, rw_u_limit) - np.clip(u_rw_cmd, -rw_cap, rw_cap))
        u_rw_cmd = float(np.clip(u_rw_cmd, -rw_cap, rw_cap))

    # Explicit wheel momentum management: push wheel speed back toward zero.
    hard_frac = cfg.wheel_spin_hard_frac
    budget_speed = min(cfg.wheel_spin_budget_frac * cfg.max_wheel_speed_rad_s, cfg.wheel_spin_budget_abs_rad_s)
    hard_speed = min(hard_frac * cfg.max_wheel_speed_rad_s, cfg.wheel_spin_hard_abs_rad_s)
    if not cfg.allow_base_motion:
        hard_speed *= 1.10
    if high_spin_active:
        high_spin_exit_speed = cfg.high_spin_exit_frac * hard_speed
        if wheel_speed_abs_est < high_spin_exit_speed:
            high_spin_active = False
    elif wheel_speed_abs_est > hard_speed:
        high_spin_active = True

    momentum_speed = min(cfg.wheel_momentum_thresh_frac * cfg.max_wheel_speed_rad_s, budget_speed)
    if wheel_speed_abs_est > momentum_speed:
        pre_span = max(budget_speed - momentum_speed, 1e-6)
        pre_over = float(np.clip((wheel_speed_abs_est - momentum_speed) / pre_span, 0.0, 1.0))
        despin_term = float(
            np.clip(
                -np.sign(x_est[4]) * 0.35 * cfg.wheel_momentum_k * pre_over * rw_u_limit,
                -0.30 * rw_u_limit,
                0.30 * rw_u_limit,
            )
        )
        terms["term_despin"][0] += despin_term
        u_rw_cmd += despin_term

    if wheel_speed_abs_est > budget_speed:
        wheel_over_budget = True
        speed_span = max(hard_speed - budget_speed, 1e-6)
        over = np.clip((wheel_speed_abs_est - budget_speed) / speed_span, 0.0, 1.5)
        despin_term = float(
            np.clip(
                -np.sign(x_est[4]) * cfg.wheel_momentum_k * over * rw_u_limit,
                -0.65 * rw_u_limit,
                0.65 * rw_u_limit,
            )
        )
        terms["term_despin"][0] += despin_term
        u_rw_cmd += despin_term
        if (wheel_speed_abs_est <= hard_speed) and (not high_spin_active):
            rw_cap_scale = max(0.55, 1.0 - 0.45 * float(over))
        else:
            wheel_over_hard = True
            rw_cap_scale = 0.35
            tilt_mag = max(abs(float(x_est[0])), abs(float(x_est[1])))
            if balance_phase == "recovery" and recovery_time_s < 0.12 and tilt_mag > cfg.hold_exit_angle_rad:
                rw_cap_scale = max(rw_cap_scale, 0.55)
            over_hard = float(np.clip((wheel_speed_abs_est - hard_speed) / max(hard_speed, 1e-6), 0.0, 1.0))
            emergency_counter = -np.sign(x_est[4]) * (0.60 + 0.35 * over_hard) * rw_u_limit
            # At very high wheel speed, prefer desaturation over same-direction torque.
            if balance_phase != "recovery" or recovery_time_s >= 0.12 or tilt_mag <= cfg.hold_exit_angle_rad:
                if np.sign(u_rw_cmd) == np.sign(x_est[4]):
                    u_rw_cmd = float(emergency_counter)
            # Keep a minimum opposite-direction command while latched to high-spin.
            if np.sign(u_rw_cmd) != np.sign(x_est[4]):
                min_counter = cfg.high_spin_counter_min_frac * rw_u_limit
                u_rw_cmd = float(np.sign(u_rw_cmd) * max(abs(u_rw_cmd), min_counter))
            if high_spin_active:
                u_rw_cmd = float(0.25 * u_rw_cmd + 0.75 * emergency_counter)
        u_rw_cmd = float(np.clip(u_rw_cmd, -rw_cap_scale * rw_u_limit, rw_cap_scale * rw_u_limit))

    near_upright_for_wheel = (
        abs(x_true[0]) < cfg.upright_angle_thresh
        and abs(x_true[1]) < cfg.upright_angle_thresh
        and abs(x_true[2]) < cfg.upright_vel_thresh
        and abs(x_true[3]) < cfg.upright_vel_thresh
    )
    if near_upright_for_wheel:
        phase_scale = cfg.hold_wheel_despin_scale if balance_phase == "hold" else cfg.recovery_wheel_despin_scale
        despin_term = float(
            np.clip(-phase_scale * cfg.wheel_momentum_upright_k * x_est[4], -0.35 * rw_u_limit, 0.35 * rw_u_limit)
        )
        terms["term_despin"][0] += despin_term
        u_rw_cmd += despin_term
    u_rw_cmd = float(np.clip(u_rw_cmd, -rw_u_limit, rw_u_limit))

    if cfg.allow_base_motion:
        tilt_span = max(cfg.base_tilt_full_authority_rad - cfg.base_tilt_deadband_rad, 1e-6)
        tilt_mag = max(abs(x_est[0]), abs(x_est[1]))
        base_authority_raw = float(np.clip((tilt_mag - cfg.base_tilt_deadband_rad) / tilt_span, 0.0, 1.0))
        if high_spin_active:
            base_authority_raw = max(base_authority_raw, cfg.high_spin_base_authority_min)
        if tilt_mag > 1.5 * cfg.base_tilt_full_authority_rad:
            base_authority_raw *= 0.55
        max_auth_delta = cfg.base_authority_rate_per_s * control_dt
        base_authority_state += float(np.clip(base_authority_raw - base_authority_state, -max_auth_delta, max_auth_delta))
        base_authority = float(np.clip(base_authority_state, 0.0, 1.0))
        follow_alpha = float(np.clip(cfg.base_ref_follow_rate_hz * control_dt, 0.0, 1.0))
        recenter_alpha = float(np.clip(cfg.base_ref_recenter_rate_hz * control_dt, 0.0, 1.0))
        base_disp = float(np.hypot(x_est[5], x_est[6]))
        if base_authority > 0.35 and base_disp < cfg.base_hold_radius_m:
            base_ref[0] += follow_alpha * (x_est[5] - base_ref[0])
            base_ref[1] += follow_alpha * (x_est[6] - base_ref[1])
        else:
            base_ref[0] += recenter_alpha * (0.0 - base_ref[0])
            base_ref[1] += recenter_alpha * (0.0 - base_ref[1])

        base_x_err = float(np.clip(x_est[5] - base_ref[0], -cfg.base_centering_pos_clip_m, cfg.base_centering_pos_clip_m))
        base_y_err = float(np.clip(x_est[6] - base_ref[1], -cfg.base_centering_pos_clip_m, cfg.base_centering_pos_clip_m))
        hold_x = -cfg.base_damping_gain * x_est[7] - cfg.base_centering_gain * base_x_err
        hold_y = -cfg.base_damping_gain * x_est[8] - cfg.base_centering_gain * base_y_err
        terms["term_base_hold"][1:] = np.array([hold_x, hold_y], dtype=float)
        balance_x = cfg.base_command_gain * (cfg.base_pitch_kp * x_est[0] + cfg.base_pitch_kd * x_est[2])
        balance_y = -cfg.base_command_gain * (cfg.base_roll_kp * x_est[1] + cfg.base_roll_kd * x_est[3])
        terms["term_pitch_stability"][1] = balance_x
        terms["term_roll_stability"][2] = balance_y
        if cfg.controller_family == "hybrid_modern":
            # Hybrid modern: explicit cross-coupled stabilization terms.
            cross_pitch = float(-0.08 * cfg.base_roll_kp * x_est[1] - 0.05 * cfg.base_roll_kd * x_est[3])
            cross_roll = float(0.08 * cfg.base_pitch_kp * x_est[0] + 0.05 * cfg.base_pitch_kd * x_est[2])
            balance_x += cross_pitch
            balance_y += cross_roll
            terms["term_roll_stability"][1] += cross_pitch
            terms["term_pitch_stability"][2] += cross_roll
        base_target_x = (1.0 - base_authority) * hold_x + base_authority * balance_x
        base_target_y = (1.0 - base_authority) * hold_y + base_authority * balance_y
        if cfg.base_integrator_enabled and cfg.ki_base > 0.0:
            base_target_x += -cfg.ki_base * base_int[0]
            base_target_y += -cfg.ki_base * base_int[1]
        if wheel_speed_abs_est > budget_speed:
            over_budget = float(
                np.clip(
                    (wheel_speed_abs_est - budget_speed) / max(hard_speed - budget_speed, 1e-6),
                    0.0,
                    1.0,
                )
            )
            extra_bias = 1.25 if high_spin_active else 1.0
            bias_term = -np.sign(x_est[4]) * cfg.wheel_to_base_bias_gain * extra_bias * over_budget
            terms["term_despin"][1] += bias_term
            base_target_x += bias_term

        du_base_cmd = np.array([base_target_x, base_target_y]) - u_eff_applied[1:]
        base_du_limit = cfg.max_du[1:].copy()
        near_upright_for_base = (
            abs(x_true[0]) < cfg.upright_angle_thresh
            and abs(x_true[1]) < cfg.upright_angle_thresh
            and abs(x_true[2]) < cfg.upright_vel_thresh
            and abs(x_true[3]) < cfg.upright_vel_thresh
        )
        if near_upright_for_base:
            base_du_limit *= cfg.upright_base_du_scale
        du_hits[1:] += (np.abs(du_base_cmd) > base_du_limit).astype(int)
        du_base = np.clip(du_base_cmd, -base_du_limit, base_du_limit)
        u_base_unc = u_eff_applied[1:] + du_base
        sat_hits[1:] += (np.abs(u_base_unc) > cfg.max_u[1:]).astype(int)
        u_base_cmd = np.clip(u_base_unc, -cfg.max_u[1:], cfg.max_u[1:])
        base_lpf_alpha = float(np.clip(cfg.base_command_lpf_hz * control_dt, 0.0, 1.0))
        u_base_smooth += base_lpf_alpha * (u_base_cmd - u_base_smooth)
        u_base_cmd = u_base_smooth.copy()
    else:
        base_int[:] = 0.0
        base_ref[:] = 0.0
        base_authority_state = 0.0
        du_base_cmd = -u_eff_applied[1:]
        du_hits[1:] += (np.abs(du_base_cmd) > cfg.max_du[1:]).astype(int)
        du_base = np.clip(du_base_cmd, -cfg.max_du[1:], cfg.max_du[1:])
        u_base_cmd = u_eff_applied[1:] + du_base
        u_base_smooth[:] = 0.0

    u_cmd = np.array([u_rw_cmd, u_base_cmd[0], u_base_cmd[1]], dtype=float)
    if cfg.hardware_safe:
        terms["term_safety_shaping"][1:] += u_cmd[1:] * (-0.75)
        u_cmd[1:] = np.clip(0.25 * u_cmd[1:], -0.35, 0.35)

    return (
        u_cmd,
        base_int,
        base_ref,
        base_authority_state,
        u_base_smooth,
        wheel_pitch_int,
        rw_u_limit,
        wheel_over_budget,
        wheel_over_hard,
        high_spin_active,
        terms,
    )


def apply_upright_postprocess(
    cfg: RuntimeConfig,
    u_cmd: np.ndarray,
    x_est: np.ndarray,
    x_true: np.ndarray,
    upright_blend: float,
    balance_phase: str,
    high_spin_active: bool,
    despin_gain: float,
    rw_u_limit: float,
):
    near_upright = (
        abs(x_true[0]) < cfg.upright_angle_thresh
        and abs(x_true[1]) < cfg.upright_angle_thresh
        and abs(x_true[2]) < cfg.upright_vel_thresh
        and abs(x_true[3]) < cfg.upright_vel_thresh
        and abs(x_true[5]) < cfg.upright_pos_thresh
        and abs(x_true[6]) < cfg.upright_pos_thresh
    )
    quasi_upright = (
        abs(x_true[0]) < 1.8 * cfg.upright_angle_thresh
        and abs(x_true[1]) < 1.8 * cfg.upright_angle_thresh
        and abs(x_true[2]) < 1.8 * cfg.upright_vel_thresh
        and abs(x_true[3]) < 1.8 * cfg.upright_vel_thresh
    )
    if near_upright:
        upright_target = 1.0
    elif quasi_upright:
        upright_target = 0.35
    else:
        upright_target = 0.0
    blend_alpha = cfg.upright_blend_rise if upright_target > upright_blend else cfg.upright_blend_fall
    upright_blend += blend_alpha * (upright_target - upright_blend)
    if upright_blend > 1e-6:
        bleed_scale = 1.0 - upright_blend * (1.0 - cfg.u_bleed)
        if high_spin_active:
            # Do not bleed emergency wheel despin torque in high-spin recovery.
            u_cmd[1:] *= bleed_scale
        else:
            u_cmd *= bleed_scale
        phase_scale = cfg.hold_wheel_despin_scale if balance_phase == "hold" else cfg.recovery_wheel_despin_scale
        u_cmd[0] += upright_blend * float(
            np.clip(-phase_scale * despin_gain * x_est[4], -0.50 * rw_u_limit, 0.50 * rw_u_limit)
        )
        u_cmd[np.abs(u_cmd) < 1e-3] = 0.0
    return u_cmd, upright_blend


def apply_control_delay(cfg: RuntimeConfig, cmd_queue: deque, u_cmd: np.ndarray) -> np.ndarray:
    if cfg.hardware_realistic:
        cmd_queue.append(u_cmd.copy())
        return cmd_queue.popleft()
    return u_cmd.copy()


def main():
    # 1) Parse CLI and build runtime tuning/safety profile.
    args = parse_args()
    cfg = build_config(args)
    initial_roll_rad = float(np.radians(args.initial_y_tilt_deg))
    push_force_world = np.array([float(args.push_x), float(args.push_y), 0.0], dtype=float)
    push_end_s = float(args.push_start_s + max(0.0, args.push_duration_s))

    # 2) Load MuJoCo model from XML (source of truth for geometry/joints/actuators).
    xml_path = Path(__file__).with_name("final.xml")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    if cfg.smooth_viewer and cfg.easy_mode:
        model.opt.gravity[2] = -6.5

    ids = lookup_model_ids(model)
    push_body_id = {
        "stick": ids.stick_body_id,
        "base_y": ids.base_y_body_id,
        "base_x": ids.base_x_body_id,
    }[args.push_body]
    reset_state(model, data, ids.q_pitch, ids.q_roll, pitch_eq=0.0, roll_eq=initial_roll_rad)

    # 3) Linearize XML-defined dynamics about upright and build controller gains.
    nx = model.nq + model.nv
    nu = model.nu
    A_full = np.zeros((nx, nx))
    B_full = np.zeros((nx, nu))
    mujoco.mjd_transitionFD(model, data, 1e-6, True, A_full, B_full, None, None)

    idx = [
        ids.q_pitch,
        ids.q_roll,
        model.nq + ids.v_pitch,
        model.nq + ids.v_roll,
        model.nq + ids.v_rw,
        ids.q_base_x,
        ids.q_base_y,
        model.nq + ids.v_base_x,
        model.nq + ids.v_base_y,
    ]
    A = A_full[np.ix_(idx, idx)]
    B = B_full[np.ix_(idx, [ids.aid_rw, ids.aid_base_x, ids.aid_base_y])]
    NX = A.shape[0]
    NU = B.shape[1]

    A_aug = np.block([[A, B], [np.zeros((NU, NX)), np.eye(NU)]])
    B_aug = np.vstack([B, np.eye(NU)])
    Q_aug = np.block([[cfg.qx, np.zeros((NX, NU))], [np.zeros((NU, NX)), cfg.qu]])
    P_aug = solve_discrete_are(A_aug, B_aug, Q_aug, cfg.r_du)
    K_du = np.linalg.inv(B_aug.T @ P_aug @ B_aug + cfg.r_du) @ (B_aug.T @ P_aug @ A_aug)
    A_w = A[np.ix_([0, 2, 4], [0, 2, 4])]
    B_w = B[np.ix_([0, 2, 4], [0])]
    Q_w = np.diag([260.0, 35.0, 0.6])
    R_w = np.array([[0.08]])
    P_w = solve_discrete_are(A_w, B_w, Q_w, R_w)
    K_paper_pitch = np.linalg.inv(B_w.T @ P_w @ B_w + R_w) @ (B_w.T @ P_w @ A_w)
    K_wheel_only = None
    if cfg.wheel_only:
        K_wheel_only = K_paper_pitch.copy()

    # 4) Build estimator model from configured sensor channels/noise.
    control_steps = 1 if not cfg.hardware_realistic else max(1, int(round(1.0 / (model.opt.timestep * cfg.control_hz))))
    control_dt = control_steps * model.opt.timestep
    wheel_lsb = (2.0 * np.pi) / (cfg.wheel_encoder_ticks_per_rev * control_dt)
    C = build_partial_measurement_matrix(cfg)
    Qn = np.diag([1e-4, 1e-4, 1e-4, 1e-4, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5])
    Rn = build_measurement_noise_cov(cfg, wheel_lsb)
    L = build_kalman_gain(A, Qn, C, Rn)

    ACT_IDS = np.array([ids.aid_rw, ids.aid_base_x, ids.aid_base_y], dtype=int)
    ACT_NAMES = np.array(["wheel_spin", "base_x_force", "base_y_force"])
    XML_CTRL_LOW = model.actuator_ctrlrange[ACT_IDS, 0]
    XML_CTRL_HIGH = model.actuator_ctrlrange[ACT_IDS, 1]
    for i, name in enumerate(ACT_NAMES):
        if XML_CTRL_LOW[i] > -cfg.max_u[i] or XML_CTRL_HIGH[i] < cfg.max_u[i]:
            raise ValueError(
                f"{name}: xml=[{XML_CTRL_LOW[i]:.3f}, {XML_CTRL_HIGH[i]:.3f}] vs python=+/-{cfg.max_u[i]:.3f}"
            )

    print("\n=== LINEARIZATION ===")
    print(f"A shape: {A.shape}, B shape: {B.shape}")
    print(f"A eigenvalues: {np.linalg.eigvals(A)}")
    print("\n=== DELTA-U LQR ===")
    print(f"K_du shape: {K_du.shape}")
    print(f"controller_family={cfg.controller_family}")
    if K_wheel_only is not None:
        print(f"wheel_only_K: {K_wheel_only}")
    print("\n=== VIEWER MODE ===")
    print(f"preset={cfg.preset} stable_demo_profile={cfg.stable_demo_profile}")
    print(f"stability_profile={cfg.stability_profile} low_spin_robust_profile={cfg.low_spin_robust_profile}")
    print(f"mode={'smooth' if cfg.smooth_viewer else 'robust'}")
    print(f"real_hardware_profile={cfg.real_hardware_profile}")
    print(f"hardware_safe={cfg.hardware_safe}")
    print(f"easy_mode={cfg.easy_mode}")
    print(f"stop_on_crash={cfg.stop_on_crash}")
    print(f"wheel_only={cfg.wheel_only}")
    print(f"allow_base_motion={cfg.allow_base_motion}")
    print(f"wheel_only_forced={cfg.wheel_only_forced}")
    print(f"seed={cfg.seed}")
    print(
        f"hardware_realistic={cfg.hardware_realistic} control_hz={cfg.control_hz:.1f} "
        f"delay_steps={cfg.control_delay_steps} wheel_ticks={cfg.wheel_encoder_ticks_per_rev}"
    )
    print(f"Disturbance: magnitude={cfg.disturbance_magnitude}, interval={cfg.disturbance_interval}")
    print(f"base_integrator_enabled={cfg.base_integrator_enabled}")
    print(f"Gravity z: {model.opt.gravity[2]}")
    print(f"Delta-u limits: {cfg.max_du}")
    print(f"Absolute-u limits: {cfg.max_u}")
    print(
        "Wheel motor model: "
        f"KV={cfg.wheel_motor_kv_rpm_per_v:.1f}RPM/V "
        f"R={cfg.wheel_motor_resistance_ohm:.3f}ohm "
        f"Ilim={cfg.wheel_current_limit_a:.2f}A "
        f"Vbus={cfg.bus_voltage_v:.2f}V "
        f"gear={cfg.wheel_gear_ratio:.2f} "
        f"eta={cfg.drive_efficiency:.2f}"
    )
    print(f"Wheel torque limit (stall/current): {cfg.wheel_torque_limit_nm:.4f} Nm")
    print(f"Wheel motor limit enforced: {cfg.enforce_wheel_motor_limit}")
    print(
        "Velocity limits: "
        f"wheel={cfg.max_wheel_speed_rad_s:.2f}rad/s "
        f"tilt_rate={cfg.max_pitch_roll_rate_rad_s:.2f}rad/s "
        f"base_rate={cfg.max_base_speed_m_s:.2f}m/s"
    )
    print(f"Wheel torque derate starts at {cfg.wheel_torque_derate_start * 100.0:.1f}% of max wheel speed")
    print(
        "Wheel momentum manager: "
        f"thresh={cfg.wheel_momentum_thresh_frac * 100.0:.1f}% "
        f"k={cfg.wheel_momentum_k:.2f} "
        f"upright_k={cfg.wheel_momentum_upright_k:.2f}"
    )
    print(
        "Wheel budget policy: "
        f"budget={cfg.wheel_spin_budget_frac * 100.0:.1f}% "
        f"hard={cfg.wheel_spin_hard_frac * 100.0:.1f}% "
        f"budget_abs={cfg.wheel_spin_budget_abs_rad_s:.1f}rad/s "
        f"hard_abs={cfg.wheel_spin_hard_abs_rad_s:.1f}rad/s "
        f"base_bias={cfg.wheel_to_base_bias_gain:.2f}"
    )
    print(
        "High-spin latch: "
        f"exit={cfg.high_spin_exit_frac * 100.0:.1f}% of hard "
        f"min_counter={cfg.high_spin_counter_min_frac * 100.0:.1f}% "
        f"base_auth_min={cfg.high_spin_base_authority_min:.2f}"
    )
    print(
        "Phase hysteresis: "
        f"enter={np.degrees(cfg.hold_enter_angle_rad):.2f}deg/{cfg.hold_enter_rate_rad_s:.2f}rad/s "
        f"exit={np.degrees(cfg.hold_exit_angle_rad):.2f}deg/{cfg.hold_exit_rate_rad_s:.2f}rad/s"
    )
    print(f"Base torque derate starts at {cfg.base_torque_derate_start * 100.0:.1f}% of max base speed")
    print(
        f"Base stabilization: force_limit={cfg.base_force_soft_limit:.2f} "
        f"damping={cfg.base_damping_gain:.2f} centering={cfg.base_centering_gain:.2f}"
    )
    print(
        "Base authority gate: "
        f"deadband={np.degrees(cfg.base_tilt_deadband_rad):.2f}deg "
        f"full={np.degrees(cfg.base_tilt_full_authority_rad):.2f}deg"
    )
    print(f"Base command gain: {cfg.base_command_gain:.2f}")
    print(
        f"Base anti-sprint: pos_clip={cfg.base_centering_pos_clip_m:.2f}m "
        f"soft_speed={cfg.base_speed_soft_limit_frac * 100.0:.0f}%"
    )
    print(
        f"Base recenter: hold_radius={cfg.base_hold_radius_m:.2f}m "
        f"follow={cfg.base_ref_follow_rate_hz:.2f}Hz recenter={cfg.base_ref_recenter_rate_hz:.2f}Hz"
    )
    print(
        f"Base smoothers: authority_rate={cfg.base_authority_rate_per_s:.2f}/s "
        f"base_lpf={cfg.base_command_lpf_hz:.2f}Hz upright_du_scale={cfg.upright_base_du_scale:.2f}"
    )
    print(
        f"Base PD: pitch(kp={cfg.base_pitch_kp:.1f}, kd={cfg.base_pitch_kd:.1f}) "
        f"roll(kp={cfg.base_roll_kp:.1f}, kd={cfg.base_roll_kd:.1f})"
    )
    print(
        f"Wheel-only PD: kp={cfg.wheel_only_pitch_kp:.1f} "
        f"kd={cfg.wheel_only_pitch_kd:.1f} ki={cfg.wheel_only_pitch_ki:.1f} "
        f"kw={cfg.wheel_only_wheel_rate_kd:.3f} "
        f"u={cfg.wheel_only_max_u:.1f} du={cfg.wheel_only_max_du:.1f}"
    )
    print(f"Wheel-rate lsb @ control_dt={control_dt:.6f}s: {wheel_lsb:.6e} rad/s")
    print(f"XML ctrlrange low: {XML_CTRL_LOW}")
    print(f"XML ctrlrange high: {XML_CTRL_HIGH}")
    print(f"Initial Y-direction tilt (roll): {args.initial_y_tilt_deg:.2f} deg")
    print(
        "Scripted push: "
        f"body={args.push_body} F=({args.push_x:.2f}, {args.push_y:.2f})N "
        f"start={args.push_start_s:.2f}s duration={args.push_duration_s:.2f}s"
    )
    if cfg.hardware_safe:
        print("HARDWARE-SAFE profile active: conservative torque/slew/speed limits enabled.")
    if cfg.real_hardware_profile:
        base_msg = "enabled" if cfg.allow_base_motion else "disabled (unlock required)"
        print(f"REAL-HARDWARE profile active: strict bring-up limits + forced stop_on_crash + base motion {base_msg}.")

    reset_state(model, data, ids.q_pitch, ids.q_roll, pitch_eq=0.0, roll_eq=initial_roll_rad)
    upright_blend = 0.0
    despin_gain = 0.25
    rng = np.random.default_rng(cfg.seed)

    queue_len = 1 if not cfg.hardware_realistic else cfg.control_delay_steps + 1
    (
        x_est,
        u_applied,
        u_eff_applied,
        base_int,
        wheel_pitch_int,
        base_ref,
        base_authority_state,
        u_base_smooth,
        balance_phase,
        recovery_time_s,
        high_spin_active,
        cmd_queue,
    ) = reset_controller_buffers(NX, NU, queue_len)

    sat_hits = np.zeros(NU, dtype=int)
    du_hits = np.zeros(NU, dtype=int)
    xml_limit_margin_hits = np.zeros(NU, dtype=int)
    speed_limit_hits = np.zeros(5, dtype=int)  # [wheel, pitch, roll, base_x, base_y]
    derate_hits = np.zeros(3, dtype=int)  # [wheel, base_x, base_y]
    step_count = 0
    control_updates = 0
    crash_count = 0
    max_pitch = 0.0
    max_roll = 0.0
    phase_switch_count = 0
    hold_steps = 0
    wheel_over_budget_count = 0
    wheel_over_hard_count = 0
    high_spin_steps = 0
    prev_script_force = np.zeros(3, dtype=float)
    control_terms_writer = None
    control_terms_file = None
    trace_events_writer = None
    trace_events_file = None
    if cfg.log_control_terms:
        terms_path = (
            Path(cfg.control_terms_csv)
            if cfg.control_terms_csv
            else Path(__file__).with_name("results") / f"control_terms_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        terms_path.parent.mkdir(parents=True, exist_ok=True)
        control_terms_file = terms_path.open("w", newline="", encoding="utf-8")
        control_terms_writer = csv.DictWriter(
            control_terms_file,
            fieldnames=[
                "step",
                "sim_time_s",
                "controller_family",
                "balance_phase",
                "term_lqr_core_rw",
                "term_lqr_core_bx",
                "term_lqr_core_by",
                "term_roll_stability_rw",
                "term_roll_stability_bx",
                "term_roll_stability_by",
                "term_pitch_stability_rw",
                "term_pitch_stability_bx",
                "term_pitch_stability_by",
                "term_despin_rw",
                "term_despin_bx",
                "term_despin_by",
                "term_base_hold_rw",
                "term_base_hold_bx",
                "term_base_hold_by",
                "term_safety_shaping_rw",
                "term_safety_shaping_bx",
                "term_safety_shaping_by",
                "u_cmd_rw",
                "u_cmd_bx",
                "u_cmd_by",
            ],
        )
        control_terms_writer.writeheader()
    if cfg.trace_events_csv:
        trace_path = Path(cfg.trace_events_csv)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_events_file = trace_path.open("w", newline="", encoding="utf-8")
        trace_events_writer = csv.DictWriter(
            trace_events_file,
            fieldnames=[
                "step",
                "sim_time_s",
                "event",
                "controller_family",
                "balance_phase",
                "pitch",
                "roll",
                "pitch_rate",
                "roll_rate",
                "wheel_rate",
                "base_x",
                "base_y",
                "u_rw",
                "u_bx",
                "u_by",
            ],
        )
        trace_events_writer.writeheader()

    # 5) Closed-loop runtime: estimate -> control -> clamp -> simulate -> render.
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_count += 1
            # Pull GUI events/perturbations before control+step so dragging is applied immediately.
            viewer.sync()
            if np.any(prev_script_force):
                data.xfrc_applied[push_body_id, :3] -= prev_script_force
                prev_script_force[:] = 0.0
            if cfg.wheel_only:
                enforce_wheel_only_constraints(model, data, ids)

            x_true = get_true_state(data, ids)

            if not np.all(np.isfinite(x_true)):
                print(f"\nNumerical instability at step {step_count}; resetting state.")
                reset_state(model, data, ids.q_pitch, ids.q_roll, pitch_eq=0.0, roll_eq=initial_roll_rad)
                (
                    x_est,
                    u_applied,
                    u_eff_applied,
                    base_int,
                    wheel_pitch_int,
                    base_ref,
                    base_authority_state,
                    u_base_smooth,
                    balance_phase,
                    recovery_time_s,
                    high_spin_active,
                    cmd_queue,
                ) = reset_controller_buffers(NX, NU, queue_len)
                continue

            x_pred = A @ x_est + B @ u_eff_applied
            x_est = x_pred

            if step_count % control_steps == 0:
                control_updates += 1
                x_est = estimator_measurement_update(cfg, x_true, x_pred, C, L, rng, wheel_lsb)
                angle_mag = max(abs(float(x_true[0])), abs(float(x_true[1])))
                rate_mag = max(abs(float(x_true[2])), abs(float(x_true[3])))
                prev_phase = balance_phase
                if balance_phase == "recovery":
                    if angle_mag < cfg.hold_enter_angle_rad and rate_mag < cfg.hold_enter_rate_rad_s:
                        balance_phase = "hold"
                        recovery_time_s = 0.0
                else:
                    if angle_mag > cfg.hold_exit_angle_rad or rate_mag > cfg.hold_exit_rate_rad_s:
                        balance_phase = "recovery"
                        recovery_time_s = 0.0
                if balance_phase != prev_phase:
                    phase_switch_count += 1
                if balance_phase == "recovery":
                    recovery_time_s += control_dt
                (
                    u_cmd,
                    base_int,
                    base_ref,
                    base_authority_state,
                    u_base_smooth,
                    wheel_pitch_int,
                    rw_u_limit,
                    wheel_over_budget,
                    wheel_over_hard,
                    high_spin_active,
                    control_terms,
                ) = compute_control_command(
                    cfg=cfg,
                    x_est=x_est,
                    x_true=x_true,
                    u_eff_applied=u_eff_applied,
                    base_int=base_int,
                    base_ref=base_ref,
                    base_authority_state=base_authority_state,
                    u_base_smooth=u_base_smooth,
                    wheel_pitch_int=wheel_pitch_int,
                    balance_phase=balance_phase,
                    recovery_time_s=recovery_time_s,
                    high_spin_active=high_spin_active,
                    control_dt=control_dt,
                    K_du=K_du,
                    K_wheel_only=K_wheel_only,
                    K_paper_pitch=K_paper_pitch,
                    du_hits=du_hits,
                    sat_hits=sat_hits,
                )
                wheel_over_budget_count += int(wheel_over_budget)
                wheel_over_hard_count += int(wheel_over_hard)
                u_cmd, upright_blend = apply_upright_postprocess(
                    cfg=cfg,
                    u_cmd=u_cmd,
                    x_est=x_est,
                    x_true=x_true,
                    upright_blend=upright_blend,
                    balance_phase=balance_phase,
                    high_spin_active=high_spin_active,
                    despin_gain=despin_gain,
                    rw_u_limit=rw_u_limit,
                )
                xml_limit_margin_hits += ((u_cmd < XML_CTRL_LOW) | (u_cmd > XML_CTRL_HIGH)).astype(int)
                if control_terms_writer is not None:
                    control_terms_writer.writerow(
                        {
                            "step": step_count,
                            "sim_time_s": float(data.time),
                            "controller_family": cfg.controller_family,
                            "balance_phase": balance_phase,
                            "term_lqr_core_rw": float(control_terms["term_lqr_core"][0]),
                            "term_lqr_core_bx": float(control_terms["term_lqr_core"][1]),
                            "term_lqr_core_by": float(control_terms["term_lqr_core"][2]),
                            "term_roll_stability_rw": float(control_terms["term_roll_stability"][0]),
                            "term_roll_stability_bx": float(control_terms["term_roll_stability"][1]),
                            "term_roll_stability_by": float(control_terms["term_roll_stability"][2]),
                            "term_pitch_stability_rw": float(control_terms["term_pitch_stability"][0]),
                            "term_pitch_stability_bx": float(control_terms["term_pitch_stability"][1]),
                            "term_pitch_stability_by": float(control_terms["term_pitch_stability"][2]),
                            "term_despin_rw": float(control_terms["term_despin"][0]),
                            "term_despin_bx": float(control_terms["term_despin"][1]),
                            "term_despin_by": float(control_terms["term_despin"][2]),
                            "term_base_hold_rw": float(control_terms["term_base_hold"][0]),
                            "term_base_hold_bx": float(control_terms["term_base_hold"][1]),
                            "term_base_hold_by": float(control_terms["term_base_hold"][2]),
                            "term_safety_shaping_rw": float(control_terms["term_safety_shaping"][0]),
                            "term_safety_shaping_bx": float(control_terms["term_safety_shaping"][1]),
                            "term_safety_shaping_by": float(control_terms["term_safety_shaping"][2]),
                            "u_cmd_rw": float(u_cmd[0]),
                            "u_cmd_bx": float(u_cmd[1]),
                            "u_cmd_by": float(u_cmd[2]),
                        }
                    )
                if trace_events_writer is not None:
                    trace_events_writer.writerow(
                        {
                            "step": step_count,
                            "sim_time_s": float(data.time),
                            "event": "control_update",
                            "controller_family": cfg.controller_family,
                            "balance_phase": balance_phase,
                            "pitch": float(x_true[0]),
                            "roll": float(x_true[1]),
                            "pitch_rate": float(x_true[2]),
                            "roll_rate": float(x_true[3]),
                            "wheel_rate": float(x_true[4]),
                            "base_x": float(x_true[5]),
                            "base_y": float(x_true[6]),
                            "u_rw": float(u_cmd[0]),
                            "u_bx": float(u_cmd[1]),
                            "u_by": float(u_cmd[2]),
                        }
                    )
                u_applied = apply_control_delay(cfg, cmd_queue, u_cmd)
            if balance_phase == "hold":
                hold_steps += 1
            if high_spin_active:
                high_spin_steps += 1

            drag_control_scale = 1.0
            if args.drag_assist:
                if np.any(np.abs(data.xfrc_applied[:, :3]) > 1e-9) or np.any(np.abs(data.xfrc_applied[:, 3:]) > 1e-9):
                    drag_control_scale = 0.2
            u_drag = u_applied * drag_control_scale

            data.ctrl[:] = 0.0
            wheel_speed = float(data.qvel[ids.v_rw])
            wheel_cmd = wheel_command_with_limits(cfg, wheel_speed, float(u_drag[0]))
            derate_hits[0] += int(abs(wheel_cmd - u_drag[0]) > 1e-9)
            data.ctrl[ids.aid_rw] = wheel_cmd

            if cfg.allow_base_motion:
                base_x_cmd, base_y_cmd = base_commands_with_limits(
                    cfg=cfg,
                    base_x_speed=float(data.qvel[ids.v_base_x]),
                    base_y_speed=float(data.qvel[ids.v_base_y]),
                    base_x=float(data.qpos[ids.q_base_x]),
                    base_y=float(data.qpos[ids.q_base_y]),
                    base_x_request=float(u_drag[1]),
                    base_y_request=float(u_drag[2]),
                )
            else:
                base_x_cmd = 0.0
                base_y_cmd = 0.0
            derate_hits[1] += int(abs(base_x_cmd - u_drag[1]) > 1e-9)
            derate_hits[2] += int(abs(base_y_cmd - u_drag[2]) > 1e-9)
            data.ctrl[ids.aid_base_x] = base_x_cmd
            data.ctrl[ids.aid_base_y] = base_y_cmd
            u_eff_applied[:] = [wheel_cmd, base_x_cmd, base_y_cmd]
            if args.planar_perturb:
                data.xfrc_applied[:, 2] = 0.0
            if args.push_duration_s > 0.0 and args.push_start_s <= data.time < push_end_s:
                data.xfrc_applied[push_body_id, :3] += push_force_world
                prev_script_force[:] = push_force_world

            mujoco.mj_step(model, data)
            if cfg.wheel_only:
                enforce_wheel_only_constraints(model, data, ids)
            speed_limit_hits[0] += int(abs(data.qvel[ids.v_rw]) > cfg.max_wheel_speed_rad_s)
            speed_limit_hits[1] += int(abs(data.qvel[ids.v_pitch]) > cfg.max_pitch_roll_rate_rad_s)
            speed_limit_hits[2] += int(abs(data.qvel[ids.v_roll]) > cfg.max_pitch_roll_rate_rad_s)
            speed_limit_hits[3] += int(abs(data.qvel[ids.v_base_x]) > cfg.max_base_speed_m_s)
            speed_limit_hits[4] += int(abs(data.qvel[ids.v_base_y]) > cfg.max_base_speed_m_s)
            # Push latest state to viewer.
            viewer.sync()

            pitch = float(data.qpos[ids.q_pitch])
            roll = float(data.qpos[ids.q_roll])
            base_x = float(data.qpos[ids.q_base_x])
            base_y = float(data.qpos[ids.q_base_y])
            max_pitch = max(max_pitch, abs(pitch))
            max_roll = max(max_roll, abs(roll))

            if step_count % 100 == 0:
                print(
                    f"Step {step_count}: pitch={np.degrees(pitch):6.2f}deg roll={np.degrees(roll):6.2f}deg "
                    f"x={base_x:7.3f} y={base_y:7.3f} u_rw={u_eff_applied[0]:8.1f} "
                    f"u_bx={u_eff_applied[1]:7.2f} u_by={u_eff_applied[2]:7.2f}"
                )

            if abs(pitch) >= cfg.crash_angle_rad or abs(roll) >= cfg.crash_angle_rad:
                crash_count += 1
                if trace_events_writer is not None:
                    trace_events_writer.writerow(
                        {
                            "step": step_count,
                            "sim_time_s": float(data.time),
                            "event": "crash",
                            "controller_family": cfg.controller_family,
                            "balance_phase": balance_phase,
                            "pitch": float(pitch),
                            "roll": float(roll),
                            "pitch_rate": float(data.qvel[ids.v_pitch]),
                            "roll_rate": float(data.qvel[ids.v_roll]),
                            "wheel_rate": float(data.qvel[ids.v_rw]),
                            "base_x": float(data.qpos[ids.q_base_x]),
                            "base_y": float(data.qpos[ids.q_base_y]),
                            "u_rw": float(u_eff_applied[0]),
                            "u_bx": float(u_eff_applied[1]),
                            "u_by": float(u_eff_applied[2]),
                        }
                    )
                print(
                    f"\nCRASH #{crash_count} at step {step_count}: "
                    f"pitch={np.degrees(pitch):.2f}deg roll={np.degrees(roll):.2f}deg"
                )
                if cfg.stop_on_crash:
                    break
                reset_state(model, data, ids.q_pitch, ids.q_roll, pitch_eq=0.0, roll_eq=initial_roll_rad)
                (
                    x_est,
                    u_applied,
                    u_eff_applied,
                    base_int,
                    wheel_pitch_int,
                    base_ref,
                    base_authority_state,
                    u_base_smooth,
                    balance_phase,
                    recovery_time_s,
                    high_spin_active,
                    cmd_queue,
                ) = reset_controller_buffers(NX, NU, queue_len)
                continue

    if control_terms_file is not None:
        control_terms_file.close()
    if trace_events_file is not None:
        trace_events_file.close()

    print("\n=== SIMULATION ENDED ===")
    print(f"Total steps: {step_count}")
    print(f"Control updates: {control_updates}")
    print(f"Crash count: {crash_count}")
    print(f"Max |pitch|: {np.degrees(max_pitch):.2f}deg")
    print(f"Max |roll|: {np.degrees(max_roll):.2f}deg")
    denom = max(control_updates, 1)
    print(f"Abs-limit hit rate [rw,bx,by]: {(sat_hits / denom)}")
    print(f"Delta-u clip rate [rw,bx,by]: {(du_hits / denom)}")
    print(f"XML margin violation rate [rw,bx,by]: {(xml_limit_margin_hits / denom)}")
    print(f"Speed-over-limit counts [wheel,pitch,roll,base_x,base_y]: {speed_limit_hits}")
    print(f"Torque-derate activation counts [wheel,base_x,base_y]: {derate_hits}")
    print(f"Phase switch count: {phase_switch_count}")
    print(f"Hold phase ratio: {hold_steps / max(step_count, 1):.3f}")
    print(f"Wheel over-budget count: {wheel_over_budget_count}")
    print(f"Wheel over-hard count: {wheel_over_hard_count}")
    print(f"High-spin active ratio: {high_spin_steps / max(step_count, 1):.3f}")


if __name__ == "__main__":
    main()
