import argparse
from dataclasses import dataclass

import numpy as np


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
    residual_model_path: str | None
    residual_scale: float
    residual_max_abs_u: np.ndarray
    residual_gate_tilt_rad: float
    residual_gate_rate_rad_s: float
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
    payload_mass_kg: float
    payload_support_radius_m: float
    payload_com_fail_steps: int
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
    use_mpc: bool
    mpc_horizon: int
    mpc_q_angles: float
    mpc_q_rates: float
    mpc_q_position: float
    mpc_r_control: float
    mpc_com_constraint_radius_m: float
    mpc_verbose: bool




def parse_args(argv=None):
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
        "--residual-model",
        type=str,
        default=None,
        help="Optional PyTorch residual model checkpoint path (.pt/.pth).",
    )
    parser.add_argument(
        "--residual-scale",
        type=float,
        default=0.0,
        help="Global multiplier for residual command output (0 disables residual path).",
    )
    parser.add_argument(
        "--residual-max-rw",
        type=float,
        default=6.0,
        help="Max residual wheel command magnitude (same units as u_rw).",
    )
    parser.add_argument(
        "--residual-max-bx",
        type=float,
        default=1.0,
        help="Max residual base-x command magnitude.",
    )
    parser.add_argument(
        "--residual-max-by",
        type=float,
        default=1.0,
        help="Max residual base-y command magnitude.",
    )
    parser.add_argument(
        "--residual-gate-tilt-deg",
        type=float,
        default=8.0,
        help="Disable residual output when |pitch|/|roll| exceeds this tilt (deg).",
    )
    parser.add_argument(
        "--residual-gate-rate",
        type=float,
        default=3.0,
        help="Disable residual output when |pitch_rate|/|roll_rate| exceeds this value (rad/s).",
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
    parser.add_argument(
        "--payload-mass",
        type=float,
        default=0.0,
        help="Physical payload mass attached on top of the stick (kg).",
    )
    parser.add_argument(
        "--payload-support-radius-m",
        type=float,
        default=0.145,
        help="Support radius for COM-over-support failure check (m).",
    )
    parser.add_argument(
        "--payload-com-fail-steps",
        type=int,
        default=15,
        help="Consecutive steps beyond support radius required to trigger overload failure.",
    )
    parser.add_argument("--disturbance-mag", type=float, default=None)
    parser.add_argument("--disturbance-interval", type=int, default=None)
    parser.add_argument("--control-hz", type=float, default=250.0)
    parser.add_argument("--control-delay-steps", type=int, default=1)
    parser.add_argument(
        "--use-mpc",
        action="store_true",
        help="Use Model Predictive Control with hard COM position constraints instead of LQR.",
    )
    parser.add_argument(
        "--mpc-horizon",
        type=int,
        default=25,
        help="MPC prediction horizon in steps (25 steps = 100ms at 250Hz).",
    )
    parser.add_argument(
        "--mpc-q-angles",
        type=float,
        default=200.0,
        help="MPC cost weight for pitch/roll angle errors.",
    )
    parser.add_argument(
        "--mpc-q-rates",
        type=float,
        default=100.0,
        help="MPC cost weight for pitch/roll rate errors.",
    )
    parser.add_argument(
        "--mpc-q-position",
        type=float,
        default=50.0,
        help="MPC cost weight for base position errors.",
    )
    parser.add_argument(
        "--mpc-r-control",
        type=float,
        default=0.5,
        help="MPC cost weight for control effort.",
    )
    parser.add_argument(
        "--mpc-verbose",
        action="store_true",
        help="Print MPC solver diagnostics.",
    )
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
        choices=["stick", "base_y", "base_x", "payload"],
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
    return parser.parse_args(argv)


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
        base_damping_gain = 5.0
        base_centering_gain = 1.0
        base_tilt_deadband_deg = 0.4
        base_tilt_full_authority_deg = 1.8
        base_command_gain = 0.70
        base_centering_pos_clip_m = 0.25
        base_speed_soft_limit_frac = 0.55
        base_hold_radius_m = 0.15
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
        base_damping_gain = 6.0
        base_centering_gain = 1.5
        base_tilt_deadband_deg = 0.5
        base_tilt_full_authority_deg = 2.0
        base_command_gain = 0.60
        base_centering_pos_clip_m = 0.20
        base_speed_soft_limit_frac = 0.70
        base_hold_radius_m = 0.15
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

    residual_scale = float(max(getattr(args, "residual_scale", 0.0), 0.0))
    residual_max_abs_u = np.array(
        [
            max(float(getattr(args, "residual_max_rw", 0.0)), 0.0),
            max(float(getattr(args, "residual_max_bx", 0.0)), 0.0),
            max(float(getattr(args, "residual_max_by", 0.0)), 0.0),
        ],
        dtype=float,
    )
    residual_max_abs_u = np.minimum(residual_max_abs_u, max_u)
    residual_gate_tilt_rad = float(np.radians(max(float(getattr(args, "residual_gate_tilt_deg", 0.0)), 0.0)))
    residual_gate_rate_rad_s = float(max(float(getattr(args, "residual_gate_rate", 0.0)), 0.0))
    payload_mass_kg = float(max(getattr(args, "payload_mass", 0.0), 0.0))
    payload_support_radius_m = float(max(getattr(args, "payload_support_radius_m", 0.145), 0.01))
    payload_com_fail_steps = int(max(getattr(args, "payload_com_fail_steps", 15), 1))

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
        residual_model_path=(str(getattr(args, "residual_model", "")) if getattr(args, "residual_model", None) else None),
        residual_scale=residual_scale,
        residual_max_abs_u=residual_max_abs_u,
        residual_gate_tilt_rad=residual_gate_tilt_rad,
        residual_gate_rate_rad_s=residual_gate_rate_rad_s,
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
        payload_mass_kg=payload_mass_kg,
        payload_support_radius_m=payload_support_radius_m,
        payload_com_fail_steps=payload_com_fail_steps,
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
        use_mpc=bool(getattr(args, "use_mpc", False)),
        mpc_horizon=int(max(getattr(args, "mpc_horizon", 25), 5)),
        mpc_q_angles=float(max(getattr(args, "mpc_q_angles", 200.0), 1.0)),
        mpc_q_rates=float(max(getattr(args, "mpc_q_rates", 100.0), 1.0)),
        mpc_q_position=float(max(getattr(args, "mpc_q_position", 50.0), 1.0)),
        mpc_r_control=float(max(getattr(args, "mpc_r_control", 0.5), 0.01)),
        mpc_com_constraint_radius_m=payload_support_radius_m,
        mpc_verbose=bool(getattr(args, "mpc_verbose", False)),
    )
