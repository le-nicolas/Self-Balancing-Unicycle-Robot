import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency in some environments
    yaml = None


DEFAULT_RUNTIME_CONFIG_PATH = Path(__file__).with_name("config.yaml")
TUNABLE_KEYS = {
    "qx_diag",
    "qu_diag",
    "r_du_diag",
    "max_u",
    "max_du",
    "ki_base",
    "u_bleed",
    "wheel_momentum_thresh_frac",
    "wheel_momentum_k",
    "wheel_momentum_upright_k",
    "wheel_spin_budget_frac",
    "wheel_spin_hard_frac",
    "wheel_spin_budget_abs_rad_s",
    "wheel_spin_hard_abs_rad_s",
    "high_spin_exit_frac",
    "high_spin_counter_min_frac",
    "high_spin_base_authority_min",
    "wheel_to_base_bias_gain",
    "recovery_wheel_despin_scale",
    "hold_wheel_despin_scale",
    "base_force_soft_limit",
    "base_damping_gain",
    "base_centering_gain",
    "base_tilt_deadband_deg",
    "base_tilt_full_authority_deg",
    "base_command_gain",
    "base_centering_pos_clip_m",
    "base_speed_soft_limit_frac",
    "base_hold_radius_m",
    "base_ref_follow_rate_hz",
    "base_ref_recenter_rate_hz",
    "base_authority_rate_per_s",
    "base_command_lpf_hz",
    "upright_base_du_scale",
    "base_pitch_kp",
    "base_pitch_kd",
    "base_roll_kp",
    "base_roll_kd",
    "wheel_only_pitch_kp",
    "wheel_only_pitch_kd",
    "wheel_only_pitch_ki",
    "wheel_only_wheel_rate_kd",
    "wheel_only_max_u",
    "wheel_only_max_du",
    "online_id_forgetting",
    "online_id_init_cov",
    "online_id_min_excitation",
    "online_id_recompute_every",
    "online_id_min_updates",
    "online_id_gravity_scale_min",
    "online_id_gravity_scale_max",
    "online_id_inertia_inv_scale_min",
    "online_id_inertia_inv_scale_max",
    "online_id_scale_rate_per_s",
    "online_id_gain_blend_alpha",
    "online_id_gain_max_delta",
    "online_id_innovation_clip",
}


def _merge_tunable_block(target: dict[str, Any], block: Any, label: str) -> None:
    if block is None:
        return
    if not isinstance(block, dict):
        raise ValueError(f"Runtime config block '{label}' must be a mapping.")
    for key, value in block.items():
        if key in TUNABLE_KEYS:
            target[key] = value


def _parse_tuning_file(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return {}
    if yaml is None:
        raise RuntimeError(
            f"Runtime tuning file '{path}' requires PyYAML. Install with: pip install pyyaml"
        )
    loaded = yaml.safe_load(text)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Runtime tuning file '{path}' must define a mapping at top level.")
    return loaded


def _resolve_runtime_tuning_overrides(
    config_path_arg: str | None,
    *,
    mode: str,
    preset: str,
    stability_profile: str,
) -> dict[str, Any]:
    cfg_path = Path(config_path_arg).expanduser() if config_path_arg else DEFAULT_RUNTIME_CONFIG_PATH
    if not cfg_path.exists():
        return {}
    data = _parse_tuning_file(cfg_path)
    merged: dict[str, Any] = {}

    # Backward-compatible: allow top-level direct tunables.
    _merge_tunable_block(merged, data, "top_level")
    _merge_tunable_block(merged, data.get("global"), "global")

    mode_map = data.get("mode")
    if isinstance(mode_map, dict):
        _merge_tunable_block(merged, mode_map.get(mode), f"mode.{mode}")

    preset_map = data.get("preset")
    if isinstance(preset_map, dict):
        _merge_tunable_block(merged, preset_map.get(preset), f"preset.{preset}")

    stability_map = data.get("stability_profile")
    if isinstance(stability_map, dict):
        _merge_tunable_block(merged, stability_map.get(stability_profile), f"stability_profile.{stability_profile}")

    return merged


def _override_float(
    overrides: dict[str, Any],
    key: str,
    current: float,
) -> float:
    if key not in overrides:
        return float(current)
    value = overrides[key]
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Runtime tuning key '{key}' must be numeric, got {value!r}.") from exc


def _override_array(
    overrides: dict[str, Any],
    key: str,
    current: np.ndarray,
) -> np.ndarray:
    if key not in overrides:
        return current
    value = np.asarray(overrides[key], dtype=float)
    if value.shape != current.shape:
        raise ValueError(
            f"Runtime tuning key '{key}' must have shape {current.shape}, got {value.shape}."
        )
    return value.astype(float, copy=False)


def _override_diag(
    overrides: dict[str, Any],
    key: str,
    size: int,
    current: np.ndarray,
) -> np.ndarray:
    if key not in overrides:
        return current
    diag = np.asarray(overrides[key], dtype=float).reshape(-1)
    if diag.shape[0] != size:
        raise ValueError(
            f"Runtime tuning key '{key}' must have {size} entries, got {diag.shape[0]}."
        )
    return np.diag(diag.astype(float, copy=False))


def _override_int(
    overrides: dict[str, Any],
    key: str,
    current: int,
) -> int:
    if key not in overrides:
        return int(current)
    value = overrides[key]
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Runtime tuning key '{key}' must be integer-compatible, got {value!r}.") from exc


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
    lock_root_attitude: bool
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
    sensor_source: str
    sensor_hz: float
    sensor_delay_steps: int
    imu_angle_bias_rw_std_rad_sqrt_s: float
    imu_rate_bias_rw_std_rad_s_sqrt_s: float
    wheel_encoder_bias_rw_std_rad_s_sqrt_s: float
    base_encoder_pos_bias_rw_std_m_sqrt_s: float
    base_encoder_vel_bias_rw_std_m_s_sqrt_s: float
    imu_angle_clip_rad: float
    imu_rate_clip_rad_s: float
    wheel_rate_clip_rad_s: float
    base_pos_clip_m: float
    base_vel_clip_m_s: float
    imu_angle_lpf_hz: float
    imu_rate_lpf_hz: float
    wheel_rate_lpf_hz: float
    base_pos_lpf_hz: float
    base_vel_lpf_hz: float
    residual_model_path: str | None
    residual_scale: float
    residual_max_abs_u: np.ndarray
    residual_gate_tilt_rad: float
    residual_gate_rate_rad_s: float
    dob_enabled: bool
    dob_gain: float
    dob_leak_per_s: float
    dob_max_abs_u: np.ndarray
    gain_schedule_enabled: bool
    gain_schedule_min: float
    gain_schedule_max: float
    gain_schedule_disturbance_ref: float
    gain_schedule_rate_per_s: float
    gain_schedule_weights: np.ndarray
    max_u: np.ndarray
    max_du: np.ndarray
    rw_emergency_du_enabled: bool
    rw_emergency_pitch_rad: float
    rw_emergency_du_scale: float
    disturbance_magnitude: float
    disturbance_interval: int
    qx: np.ndarray
    qu: np.ndarray
    r_du: np.ndarray
    ki_base: float
    base_integrator_enabled: bool
    u_bleed: float
    crash_angle_rad: float
    crash_divergence_gate_enabled: bool
    crash_recovery_window_steps: int
    payload_mass_kg: float
    payload_support_radius_m: float
    payload_com_fail_steps: int
    x_ref: float
    y_ref: float
    linearize_pitch_rad: float
    linearize_roll_rad: float
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
    hold_base_x_centering_gain: float
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
    online_id_enabled: bool
    online_id_forgetting: float
    online_id_init_cov: float
    online_id_min_excitation: float
    online_id_recompute_every: int
    online_id_min_updates: int
    online_id_gravity_scale_min: float
    online_id_gravity_scale_max: float
    online_id_inertia_inv_scale_min: float
    online_id_inertia_inv_scale_max: float
    online_id_scale_rate_per_s: float
    online_id_gain_blend_alpha: float
    online_id_gain_max_delta: float
    online_id_innovation_clip: float
    online_id_verbose: bool
    use_mpc: bool
    mpc_horizon: int
    mpc_q_angles: float
    mpc_q_rates: float
    mpc_q_position: float
    mpc_r_control: float
    mpc_terminal_weight: float
    mpc_target_rate_gain: float
    mpc_terminal_rate_gain: float
    mpc_target_rate_clip_rad_s: float
    mpc_com_constraint_radius_m: float
    mpc_pitch_i_gain: float
    mpc_pitch_i_clamp: float
    mpc_pitch_i_deadband_rad: float
    mpc_pitch_i_leak_per_s: float
    mpc_pitch_guard_angle_frac: float
    mpc_pitch_guard_rate_entry_rad_s: float
    mpc_pitch_guard_kp: float
    mpc_pitch_guard_kd: float
    mpc_pitch_guard_max_frac: float
    mpc_roll_i_gain: float
    mpc_roll_i_clamp: float
    mpc_roll_i_deadband_rad: float
    mpc_roll_i_leak_per_s: float
    mpc_verbose: bool




def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="MuJoCo wheel-on-stick viewer controller.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Optional runtime tuning YAML file. "
            f"If omitted, auto-loads {DEFAULT_RUNTIME_CONFIG_PATH.as_posix()} when present."
        ),
    )
    parser.add_argument(
        "--controller-family",
        choices=[
            "current",
            "current_dob",
            "hybrid_modern",
            "paper_split_baseline",
        ],
        default="current",
        help="Controller family selector. 'current_dob' uses the current stack with DOB enabled.",
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
        help=(
            "Optional PyTorch residual checkpoint (.pt/.pth). "
            "Expected feature order: x_est[9], u_eff_applied[3], u_nominal[3]."
        ),
    )
    parser.add_argument(
        "--residual-scale",
        type=float,
        default=0.0,
        help=(
            "Multiplier on residual output after model output_scale calibration. "
            "0 disables residual path; 0.20 means 20%% residual authority before max-abs clipping."
        ),
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
        "--enable-dob",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable adaptive disturbance observer (DOb) feed-forward compensation.",
    )
    parser.add_argument(
        "--dob-gain",
        type=float,
        default=14.0,
        help="Observer adaptation gain (1/s) for disturbance estimate update.",
    )
    parser.add_argument(
        "--dob-cutoff-hz",
        type=float,
        default=0.0,
        help="If >0, sets DOB low-pass cutoff (Hz) and overrides --dob-gain via gain=2*pi*cutoff.",
    )
    parser.add_argument(
        "--dob-leak-per-s",
        type=float,
        default=0.6,
        help="Leak rate (1/s) to decay disturbance estimate in calm conditions.",
    )
    parser.add_argument(
        "--dob-max-rw",
        type=float,
        default=6.0,
        help="Max absolute DOb compensation magnitude on wheel channel.",
    )
    parser.add_argument(
        "--dob-max-bx",
        type=float,
        default=1.0,
        help="Max absolute DOb compensation magnitude on base-x channel.",
    )
    parser.add_argument(
        "--dob-max-by",
        type=float,
        default=1.0,
        help="Max absolute DOb compensation magnitude on base-y channel.",
    )
    parser.add_argument(
        "--enable-gain-scheduling",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable real-time gain scheduling from disturbance estimate magnitude.",
    )
    parser.add_argument(
        "--gain-sched-min",
        type=float,
        default=1.0,
        help="Minimum gain schedule multiplier.",
    )
    parser.add_argument(
        "--gain-sched-max",
        type=float,
        default=1.8,
        help="Maximum gain schedule multiplier under large disturbances.",
    )
    parser.add_argument(
        "--gain-sched-ref",
        type=float,
        default=2.0,
        help="Disturbance magnitude reference at which schedule approaches max.",
    )
    parser.add_argument(
        "--gain-sched-rate-per-s",
        type=float,
        default=3.0,
        help="Rate limit (1/s) for schedule multiplier changes.",
    )
    parser.add_argument(
        "--gain-sched-rw-weight",
        type=float,
        default=1.0,
        help="Disturbance weighting for wheel channel in schedule magnitude.",
    )
    parser.add_argument(
        "--gain-sched-bx-weight",
        type=float,
        default=0.6,
        help="Disturbance weighting for base-x channel in schedule magnitude.",
    )
    parser.add_argument(
        "--gain-sched-by-weight",
        type=float,
        default=0.6,
        help="Disturbance weighting for base-y channel in schedule magnitude.",
    )
    parser.add_argument(
        "--enable-online-id",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable online system identification + adaptive gain-scheduled LQR updates.",
    )
    parser.add_argument(
        "--online-id-forgetting",
        type=float,
        default=0.995,
        help="RLS forgetting factor (close to 1.0 keeps longer memory).",
    )
    parser.add_argument(
        "--online-id-init-cov",
        type=float,
        default=500.0,
        help="Initial covariance for RLS parameter uncertainty.",
    )
    parser.add_argument(
        "--online-id-min-excitation",
        type=float,
        default=0.015,
        help="Minimum feature norm required before an RLS update is accepted.",
    )
    parser.add_argument(
        "--online-id-recompute-every",
        type=int,
        default=25,
        help="Recompute adaptive LQR gain every N control updates.",
    )
    parser.add_argument(
        "--online-id-min-updates",
        type=int,
        default=60,
        help="Minimum accepted RLS updates before enabling adaptive gain recompute.",
    )
    parser.add_argument(
        "--online-id-gravity-scale-min",
        type=float,
        default=0.55,
        help="Lower bound on estimated gravity-stiffness scaling.",
    )
    parser.add_argument(
        "--online-id-gravity-scale-max",
        type=float,
        default=1.80,
        help="Upper bound on estimated gravity-stiffness scaling.",
    )
    parser.add_argument(
        "--online-id-inertia-inv-scale-min",
        type=float,
        default=0.45,
        help="Lower bound on estimated inverse-inertia scaling.",
    )
    parser.add_argument(
        "--online-id-inertia-inv-scale-max",
        type=float,
        default=2.20,
        help="Upper bound on estimated inverse-inertia scaling.",
    )
    parser.add_argument(
        "--online-id-scale-rate-per-s",
        type=float,
        default=0.60,
        help="Rate limit for adaptive scale changes (1/s).",
    )
    parser.add_argument(
        "--online-id-gain-blend-alpha",
        type=float,
        default=0.18,
        help="Blend factor when applying newly solved adaptive gain matrix.",
    )
    parser.add_argument(
        "--online-id-gain-max-delta",
        type=float,
        default=0.35,
        help="Per-entry absolute clip for each adaptive gain matrix update step.",
    )
    parser.add_argument(
        "--online-id-innovation-clip",
        type=float,
        default=4.0,
        help="Absolute clip limit for RLS innovation term.",
    )
    parser.add_argument(
        "--online-id-verbose",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print adaptive-ID diagnostics periodically.",
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
        "--lock-root-attitude",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clamp free-joint base attitude to upright (enabled by default).",
    )
    parser.add_argument(
        "--enable-base-integrator",
        action="store_true",
        help="Enable base x/y integral action. Disabled by default to avoid drift with unobserved base states.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--initial-y-tilt-deg",
        type=float,
        default=0.0,
        help="Initial stick tilt in degrees in Y direction (implemented as roll for base_y response).",
    )
    parser.add_argument(
        "--linearize-pitch-deg",
        type=float,
        default=0.0,
        help="Pitch operating point (deg) used for dynamics linearization before LQR/MPC gain construction.",
    )
    parser.add_argument(
        "--linearize-roll-deg",
        type=float,
        default=0.0,
        help="Roll operating point (deg) used for dynamics linearization before LQR/MPC gain construction.",
    )
    parser.add_argument("--crash-angle-deg", type=float, default=25.0)
    parser.add_argument(
        "--crash-gate-divergence",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Require same-sign pitch_rate when |pitch| exceeds crash angle. "
            "If converging (opposite sign), apply recovery window before crash."
        ),
    )
    parser.add_argument(
        "--crash-recovery-steps",
        type=int,
        default=500,
        help="Recovery grace window (simulation steps) while over-angle but converging.",
    )
    parser.add_argument(
        "--rw-emergency-du",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable high-angle emergency boost on reaction-wheel delta-u limit.",
    )
    parser.add_argument(
        "--rw-emergency-pitch-deg",
        type=float,
        default=15.0,
        help="Pitch angle threshold (deg) for emergency reaction-wheel delta-u boost.",
    )
    parser.add_argument(
        "--rw-emergency-du-scale",
        type=float,
        default=1.5,
        help="Multiplier on reaction-wheel delta-u limit while emergency boost is active.",
    )
    parser.add_argument(
        "--hold-base-x-centering-gain",
        type=float,
        default=0.0,
        help="Extra base-x centering spring gain (N/m) applied only during hold phase.",
    )
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
        default=32,
        help="MPC prediction horizon in steps (32 steps = 128ms at 250Hz).",
    )
    parser.add_argument(
        "--mpc-q-angles",
        type=float,
        default=280.0,
        help="MPC cost weight for pitch/roll angle errors.",
    )
    parser.add_argument(
        "--mpc-q-rates",
        type=float,
        default=180.0,
        help="MPC cost weight for pitch/roll rate errors.",
    )
    parser.add_argument(
        "--mpc-q-position",
        type=float,
        default=30.0,
        help="MPC cost weight for base position errors.",
    )
    parser.add_argument(
        "--mpc-r-control",
        type=float,
        default=0.25,
        help="MPC cost weight for control effort.",
    )
    parser.add_argument(
        "--mpc-terminal-weight",
        type=float,
        default=8.0,
        help="Multiplier on terminal state cost matrix (Qf = weight * Q).",
    )
    parser.add_argument(
        "--mpc-target-rate-gain",
        type=float,
        default=4.0,
        help="Shaped MPC rate target gain: rate_ref = -gain * angle.",
    )
    parser.add_argument(
        "--mpc-terminal-rate-gain",
        type=float,
        default=7.0,
        help="Terminal shaped rate target gain at horizon end.",
    )
    parser.add_argument(
        "--mpc-target-rate-clip",
        type=float,
        default=4.0,
        help="Clamp for shaped pitch/roll rate targets (rad/s).",
    )
    parser.add_argument(
        "--mpc-pitch-i-gain",
        type=float,
        default=9.0,
        help="MPC pitch anti-drift integral gain added to wheel command.",
    )
    parser.add_argument(
        "--mpc-pitch-i-clamp",
        type=float,
        default=0.45,
        help="Clamp for MPC pitch integral state (rad*s).",
    )
    parser.add_argument(
        "--mpc-pitch-i-deadband-deg",
        type=float,
        default=0.15,
        help="Pitch error deadband for MPC pitch integral action (deg).",
    )
    parser.add_argument(
        "--mpc-pitch-i-leak-per-s",
        type=float,
        default=0.8,
        help="Leak rate for MPC pitch integral state while inside deadband.",
    )
    parser.add_argument(
        "--mpc-pitch-guard-angle-frac",
        type=float,
        default=0.45,
        help="Pitch rescue starts at this fraction of crash angle.",
    )
    parser.add_argument(
        "--mpc-pitch-guard-rate",
        type=float,
        default=1.2,
        help="Pitch-rate threshold (rad/s) to trigger rescue when moving away from upright.",
    )
    parser.add_argument(
        "--mpc-pitch-guard-kp",
        type=float,
        default=120.0,
        help="Pitch rescue proportional gain on wheel command.",
    )
    parser.add_argument(
        "--mpc-pitch-guard-kd",
        type=float,
        default=28.0,
        help="Pitch rescue derivative gain on wheel command.",
    )
    parser.add_argument(
        "--mpc-pitch-guard-max-frac",
        type=float,
        default=1.0,
        help="Max wheel command fraction available to pitch rescue blend.",
    )
    parser.add_argument(
        "--mpc-roll-i-gain",
        type=float,
        default=8.0,
        help="MPC roll anti-drift integral gain added to roll actuator command.",
    )
    parser.add_argument(
        "--mpc-roll-i-clamp",
        type=float,
        default=0.35,
        help="Clamp for MPC roll integral state (rad*s).",
    )
    parser.add_argument(
        "--mpc-roll-i-deadband-deg",
        type=float,
        default=0.12,
        help="Roll error deadband for MPC roll integral action (deg).",
    )
    parser.add_argument(
        "--mpc-roll-i-leak-per-s",
        type=float,
        default=0.8,
        help="Leak rate for MPC roll integral state while inside deadband.",
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
        "--sensor-source",
        choices=["auto", "mujoco", "direct"],
        default="auto",
        help="Measurement backend: MuJoCo sensordata, direct state, or auto fallback.",
    )
    parser.add_argument(
        "--sensor-hz",
        type=float,
        default=None,
        help="Sensor sample rate in Hz. If omitted, tracks control-hz.",
    )
    parser.add_argument(
        "--sensor-delay-steps",
        type=int,
        default=0,
        help="Additional measurement latency in control updates.",
    )
    parser.add_argument(
        "--imu-angle-bias-rw-deg",
        type=float,
        default=0.02,
        help="IMU angle-bias random walk std (deg/sqrt(s)).",
    )
    parser.add_argument(
        "--imu-rate-bias-rw",
        type=float,
        default=0.003,
        help="IMU rate-bias random walk std ((rad/s)/sqrt(s)).",
    )
    parser.add_argument(
        "--wheel-rate-bias-rw",
        type=float,
        default=0.002,
        help="Wheel-rate bias random walk std ((rad/s)/sqrt(s)).",
    )
    parser.add_argument(
        "--base-pos-bias-rw",
        type=float,
        default=3e-4,
        help="Base-position bias random walk std (m/sqrt(s)).",
    )
    parser.add_argument(
        "--base-vel-bias-rw",
        type=float,
        default=0.004,
        help="Base-velocity bias random walk std ((m/s)/sqrt(s)).",
    )
    parser.add_argument(
        "--imu-angle-clip-deg",
        type=float,
        default=85.0,
        help="IMU angle saturation limit (deg).",
    )
    parser.add_argument(
        "--imu-rate-clip",
        type=float,
        default=30.0,
        help="IMU rate saturation limit (rad/s).",
    )
    parser.add_argument(
        "--wheel-rate-clip",
        type=float,
        default=None,
        help="Wheel-rate sensor saturation limit (rad/s). Defaults to wheel speed limit.",
    )
    parser.add_argument(
        "--base-pos-clip",
        type=float,
        default=0.75,
        help="Base-position sensor saturation limit (m).",
    )
    parser.add_argument(
        "--base-vel-clip",
        type=float,
        default=6.0,
        help="Base-velocity sensor saturation limit (m/s).",
    )
    parser.add_argument(
        "--imu-angle-lpf-hz",
        type=float,
        default=45.0,
        help="First-order LPF cutoff for IMU angle channels (Hz).",
    )
    parser.add_argument(
        "--imu-rate-lpf-hz",
        type=float,
        default=70.0,
        help="First-order LPF cutoff for IMU rate channels (Hz).",
    )
    parser.add_argument(
        "--wheel-rate-lpf-hz",
        type=float,
        default=120.0,
        help="First-order LPF cutoff for wheel-rate channel (Hz).",
    )
    parser.add_argument(
        "--base-pos-lpf-hz",
        type=float,
        default=25.0,
        help="First-order LPF cutoff for base-position channels (Hz).",
    )
    parser.add_argument(
        "--base-vel-lpf-hz",
        type=float,
        default=40.0,
        help="First-order LPF cutoff for base-velocity channels (Hz).",
    )
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
    controller_family = str(getattr(args, "controller_family", "current"))
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
                240.0 * 1.8881272496398052,
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
        qx = np.diag([300.0, 90.0, 70.0, 50.0, 1.0, 220.0, 220.0, 520.0, 520.0])
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

    online_id_enabled = bool(getattr(args, "enable_online_id", False))
    online_id_forgetting = float(max(getattr(args, "online_id_forgetting", 0.995), 0.90))
    online_id_init_cov = float(max(getattr(args, "online_id_init_cov", 500.0), 1.0))
    online_id_min_excitation = float(max(getattr(args, "online_id_min_excitation", 0.015), 1e-6))
    online_id_recompute_every = int(max(getattr(args, "online_id_recompute_every", 25), 1))
    online_id_min_updates = int(max(getattr(args, "online_id_min_updates", 60), 1))
    online_id_gravity_scale_min = float(max(getattr(args, "online_id_gravity_scale_min", 0.55), 0.05))
    online_id_gravity_scale_max = float(
        max(getattr(args, "online_id_gravity_scale_max", 1.80), online_id_gravity_scale_min + 1e-6)
    )
    online_id_inertia_inv_scale_min = float(max(getattr(args, "online_id_inertia_inv_scale_min", 0.45), 0.05))
    online_id_inertia_inv_scale_max = float(
        max(getattr(args, "online_id_inertia_inv_scale_max", 2.20), online_id_inertia_inv_scale_min + 1e-6)
    )
    online_id_scale_rate_per_s = float(max(getattr(args, "online_id_scale_rate_per_s", 0.60), 0.0))
    online_id_gain_blend_alpha = float(np.clip(getattr(args, "online_id_gain_blend_alpha", 0.18), 0.0, 1.0))
    online_id_gain_max_delta = float(max(getattr(args, "online_id_gain_max_delta", 0.35), 0.0))
    online_id_innovation_clip = float(max(getattr(args, "online_id_innovation_clip", 4.0), 1e-6))
    online_id_verbose = bool(getattr(args, "online_id_verbose", False))

    runtime_overrides = _resolve_runtime_tuning_overrides(
        getattr(args, "config", None),
        mode=("smooth" if smooth_viewer else "robust"),
        preset=preset,
        stability_profile=stability_profile,
    )
    qx = _override_diag(runtime_overrides, "qx_diag", 9, qx)
    qu = _override_diag(runtime_overrides, "qu_diag", 3, qu)
    r_du = _override_diag(runtime_overrides, "r_du_diag", 3, r_du)
    max_u = _override_array(runtime_overrides, "max_u", max_u)
    max_du = _override_array(runtime_overrides, "max_du", max_du)
    ki_base = _override_float(runtime_overrides, "ki_base", ki_base)
    u_bleed = _override_float(runtime_overrides, "u_bleed", u_bleed)
    wheel_momentum_thresh_frac = _override_float(
        runtime_overrides, "wheel_momentum_thresh_frac", wheel_momentum_thresh_frac
    )
    wheel_momentum_k = _override_float(runtime_overrides, "wheel_momentum_k", wheel_momentum_k)
    wheel_momentum_upright_k = _override_float(
        runtime_overrides, "wheel_momentum_upright_k", wheel_momentum_upright_k
    )
    wheel_spin_budget_frac = _override_float(runtime_overrides, "wheel_spin_budget_frac", wheel_spin_budget_frac)
    wheel_spin_hard_frac = _override_float(runtime_overrides, "wheel_spin_hard_frac", wheel_spin_hard_frac)
    wheel_spin_budget_abs_rad_s = _override_float(
        runtime_overrides, "wheel_spin_budget_abs_rad_s", wheel_spin_budget_abs_rad_s
    )
    wheel_spin_hard_abs_rad_s = _override_float(runtime_overrides, "wheel_spin_hard_abs_rad_s", wheel_spin_hard_abs_rad_s)
    high_spin_exit_frac = _override_float(runtime_overrides, "high_spin_exit_frac", high_spin_exit_frac)
    high_spin_counter_min_frac = _override_float(
        runtime_overrides, "high_spin_counter_min_frac", high_spin_counter_min_frac
    )
    high_spin_base_authority_min = _override_float(
        runtime_overrides, "high_spin_base_authority_min", high_spin_base_authority_min
    )
    wheel_to_base_bias_gain = _override_float(runtime_overrides, "wheel_to_base_bias_gain", wheel_to_base_bias_gain)
    recovery_wheel_despin_scale = _override_float(
        runtime_overrides, "recovery_wheel_despin_scale", recovery_wheel_despin_scale
    )
    hold_wheel_despin_scale = _override_float(
        runtime_overrides, "hold_wheel_despin_scale", hold_wheel_despin_scale
    )
    base_force_soft_limit = _override_float(runtime_overrides, "base_force_soft_limit", base_force_soft_limit)
    base_damping_gain = _override_float(runtime_overrides, "base_damping_gain", base_damping_gain)
    base_centering_gain = _override_float(runtime_overrides, "base_centering_gain", base_centering_gain)
    base_tilt_deadband_deg = _override_float(runtime_overrides, "base_tilt_deadband_deg", base_tilt_deadband_deg)
    base_tilt_full_authority_deg = _override_float(
        runtime_overrides, "base_tilt_full_authority_deg", base_tilt_full_authority_deg
    )
    base_command_gain = _override_float(runtime_overrides, "base_command_gain", base_command_gain)
    base_centering_pos_clip_m = _override_float(
        runtime_overrides, "base_centering_pos_clip_m", base_centering_pos_clip_m
    )
    base_speed_soft_limit_frac = _override_float(
        runtime_overrides, "base_speed_soft_limit_frac", base_speed_soft_limit_frac
    )
    base_hold_radius_m = _override_float(runtime_overrides, "base_hold_radius_m", base_hold_radius_m)
    base_ref_follow_rate_hz = _override_float(runtime_overrides, "base_ref_follow_rate_hz", base_ref_follow_rate_hz)
    base_ref_recenter_rate_hz = _override_float(
        runtime_overrides, "base_ref_recenter_rate_hz", base_ref_recenter_rate_hz
    )
    base_authority_rate_per_s = _override_float(
        runtime_overrides, "base_authority_rate_per_s", base_authority_rate_per_s
    )
    base_command_lpf_hz = _override_float(runtime_overrides, "base_command_lpf_hz", base_command_lpf_hz)
    upright_base_du_scale = _override_float(runtime_overrides, "upright_base_du_scale", upright_base_du_scale)
    base_pitch_kp = _override_float(runtime_overrides, "base_pitch_kp", base_pitch_kp)
    base_pitch_kd = _override_float(runtime_overrides, "base_pitch_kd", base_pitch_kd)
    base_roll_kp = _override_float(runtime_overrides, "base_roll_kp", base_roll_kp)
    base_roll_kd = _override_float(runtime_overrides, "base_roll_kd", base_roll_kd)
    wheel_only_pitch_kp = _override_float(runtime_overrides, "wheel_only_pitch_kp", wheel_only_pitch_kp)
    wheel_only_pitch_kd = _override_float(runtime_overrides, "wheel_only_pitch_kd", wheel_only_pitch_kd)
    wheel_only_pitch_ki = _override_float(runtime_overrides, "wheel_only_pitch_ki", wheel_only_pitch_ki)
    wheel_only_wheel_rate_kd = _override_float(
        runtime_overrides, "wheel_only_wheel_rate_kd", wheel_only_wheel_rate_kd
    )
    wheel_only_max_u = _override_float(runtime_overrides, "wheel_only_max_u", wheel_only_max_u)
    wheel_only_max_du = _override_float(runtime_overrides, "wheel_only_max_du", wheel_only_max_du)
    online_id_forgetting = _override_float(runtime_overrides, "online_id_forgetting", online_id_forgetting)
    online_id_init_cov = _override_float(runtime_overrides, "online_id_init_cov", online_id_init_cov)
    online_id_min_excitation = _override_float(runtime_overrides, "online_id_min_excitation", online_id_min_excitation)
    online_id_recompute_every = _override_int(runtime_overrides, "online_id_recompute_every", online_id_recompute_every)
    online_id_min_updates = _override_int(runtime_overrides, "online_id_min_updates", online_id_min_updates)
    online_id_gravity_scale_min = _override_float(
        runtime_overrides, "online_id_gravity_scale_min", online_id_gravity_scale_min
    )
    online_id_gravity_scale_max = _override_float(
        runtime_overrides, "online_id_gravity_scale_max", online_id_gravity_scale_max
    )
    online_id_inertia_inv_scale_min = _override_float(
        runtime_overrides, "online_id_inertia_inv_scale_min", online_id_inertia_inv_scale_min
    )
    online_id_inertia_inv_scale_max = _override_float(
        runtime_overrides, "online_id_inertia_inv_scale_max", online_id_inertia_inv_scale_max
    )
    online_id_scale_rate_per_s = _override_float(
        runtime_overrides, "online_id_scale_rate_per_s", online_id_scale_rate_per_s
    )
    online_id_gain_blend_alpha = _override_float(
        runtime_overrides, "online_id_gain_blend_alpha", online_id_gain_blend_alpha
    )
    online_id_gain_max_delta = _override_float(runtime_overrides, "online_id_gain_max_delta", online_id_gain_max_delta)
    online_id_innovation_clip = _override_float(
        runtime_overrides, "online_id_innovation_clip", online_id_innovation_clip
    )

    online_id_forgetting = float(np.clip(online_id_forgetting, 0.90, 0.999999))
    online_id_init_cov = float(max(online_id_init_cov, 1.0))
    online_id_min_excitation = float(max(online_id_min_excitation, 1e-6))
    online_id_recompute_every = int(max(online_id_recompute_every, 1))
    online_id_min_updates = int(max(online_id_min_updates, 1))
    online_id_gravity_scale_min = float(max(online_id_gravity_scale_min, 0.05))
    online_id_gravity_scale_max = float(max(online_id_gravity_scale_max, online_id_gravity_scale_min + 1e-6))
    online_id_inertia_inv_scale_min = float(max(online_id_inertia_inv_scale_min, 0.05))
    online_id_inertia_inv_scale_max = float(max(online_id_inertia_inv_scale_max, online_id_inertia_inv_scale_min + 1e-6))
    online_id_scale_rate_per_s = float(max(online_id_scale_rate_per_s, 0.0))
    online_id_gain_blend_alpha = float(np.clip(online_id_gain_blend_alpha, 0.0, 1.0))
    online_id_gain_max_delta = float(max(online_id_gain_max_delta, 0.0))
    online_id_innovation_clip = float(max(online_id_innovation_clip, 1e-6))

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

    if hardware_safe or real_hardware_profile:
        # Keep adaptation disabled during conservative hardware bring-up.
        online_id_enabled = False

    crash_divergence_gate_enabled = bool(getattr(args, "crash_gate_divergence", True))
    crash_recovery_window_steps = max(int(getattr(args, "crash_recovery_steps", 500)), 0)
    rw_emergency_du_enabled = bool(getattr(args, "rw_emergency_du", True))
    rw_emergency_pitch_rad = float(np.radians(max(float(getattr(args, "rw_emergency_pitch_deg", 15.0)), 0.0)))
    rw_emergency_du_scale = float(max(float(getattr(args, "rw_emergency_du_scale", 1.5)), 1.0))
    if hardware_safe or real_hardware_profile:
        # Keep real-hardware bring-up conservative unless explicitly retuned.
        rw_emergency_du_enabled = False
        rw_emergency_du_scale = 1.0
    hold_base_x_centering_gain = float(max(getattr(args, "hold_base_x_centering_gain", 0.0), 0.0))
    linearize_pitch_rad = float(np.radians(float(getattr(args, "linearize_pitch_deg", 0.0))))
    linearize_roll_rad = float(np.radians(float(getattr(args, "linearize_roll_deg", 0.0))))

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
    sensor_source = str(getattr(args, "sensor_source", "auto")).strip().lower()
    if sensor_source not in {"auto", "mujoco", "direct"}:
        sensor_source = "auto"
    sensor_hz_arg = getattr(args, "sensor_hz", None)
    if sensor_hz_arg is None:
        sensor_hz = max(control_hz, 1.0)
    else:
        sensor_hz = max(float(sensor_hz_arg), 1.0)
    sensor_delay_steps = max(int(getattr(args, "sensor_delay_steps", 0)), 0)

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
    dob_enabled = bool(getattr(args, "enable_dob", False))
    gain_schedule_enabled = bool(getattr(args, "enable_gain_scheduling", False))
    if controller_family == "current_dob":
        dob_enabled = True
    if gain_schedule_enabled:
        dob_enabled = True
    dob_gain = float(max(float(getattr(args, "dob_gain", 14.0)), 0.0))
    dob_cutoff_hz = float(max(float(getattr(args, "dob_cutoff_hz", 0.0)), 0.0))
    if dob_cutoff_hz > 0.0:
        dob_gain = float(2.0 * np.pi * dob_cutoff_hz)
    dob_leak_per_s = float(max(float(getattr(args, "dob_leak_per_s", 0.6)), 0.0))
    dob_max_abs_u = np.array(
        [
            max(float(getattr(args, "dob_max_rw", 0.0)), 0.0),
            max(float(getattr(args, "dob_max_bx", 0.0)), 0.0),
            max(float(getattr(args, "dob_max_by", 0.0)), 0.0),
        ],
        dtype=float,
    )
    dob_max_abs_u = np.minimum(dob_max_abs_u, max_u)
    gain_schedule_min = float(max(float(getattr(args, "gain_sched_min", 1.0)), 0.1))
    gain_schedule_max = float(max(float(getattr(args, "gain_sched_max", 1.8)), gain_schedule_min))
    gain_schedule_disturbance_ref = float(max(float(getattr(args, "gain_sched_ref", 2.0)), 1e-6))
    gain_schedule_rate_per_s = float(max(float(getattr(args, "gain_sched_rate_per_s", 3.0)), 0.0))
    gain_schedule_weights = np.array(
        [
            max(float(getattr(args, "gain_sched_rw_weight", 1.0)), 0.0),
            max(float(getattr(args, "gain_sched_bx_weight", 0.6)), 0.0),
            max(float(getattr(args, "gain_sched_by_weight", 0.6)), 0.0),
        ],
        dtype=float,
    )
    if float(np.sum(gain_schedule_weights)) <= 1e-9:
        gain_schedule_weights = np.ones(3, dtype=float)
    if real_hardware_profile:
        gain_schedule_max = min(gain_schedule_max, 1.4)
    payload_mass_kg = float(max(getattr(args, "payload_mass", 0.0), 0.0))
    payload_support_radius_m = float(max(getattr(args, "payload_support_radius_m", 0.145), 0.01))
    payload_com_fail_steps = int(max(getattr(args, "payload_com_fail_steps", 15), 1))

    return RuntimeConfig(
        controller_family=controller_family,
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
        lock_root_attitude=bool(getattr(args, "lock_root_attitude", True)),
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
        sensor_source=sensor_source,
        sensor_hz=sensor_hz,
        sensor_delay_steps=sensor_delay_steps,
        imu_angle_bias_rw_std_rad_sqrt_s=float(np.radians(max(getattr(args, "imu_angle_bias_rw_deg", 0.02), 0.0))),
        imu_rate_bias_rw_std_rad_s_sqrt_s=float(max(getattr(args, "imu_rate_bias_rw", 0.003), 0.0)),
        wheel_encoder_bias_rw_std_rad_s_sqrt_s=float(max(getattr(args, "wheel_rate_bias_rw", 0.002), 0.0)),
        base_encoder_pos_bias_rw_std_m_sqrt_s=float(max(getattr(args, "base_pos_bias_rw", 3e-4), 0.0)),
        base_encoder_vel_bias_rw_std_m_s_sqrt_s=float(max(getattr(args, "base_vel_bias_rw", 0.004), 0.0)),
        imu_angle_clip_rad=float(np.radians(max(getattr(args, "imu_angle_clip_deg", 85.0), 1e-3))),
        imu_rate_clip_rad_s=float(max(getattr(args, "imu_rate_clip", 30.0), 1e-3)),
        wheel_rate_clip_rad_s=float(
            max(float(getattr(args, "wheel_rate_clip", None)), 1e-3)
            if getattr(args, "wheel_rate_clip", None) is not None
            else max(max_wheel_speed_rad_s, 1e-3)
        ),
        base_pos_clip_m=float(max(getattr(args, "base_pos_clip", 0.75), 1e-4)),
        base_vel_clip_m_s=float(max(getattr(args, "base_vel_clip", 6.0), 1e-4)),
        imu_angle_lpf_hz=float(max(getattr(args, "imu_angle_lpf_hz", 45.0), 0.0)),
        imu_rate_lpf_hz=float(max(getattr(args, "imu_rate_lpf_hz", 70.0), 0.0)),
        wheel_rate_lpf_hz=float(max(getattr(args, "wheel_rate_lpf_hz", 120.0), 0.0)),
        base_pos_lpf_hz=float(max(getattr(args, "base_pos_lpf_hz", 25.0), 0.0)),
        base_vel_lpf_hz=float(max(getattr(args, "base_vel_lpf_hz", 40.0), 0.0)),
        residual_model_path=(str(getattr(args, "residual_model", "")) if getattr(args, "residual_model", None) else None),
        residual_scale=residual_scale,
        residual_max_abs_u=residual_max_abs_u,
        residual_gate_tilt_rad=residual_gate_tilt_rad,
        residual_gate_rate_rad_s=residual_gate_rate_rad_s,
        dob_enabled=dob_enabled,
        dob_gain=dob_gain,
        dob_leak_per_s=dob_leak_per_s,
        dob_max_abs_u=dob_max_abs_u,
        gain_schedule_enabled=gain_schedule_enabled,
        gain_schedule_min=gain_schedule_min,
        gain_schedule_max=gain_schedule_max,
        gain_schedule_disturbance_ref=gain_schedule_disturbance_ref,
        gain_schedule_rate_per_s=gain_schedule_rate_per_s,
        gain_schedule_weights=gain_schedule_weights,
        max_u=max_u,
        max_du=max_du,
        rw_emergency_du_enabled=rw_emergency_du_enabled,
        rw_emergency_pitch_rad=rw_emergency_pitch_rad,
        rw_emergency_du_scale=rw_emergency_du_scale,
        disturbance_magnitude=disturbance_magnitude,
        disturbance_interval=disturbance_interval,
        qx=qx,
        qu=qu,
        r_du=r_du,
        ki_base=ki_base,
        base_integrator_enabled=bool(args.enable_base_integrator),
        u_bleed=u_bleed,
        crash_angle_rad=np.radians(crash_angle_deg),
        crash_divergence_gate_enabled=crash_divergence_gate_enabled,
        crash_recovery_window_steps=crash_recovery_window_steps,
        payload_mass_kg=payload_mass_kg,
        payload_support_radius_m=payload_support_radius_m,
        payload_com_fail_steps=payload_com_fail_steps,
        x_ref=0.0,
        y_ref=0.0,
        linearize_pitch_rad=linearize_pitch_rad,
        linearize_roll_rad=linearize_roll_rad,
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
        hold_base_x_centering_gain=hold_base_x_centering_gain,
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
        online_id_enabled=online_id_enabled,
        online_id_forgetting=online_id_forgetting,
        online_id_init_cov=online_id_init_cov,
        online_id_min_excitation=online_id_min_excitation,
        online_id_recompute_every=online_id_recompute_every,
        online_id_min_updates=online_id_min_updates,
        online_id_gravity_scale_min=online_id_gravity_scale_min,
        online_id_gravity_scale_max=online_id_gravity_scale_max,
        online_id_inertia_inv_scale_min=online_id_inertia_inv_scale_min,
        online_id_inertia_inv_scale_max=online_id_inertia_inv_scale_max,
        online_id_scale_rate_per_s=online_id_scale_rate_per_s,
        online_id_gain_blend_alpha=online_id_gain_blend_alpha,
        online_id_gain_max_delta=online_id_gain_max_delta,
        online_id_innovation_clip=online_id_innovation_clip,
        online_id_verbose=online_id_verbose,
        use_mpc=bool(getattr(args, "use_mpc", False)),
        mpc_horizon=int(max(getattr(args, "mpc_horizon", 32), 5)),
        mpc_q_angles=float(max(getattr(args, "mpc_q_angles", 280.0), 1.0)),
        mpc_q_rates=float(max(getattr(args, "mpc_q_rates", 180.0), 1.0)),
        mpc_q_position=float(max(getattr(args, "mpc_q_position", 30.0), 1.0)),
        mpc_r_control=float(max(getattr(args, "mpc_r_control", 0.25), 0.01)),
        mpc_terminal_weight=float(max(getattr(args, "mpc_terminal_weight", 8.0), 1.0)),
        mpc_target_rate_gain=float(max(getattr(args, "mpc_target_rate_gain", 4.0), 0.0)),
        mpc_terminal_rate_gain=float(max(getattr(args, "mpc_terminal_rate_gain", 7.0), 0.0)),
        mpc_target_rate_clip_rad_s=float(max(getattr(args, "mpc_target_rate_clip", 4.0), 0.0)),
        mpc_com_constraint_radius_m=payload_support_radius_m,
        mpc_pitch_i_gain=float(max(getattr(args, "mpc_pitch_i_gain", 9.0), 0.0)),
        mpc_pitch_i_clamp=float(max(getattr(args, "mpc_pitch_i_clamp", 0.45), 0.0)),
        mpc_pitch_i_deadband_rad=float(np.radians(max(getattr(args, "mpc_pitch_i_deadband_deg", 0.15), 0.0))),
        mpc_pitch_i_leak_per_s=float(max(getattr(args, "mpc_pitch_i_leak_per_s", 0.8), 0.0)),
        mpc_pitch_guard_angle_frac=float(np.clip(getattr(args, "mpc_pitch_guard_angle_frac", 0.45), 0.05, 0.98)),
        mpc_pitch_guard_rate_entry_rad_s=float(max(getattr(args, "mpc_pitch_guard_rate", 1.2), 0.0)),
        mpc_pitch_guard_kp=float(max(getattr(args, "mpc_pitch_guard_kp", 120.0), 0.0)),
        mpc_pitch_guard_kd=float(max(getattr(args, "mpc_pitch_guard_kd", 28.0), 0.0)),
        mpc_pitch_guard_max_frac=float(np.clip(getattr(args, "mpc_pitch_guard_max_frac", 1.0), 0.1, 1.5)),
        mpc_roll_i_gain=float(max(getattr(args, "mpc_roll_i_gain", 8.0), 0.0)),
        mpc_roll_i_clamp=float(max(getattr(args, "mpc_roll_i_clamp", 0.35), 0.0)),
        mpc_roll_i_deadband_rad=float(np.radians(max(getattr(args, "mpc_roll_i_deadband_deg", 0.12), 0.0))),
        mpc_roll_i_leak_per_s=float(max(getattr(args, "mpc_roll_i_leak_per_s", 0.8), 0.0)),
        mpc_verbose=bool(getattr(args, "mpc_verbose", False)),
    )
