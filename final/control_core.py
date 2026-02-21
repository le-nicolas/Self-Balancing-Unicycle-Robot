from collections import deque

import numpy as np

from runtime_config import RuntimeConfig

try:
    from mpc_controller import MPCController
    MPC_AVAILABLE = True
except ImportError:
    MPC_AVAILABLE = False
    MPCController = None


def reset_controller_buffers(nx: int, nu: int, queue_len: int):
    x_est = np.zeros(nx)
    u_applied = np.zeros(nu)
    u_eff_applied = np.zeros(nu)
    base_int = np.zeros(2)
    wheel_pitch_int = 0.0
    wheel_momentum_bias_int = 0.0
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
        wheel_momentum_bias_int,
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
        # Near hard-speed, never allow same-direction torque that can increase wheel runaway.
        wheel_cmd = 0.0
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




def _init_control_terms() -> dict[str, np.ndarray]:
    return {
        "term_lqr_core": np.zeros(3, dtype=float),
        "term_roll_stability": np.zeros(3, dtype=float),
        "term_pitch_stability": np.zeros(3, dtype=float),
        "term_despin": np.zeros(3, dtype=float),
        "term_base_hold": np.zeros(3, dtype=float),
        "term_safety_shaping": np.zeros(3, dtype=float),
        "term_disturbance_comp": np.zeros(3, dtype=float),
        "term_mpc": np.zeros(3, dtype=float),
        "gain_schedule_scale": np.array([1.0], dtype=float),
        "disturbance_level": np.array([0.0], dtype=float),
    }


def _update_wheel_momentum_bias(
    cfg: RuntimeConfig,
    wheel_momentum_bias_int: float,
    wheel_speed_est: float,
    balance_phase: str,
    control_dt: float,
) -> tuple[float, float]:
    # Slow hold-phase wheel recentering: integrate wheel-speed bias, leak elsewhere.
    speed_ref = min(cfg.wheel_spin_budget_frac * cfg.max_wheel_speed_rad_s, cfg.wheel_spin_budget_abs_rad_s)
    speed_ref = max(0.35 * speed_ref, 1e-3)
    if balance_phase == "hold":
        wheel_norm = float(np.clip(wheel_speed_est / speed_ref, -2.0, 2.0))
        wheel_momentum_bias_int += wheel_norm * control_dt
        wheel_momentum_bias_int = float(np.clip(wheel_momentum_bias_int, -2.0, 2.0))
    else:
        leak = float(np.clip(0.65 * control_dt, 0.0, 1.0))
        wheel_momentum_bias_int = float((1.0 - leak) * wheel_momentum_bias_int)

    bias_cmd = float(np.clip(-0.32 * cfg.wheel_to_base_bias_gain * wheel_momentum_bias_int, -0.25, 0.25))
    return wheel_momentum_bias_int, bias_cmd


def _fuzzy_roll_gain(cfg: RuntimeConfig, roll: float, roll_rate: float) -> float:
    roll_n = abs(roll) / max(cfg.hold_exit_angle_rad, 1e-6)
    rate_n = abs(roll_rate) / max(cfg.hold_exit_rate_rad_s, 1e-6)
    level = float(np.clip(0.65 * roll_n + 0.35 * rate_n, 0.0, 1.0))
    return 0.35 + 0.95 * level


def _resolve_rw_du_limit(
    cfg: RuntimeConfig,
    x_est: np.ndarray,
    balance_phase: str,
    wheel_only: bool,
) -> float:
    rw_du_limit = float(cfg.wheel_only_max_du if wheel_only else cfg.max_du[0])
    if cfg.rw_emergency_du_enabled and balance_phase != "hold":
        if abs(float(x_est[0])) >= cfg.rw_emergency_pitch_rad:
            rw_du_limit *= cfg.rw_emergency_du_scale
    return rw_du_limit


def update_disturbance_observer(
    cfg: RuntimeConfig,
    A: np.ndarray,
    B: np.ndarray,
    B_pinv: np.ndarray,
    x_prev: np.ndarray | None,
    u_prev: np.ndarray,
    x_curr: np.ndarray,
    dob_hat: np.ndarray,
    control_dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    if (not cfg.dob_enabled) or x_prev is None:
        return dob_hat, np.zeros(3, dtype=float)
    x_model = A @ x_prev + B @ u_prev
    state_residual = x_curr - x_model
    d_raw = B_pinv @ state_residual
    if (d_raw.shape[0] != 3) or (not np.all(np.isfinite(d_raw))):
        return np.zeros(3, dtype=float), np.zeros(3, dtype=float)

    # First-order Q-filter discretization (equivalent to LPF on residual).
    gain = float(max(cfg.dob_gain, 0.0))
    alpha = float(1.0 - np.exp(-gain * max(control_dt, 0.0)))
    alpha = float(np.clip(alpha, 0.0, 1.0))
    dob_next = (1.0 - alpha) * dob_hat + alpha * d_raw
    leak = float(np.clip(cfg.dob_leak_per_s * control_dt, 0.0, 1.0))
    dob_next *= (1.0 - leak)
    dob_next = np.clip(dob_next, -cfg.dob_max_abs_u, cfg.dob_max_abs_u)
    if not np.all(np.isfinite(dob_next)):
        return np.zeros(3, dtype=float), np.zeros(3, dtype=float)
    return dob_next.astype(float, copy=False), d_raw.astype(float, copy=False)


def update_gain_schedule(
    cfg: RuntimeConfig,
    dob_hat: np.ndarray,
    gain_schedule_state: float,
    control_dt: float,
) -> tuple[float, float]:
    if not cfg.gain_schedule_enabled:
        return 1.0, 0.0
    weights = np.asarray(cfg.gain_schedule_weights, dtype=float)
    if weights.shape[0] != 3:
        weights = np.ones(3, dtype=float)
    disturbance_level = float(np.linalg.norm(weights * np.asarray(dob_hat, dtype=float)))
    level_norm = float(np.clip(disturbance_level / max(cfg.gain_schedule_disturbance_ref, 1e-6), 0.0, 1.0))
    target = float(cfg.gain_schedule_min + (cfg.gain_schedule_max - cfg.gain_schedule_min) * level_norm)
    rate = float(max(cfg.gain_schedule_rate_per_s, 0.0))
    if np.isfinite(rate) and rate > 0.0:
        max_step = rate * control_dt
        next_scale = float(gain_schedule_state + np.clip(target - gain_schedule_state, -max_step, max_step))
    else:
        next_scale = float(target)
    next_scale = float(np.clip(next_scale, cfg.gain_schedule_min, cfg.gain_schedule_max))
    return next_scale, disturbance_level


def _apply_dob_compensation(
    cfg: RuntimeConfig,
    u_cmd: np.ndarray,
    sat_hits: np.ndarray,
    terms: dict[str, np.ndarray],
    dob_compensation: np.ndarray | None,
) -> np.ndarray:
    if (not cfg.dob_enabled) or dob_compensation is None:
        return u_cmd
    dob = np.asarray(dob_compensation, dtype=float).reshape(-1)
    if dob.size != 3 or (not np.all(np.isfinite(dob))):
        return u_cmd
    comp = -dob
    terms["term_disturbance_comp"] = comp.astype(float, copy=False)
    u_comp = u_cmd + comp
    sat_hits += (np.abs(u_comp) > cfg.max_u).astype(int)
    return np.clip(u_comp, -cfg.max_u, cfg.max_u)


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
    wheel_momentum_bias_int: float,
    balance_phase: str,
    recovery_time_s: float,
    high_spin_active: bool,
    control_dt: float,
    K_du: np.ndarray,
    K_wheel_only: np.ndarray | None,
    K_paper_pitch: np.ndarray | None,
    du_hits: np.ndarray,
    sat_hits: np.ndarray,
    dob_compensation: np.ndarray | None = None,
    gain_schedule_scale: float = 1.0,
    disturbance_level: float = 0.0,
    mpc_controller: "MPCController | None" = None,
):
    """Controller core: delta-u LQR + wheel-only mode + base policy + safety shaping."""
    wheel_over_budget = False
    wheel_over_hard = False
    terms = _init_control_terms()
    schedule_scale = (
        float(np.clip(gain_schedule_scale, cfg.gain_schedule_min, cfg.gain_schedule_max))
        if cfg.gain_schedule_enabled
        else 1.0
    )
    terms["gain_schedule_scale"][0] = schedule_scale
    terms["disturbance_level"][0] = float(max(disturbance_level, 0.0))
    x_ctrl = x_est.copy()
    x_ctrl[5] -= cfg.x_ref
    x_ctrl[6] -= cfg.y_ref

    if cfg.base_integrator_enabled:
        base_int[0] = np.clip(base_int[0] + x_ctrl[5] * control_dt, -cfg.int_clamp, cfg.int_clamp)
        base_int[1] = np.clip(base_int[1] + x_ctrl[6] * control_dt, -cfg.int_clamp, cfg.int_clamp)
    else:
        base_int[:] = 0.0

    # ========== MPC PATH (if enabled) ==========
    if cfg.use_mpc and mpc_controller is not None:
        # MPC solves the constrained optimization problem directly
        # State-target shaping: convert angle error into desired damping rates.
        # This helps avoid monotonic backward tilt while keeping upright target at zero.
        x_ref = np.zeros_like(x_ctrl)
        rate_clip = float(max(cfg.mpc_target_rate_clip_rad_s, 0.0))
        run_rate_gain = float(max(cfg.mpc_target_rate_gain, 0.0))
        term_rate_gain = float(max(cfg.mpc_terminal_rate_gain, 0.0))
        if rate_clip > 0.0:
            x_ref[2] = float(np.clip(-run_rate_gain * x_ctrl[0], -rate_clip, rate_clip))
            x_ref[3] = float(np.clip(-run_rate_gain * x_ctrl[1], -rate_clip, rate_clip))
        x_ref_terminal = x_ref.copy()
        if rate_clip > 0.0:
            x_ref_terminal[2] = float(np.clip(-term_rate_gain * x_ctrl[0], -rate_clip, rate_clip))
            x_ref_terminal[3] = float(np.clip(-term_rate_gain * x_ctrl[1], -rate_clip, rate_clip))

        u_mpc, mpc_info = mpc_controller.solve(
            x_ctrl,
            x_ref=x_ref,
            x_ref_terminal=x_ref_terminal,
        )
        mpc_success = bool(mpc_info.get("success", False))
        if not mpc_success:
            # Runtime fallback: if QP fails, use clipped one-step linear feedback instead of zeroing commands.
            z_fb = np.concatenate([x_ctrl, u_eff_applied])
            du_fb = -K_du @ z_fb
            du_fb = np.clip(du_fb, -cfg.max_du, cfg.max_du)
            u_mpc = np.clip(u_eff_applied + du_fb, -cfg.max_u, cfg.max_u)
            terms["term_lqr_core"] = np.array([du_fb[0], du_fb[1], du_fb[2]], dtype=float)

        # MPC pitch anti-drift integral: adds bounded bias to wheel channel.
        # This compensates slow model mismatch that appears as monotonic pitch drift.
        i_term = 0.0
        if cfg.mpc_pitch_i_gain > 0.0 and cfg.mpc_pitch_i_clamp > 0.0:
            pitch_err = float(x_ctrl[0])
            deadband = float(cfg.mpc_pitch_i_deadband_rad)
            if abs(pitch_err) > deadband:
                err_for_i = pitch_err - np.sign(pitch_err) * deadband
                wheel_pitch_int = float(
                    np.clip(
                        wheel_pitch_int + err_for_i * control_dt,
                        -cfg.mpc_pitch_i_clamp,
                        cfg.mpc_pitch_i_clamp,
                    )
                )
            else:
                # Leak the integrator near upright to prevent bias memory buildup.
                leak = float(np.clip(cfg.mpc_pitch_i_leak_per_s * control_dt, 0.0, 1.0))
                wheel_pitch_int = float((1.0 - leak) * wheel_pitch_int)
            i_term = float(-cfg.mpc_pitch_i_gain * wheel_pitch_int)

        # Pitch rescue guard: blend in aggressive wheel PD when pitch is trending to crash.
        # This is a safety override layered on MPC to catch late-time pitch drift.
        u_rw_nom = float(u_mpc[0] + i_term)
        u_rw_raw = u_rw_nom
        pitch_guard_active = False
        pitch_guard_severity = 0.0
        rescue_base_x = 0.0
        pitch = float(x_ctrl[0])
        pitch_rate = float(x_ctrl[2])
        guard_angle = float(cfg.mpc_pitch_guard_angle_frac * cfg.crash_angle_rad)
        guard_angle = float(np.clip(guard_angle, 1e-4, max(cfg.crash_angle_rad - 1e-4, 1e-4)))
        guard_rate = float(max(cfg.mpc_pitch_guard_rate_entry_rad_s, 0.0))
        moving_out = (
            abs(pitch_rate) > guard_rate
            and abs(pitch) > 0.5 * guard_angle
            and np.sign(pitch_rate) == np.sign(pitch)
        )
        if abs(pitch) >= guard_angle or moving_out:
            pitch_guard_active = True
            sev_angle = float(
                np.clip((abs(pitch) - guard_angle) / max(cfg.crash_angle_rad - guard_angle, 1e-6), 0.0, 1.0)
            )
            sev_rate = 0.0
            if guard_rate > 1e-9:
                sev_rate = float(np.clip((abs(pitch_rate) - guard_rate) / guard_rate, 0.0, 1.0))
            pitch_guard_severity = float(np.clip(max(sev_angle, 0.6 * sev_rate), 0.0, 1.0))
            rescue_raw = float(-(cfg.mpc_pitch_guard_kp * pitch + cfg.mpc_pitch_guard_kd * pitch_rate))
            rescue_lim = float(cfg.mpc_pitch_guard_max_frac * cfg.max_u[0])
            rescue_cmd = float(np.clip(rescue_raw, -rescue_lim, rescue_lim))
            u_rw_raw = float((1.0 - pitch_guard_severity) * u_rw_nom + pitch_guard_severity * rescue_cmd)
            rescue_base_x = float(
                np.clip(
                    cfg.base_command_gain * (cfg.base_pitch_kp * pitch + cfg.base_pitch_kd * pitch_rate),
                    -cfg.max_u[1],
                    cfg.max_u[1],
                )
            )
            terms["term_pitch_stability"][0] += float(u_rw_raw - u_rw_nom)

        # Clip and apply MPC output (with integral bias and optional rescue blend).
        u_rw_cmd = float(np.clip(u_rw_raw, -cfg.max_u[0], cfg.max_u[0]))
        # Roll anti-drift integral uses base_int[1] state slot in MPC mode.
        # It provides slow bias correction on the roll actuator channel.
        i_roll_term = 0.0
        if cfg.mpc_roll_i_gain > 0.0 and cfg.mpc_roll_i_clamp > 0.0:
            roll_err = float(x_ctrl[1])
            deadband_r = float(cfg.mpc_roll_i_deadband_rad)
            if abs(roll_err) > deadband_r:
                err_for_i_r = roll_err - np.sign(roll_err) * deadband_r
                base_int[1] = float(
                    np.clip(
                        base_int[1] + err_for_i_r * control_dt,
                        -cfg.mpc_roll_i_clamp,
                        cfg.mpc_roll_i_clamp,
                    )
                )
            else:
                leak_r = float(np.clip(cfg.mpc_roll_i_leak_per_s * control_dt, 0.0, 1.0))
                base_int[1] = float((1.0 - leak_r) * base_int[1])
            i_roll_term = float(-cfg.mpc_roll_i_gain * base_int[1])

        u_base_raw = np.array([float(u_mpc[1]), float(u_mpc[2] + i_roll_term)], dtype=float)
        # Always provide a small pitch-support bias on base-x in MPC mode.
        # Scale grows with pitch-integrator magnitude so persistent drift is corrected earlier.
        drift_level = float(np.clip(abs(wheel_pitch_int) / max(cfg.mpc_pitch_i_clamp, 1e-6), 0.0, 1.0))
        support_scale = float(0.10 + 0.35 * drift_level)
        pitch_base_support = float(
            np.clip(
                support_scale * cfg.base_command_gain * (cfg.base_pitch_kp * pitch + cfg.base_pitch_kd * pitch_rate),
                -0.50 * cfg.max_u[1],
                0.50 * cfg.max_u[1],
            )
        )
        u_base_raw[0] += pitch_base_support
        terms["term_pitch_stability"][1] += pitch_base_support
        if pitch_guard_active:
            # During pitch rescue, prioritize base-x support and reduce lateral command coupling.
            base_blend = float(np.clip(0.35 + 0.65 * pitch_guard_severity, 0.0, 1.0))
            base_x_nom = float(u_base_raw[0])
            base_y_nom = float(u_base_raw[1])
            u_base_raw[0] = float((1.0 - base_blend) * base_x_nom + base_blend * rescue_base_x)
            y_keep = float(np.clip(1.0 - 0.45 * pitch_guard_severity, 0.45, 1.0))
            u_base_raw[1] = float(y_keep * u_base_raw[1])
            terms["term_pitch_stability"][1] += float(u_base_raw[0] - base_x_nom)
            terms["term_pitch_stability"][2] += float(u_base_raw[1] - base_y_nom)
        u_base_cmd = np.clip(u_base_raw, -cfg.max_u[1:], cfg.max_u[1:])

        # Back-calculation anti-windup for integrator when wheel channel clips.
        if cfg.mpc_pitch_i_gain > 1e-9:
            sat_excess = float(u_rw_raw - u_rw_cmd)
            if abs(sat_excess) > 1e-12:
                wheel_pitch_int = float(
                    np.clip(
                        wheel_pitch_int + sat_excess / cfg.mpc_pitch_i_gain,
                        -cfg.mpc_pitch_i_clamp,
                        cfg.mpc_pitch_i_clamp,
                    )
                )
        if cfg.mpc_roll_i_gain > 1e-9:
            sat_excess_r = float(u_base_raw[1] - u_base_cmd[1])
            if abs(sat_excess_r) > 1e-12:
                base_int[1] = float(
                    np.clip(
                        base_int[1] + sat_excess_r / cfg.mpc_roll_i_gain,
                        -cfg.mpc_roll_i_clamp,
                        cfg.mpc_roll_i_clamp,
                    )
                )

        # Record to terms for logging
        terms["term_mpc"] = u_mpc
        terms["term_pitch_stability"][0] += i_term
        terms["term_roll_stability"][2] += i_roll_term

        # Handle saturation hits
        sat_hits[0] += int(abs(u_rw_raw) > cfg.max_u[0])
        sat_hits[1:] += (np.abs(u_base_raw) > cfg.max_u[1:]).astype(int)

        # Return MPC result in same format as LQR path.
        # DOB feed-forward is applied here as well so MPC can cancel estimated disturbances.
        u_cmd = np.array([u_rw_cmd, u_base_cmd[0], u_base_cmd[1]], dtype=float)
        u_cmd = _apply_dob_compensation(cfg, u_cmd, sat_hits, terms, dob_compensation)
        rw_u_limit = cfg.max_u[0]
        wheel_over_budget = False  # MPC handles constraints explicitly
        wheel_over_hard = False

        return (
            u_cmd,
            base_int,
            base_ref,
            base_authority_state,
            u_base_smooth,
            wheel_pitch_int,
            wheel_momentum_bias_int,
            rw_u_limit,
            wheel_over_budget,
            wheel_over_hard,
            high_spin_active,
            terms,
        )

    # ========== DEFAULT LQR PATH ==========
    # Use effective applied command in delta-u state to avoid windup when
    # saturation/delay/motor limits differ from requested command.
    z = np.concatenate([x_ctrl, u_eff_applied])
    du_lqr = -K_du @ z
    if not cfg.use_mpc:
        du_lqr *= schedule_scale
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
        u_pitch *= schedule_scale
        u_roll_sm *= schedule_scale

        terms["term_pitch_stability"][0] = u_pitch
        terms["term_roll_stability"][0] = u_roll_sm
        u_rw_target = u_pitch + u_roll_sm
        du_rw_cmd = float(u_rw_target - u_eff_applied[0])
        rw_du_limit = _resolve_rw_du_limit(cfg=cfg, x_est=x_est, balance_phase=balance_phase, wheel_only=False)
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
            balance_x *= schedule_scale
            balance_y *= schedule_scale
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
        u_cmd = _apply_dob_compensation(cfg, u_cmd, sat_hits, terms, dob_compensation)
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
            wheel_momentum_bias_int,
            rw_u_limit,
            wheel_over_budget,
            wheel_over_hard,
            high_spin_active,
            terms,
        )

    if cfg.wheel_only:
        xw = np.array([x_est[0], x_est[2], x_est[4]], dtype=float)
        u_rw_target = float(-(K_wheel_only @ xw)[0])
        u_rw_target *= schedule_scale
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
        rw_du_limit = _resolve_rw_du_limit(cfg=cfg, x_est=x_est, balance_phase=balance_phase, wheel_only=True)
        rw_u_limit = cfg.wheel_only_max_u
        base_int[:] = 0.0
        base_ref[:] = 0.0
        wheel_momentum_bias_int = 0.0
    else:
        wheel_pitch_int = 0.0
        rw_frac = abs(float(x_est[4])) / max(cfg.max_wheel_speed_rad_s, 1e-6)
        rw_damp_gain = 0.18 + 0.60 * max(0.0, rw_frac - 0.35)
        du_rw_cmd = float(du_lqr[0] - rw_damp_gain * x_est[4])
        terms["term_despin"][0] += float(-rw_damp_gain * x_est[4])
        if cfg.controller_family == "hybrid_modern":
            pitch_stab = float(-(0.12 * cfg.base_pitch_kp) * x_est[0] - (0.10 * cfg.base_pitch_kd) * x_est[2])
            roll_stab = float((0.06 * cfg.base_roll_kp) * x_est[1] + (0.06 * cfg.base_roll_kd) * x_est[3])
            pitch_stab *= schedule_scale
            roll_stab *= schedule_scale
            du_rw_cmd += pitch_stab + roll_stab
            terms["term_pitch_stability"][0] += pitch_stab
            terms["term_roll_stability"][0] += roll_stab
        rw_du_limit = _resolve_rw_du_limit(cfg=cfg, x_est=x_est, balance_phase=balance_phase, wheel_only=False)
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
        # Restore deadband-based blending: hold mode (position centering) at small angles, 
        # balance mode (pitch stabilization) at larger angles
        tilt_mag = max(abs(float(x_est[0])), abs(float(x_est[1])))
        tilt_span = max(float(cfg.base_tilt_full_authority_rad - cfg.base_tilt_deadband_rad), 1e-6)
        base_authority_raw = float(np.clip((tilt_mag - cfg.base_tilt_deadband_rad) / tilt_span, 0.0, 1.0))
        # Rate-limit authority changes to avoid abrupt mode shifts.
        base_rate = float(max(cfg.base_authority_rate_per_s, 0.0))
        if base_rate > 0.0:
            max_step = base_rate * control_dt
            base_authority_state = float(
                base_authority_state + np.clip(base_authority_raw - base_authority_state, -max_step, max_step)
            )
        else:
            base_authority_state = base_authority_raw
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
        balance_x *= schedule_scale
        balance_y *= schedule_scale
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
        hold_center_x = 0.0
        if balance_phase == "hold" and cfg.hold_base_x_centering_gain > 0.0:
            hold_center_err_x = float(np.clip(x_est[5], -cfg.base_centering_pos_clip_m, cfg.base_centering_pos_clip_m))
            hold_center_x = float(-cfg.hold_base_x_centering_gain * hold_center_err_x)
            base_target_x += hold_center_x
            terms["term_base_hold"][1] += hold_center_x
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
        wheel_momentum_bias_int, hold_bias_term = _update_wheel_momentum_bias(
            cfg=cfg,
            wheel_momentum_bias_int=wheel_momentum_bias_int,
            wheel_speed_est=float(x_est[4]),
            balance_phase=balance_phase,
            control_dt=control_dt,
        )
        base_target_x += hold_bias_term
        terms["term_despin"][1] += hold_bias_term

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
        wheel_momentum_bias_int = 0.0
        du_base_cmd = -u_eff_applied[1:]
        du_hits[1:] += (np.abs(du_base_cmd) > cfg.max_du[1:]).astype(int)
        du_base = np.clip(du_base_cmd, -cfg.max_du[1:], cfg.max_du[1:])
        u_base_cmd = u_eff_applied[1:] + du_base
        u_base_smooth[:] = 0.0

    u_cmd = np.array([u_rw_cmd, u_base_cmd[0], u_base_cmd[1]], dtype=float)
    u_cmd = _apply_dob_compensation(cfg, u_cmd, sat_hits, terms, dob_compensation)
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
        wheel_momentum_bias_int,
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



