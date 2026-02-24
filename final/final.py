import csv
from datetime import datetime
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from scipy.linalg import solve_discrete_are
from runtime_config import build_config, parse_args
from runtime_model import (
    build_kalman_gain,
    build_measurement_noise_cov,
    build_partial_measurement_matrix,
    create_sensor_frontend_state,
    compute_robot_com_distance_xy,
    enforce_planar_root_attitude,
    enforce_wheel_only_constraints,
    estimator_measurement_update,
    get_true_state,
    has_required_mujoco_sensors,
    lookup_model_ids,
    lookup_sensor_ids,
    resolve_sensor_source,
    reset_state,
    reset_sensor_frontend_state,
    set_payload_mass,
)
from control_core import (
    apply_control_delay,
    apply_upright_postprocess,
    base_commands_with_limits,
    compute_control_command,
    reset_controller_buffers,
    update_disturbance_observer,
    update_gain_schedule,
    wheel_command_with_limits,
)
from residual_model import ResidualPolicy
from adaptive_id import AdaptiveGainScheduler

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


def _solve_discrete_are_robust(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray, label: str) -> np.ndarray:
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


def _solve_linear_robust(gram: np.ndarray, rhs: np.ndarray, label: str) -> np.ndarray:
    try:
        return np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        sol, *_ = np.linalg.lstsq(gram, rhs, rcond=None)
        if not np.all(np.isfinite(sol)):
            raise RuntimeError(f"{label} linear solve returned non-finite result.")
        return sol


def _controllability_rank(A: np.ndarray, B: np.ndarray) -> int:
    n = A.shape[0]
    ctrb = B.copy()
    Ak = np.eye(n, dtype=float)
    for _ in range(1, n):
        Ak = Ak @ A
        ctrb = np.hstack([ctrb, Ak @ B])
    return int(np.linalg.matrix_rank(ctrb))


def lift_discrete_dynamics(A: np.ndarray, B: np.ndarray, steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Zero-order-hold lifting from model-step dynamics to controller update interval."""
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


def _render_drag_force_arrow(
    viewer: mujoco.viewer.Handle,
    data: mujoco.MjData,
    body_id: int,
    force_world: np.ndarray,
    anchor_world: np.ndarray | None = None,
) -> None:
    """Draw a red world-space arrow showing current mouse drag force direction."""
    with viewer.lock():
        viewer.user_scn.ngeom = 0
        if viewer.user_scn.maxgeom < 1:
            return
        force_mag = float(np.linalg.norm(force_world))
        if force_mag <= 1e-6:
            return

        if anchor_world is not None and np.all(np.isfinite(anchor_world)):
            start = np.array(anchor_world, dtype=float)
        else:
            if body_id <= 0 or body_id >= data.xipos.shape[0]:
                return
            start = np.array(data.xipos[body_id], dtype=float)
        direction = force_world / force_mag
        length = float(np.clip(0.02 * force_mag, 0.08, 0.45))
        end = start + direction * length

        geom = viewer.user_scn.geoms[0]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_ARROW,
            np.zeros(3, dtype=float),
            np.zeros(3, dtype=float),
            np.eye(3, dtype=float).reshape(9),
            np.array([1.0, 0.1, 0.1, 0.95], dtype=np.float32),
        )
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, 0.02, start, end)
        viewer.user_scn.ngeom = 1


def main():
    # 1) Parse CLI and build runtime tuning/safety profile.
    args = parse_args()
    cfg = build_config(args)
    residual_policy = ResidualPolicy(cfg)
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
    sensor_ids = lookup_sensor_ids(model)
    mujoco_sensor_available = has_required_mujoco_sensors(cfg, sensor_ids)
    sensor_source = resolve_sensor_source(cfg, sensor_ids)
    payload_mass_kg = set_payload_mass(model, data, ids, cfg.payload_mass_kg)
    push_body_id = {
        "stick": ids.stick_body_id,
        "base_y": ids.base_y_body_id,
        "base_x": ids.base_x_body_id,
        "payload": ids.payload_body_id,
    }[args.push_body]
    # Linearize around configured operating point (defaults to upright).
    reset_state(
        model,
        data,
        ids.q_pitch,
        ids.q_roll,
        pitch_eq=cfg.linearize_pitch_rad,
        roll_eq=cfg.linearize_roll_rad,
    )
    if cfg.lock_root_attitude:
        enforce_planar_root_attitude(model, data, ids)

    # Controller update interval (used to lift model-step A/B to control-step A/B for gain design).
    control_steps = 1 if not cfg.hardware_realistic else max(1, int(round(1.0 / (model.opt.timestep * cfg.control_hz))))
    control_dt = control_steps * model.opt.timestep

    # 3) Linearize XML-defined dynamics and build controller gains at control update interval.
    nx = 2 * model.nv + model.na
    nu = model.nu
    A_full = np.zeros((nx, nx))
    B_full = np.zeros((nx, nu))
    mujoco.mjd_transitionFD(model, data, 1e-6, True, A_full, B_full, None, None)

    # mjd_transitionFD uses tangent-space position coordinates (size nv), not qpos (size nq).
    idx = [
        ids.v_pitch,
        ids.v_roll,
        model.nv + ids.v_pitch,
        model.nv + ids.v_roll,
        model.nv + ids.v_rw,
        ids.v_base_x,
        ids.v_base_y,
        model.nv + ids.v_base_x,
        model.nv + ids.v_base_y,
    ]
    A_step = A_full[np.ix_(idx, idx)]
    B_step = B_full[np.ix_(idx, [ids.aid_rw, ids.aid_base_x, ids.aid_base_y])]
    A, B = lift_discrete_dynamics(A_step, B_step, control_steps)
    NX = A.shape[0]
    NU = B.shape[1]
    B_pinv = np.linalg.pinv(B)

    A_aug = np.block([[A, B], [np.zeros((NU, NX)), np.eye(NU)]])
    B_aug = np.vstack([B, np.eye(NU)])
    Q_aug = np.block([[cfg.qx, np.zeros((NX, NU))], [np.zeros((NU, NX)), cfg.qu]])
    P_aug = _solve_discrete_are_robust(A_aug, B_aug, Q_aug, cfg.r_du, label="Controller")
    K_du = _solve_linear_robust(B_aug.T @ P_aug @ B_aug + cfg.r_du, B_aug.T @ P_aug @ A_aug, label="Controller gain")
    K_paper_pitch = None
    K_wheel_only = None
    need_paper_pitch = cfg.wheel_only or (cfg.controller_family == "paper_split_baseline")
    if need_paper_pitch:
        A_w = A[np.ix_([0, 2, 4], [0, 2, 4])]
        B_w = B[np.ix_([0, 2, 4], [0])]
        Q_w = np.diag([260.0, 35.0, 0.6])
        R_w = np.array([[0.08]])
        try:
            P_w = _solve_discrete_are_robust(A_w, B_w, Q_w, R_w, label="Paper pitch")
            K_paper_pitch = _solve_linear_robust(B_w.T @ P_w @ B_w + R_w, B_w.T @ P_w @ A_w, label="Paper pitch gain")
        except RuntimeError as exc:
            print(f"Warning: paper pitch DARE unavailable, using analytic fallback. ({exc})")
            K_paper_pitch = None
    if cfg.wheel_only:
        if K_paper_pitch is not None:
            K_wheel_only = K_paper_pitch.copy()
        else:
            # Fallback: wheel-only branch already adds dedicated I and wheel-rate damping terms.
            K_wheel_only = np.array([[cfg.wheel_only_pitch_kp, cfg.wheel_only_pitch_kd, 0.0]], dtype=float)

    # 4) Build estimator model from configured sensor channels/noise.
    adaptive_scheduler = None
    adaptive_prev_x_est = None
    if (
        cfg.online_id_enabled
        and (not cfg.use_mpc)
        and (cfg.controller_family in {"current", "current_dob", "hybrid_modern"})
        and (not cfg.wheel_only)
    ):
        adaptive_scheduler = AdaptiveGainScheduler(
            cfg=cfg,
            a_nominal=A,
            b_nominal=B,
            control_dt=control_dt,
        )
    wheel_lsb = (2.0 * np.pi) / (cfg.wheel_encoder_ticks_per_rev * control_dt)
    wheel_budget_speed = min(cfg.wheel_spin_budget_frac * cfg.max_wheel_speed_rad_s, cfg.wheel_spin_budget_abs_rad_s)
    wheel_hard_speed = min(cfg.wheel_spin_hard_frac * cfg.max_wheel_speed_rad_s, cfg.wheel_spin_hard_abs_rad_s)
    if not cfg.allow_base_motion:
        wheel_hard_speed *= 1.10
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

    # 5) Initialize MPC controller if enabled
    mpc_controller = None
    if cfg.use_mpc:
        try:
            from mpc_controller import MPCController
            # Build MPC cost matrices based on config
            q_diag = np.array([
                cfg.mpc_q_angles,                    # pitch
                cfg.mpc_q_angles,                    # roll
                cfg.mpc_q_rates,                     # pitch_rate
                cfg.mpc_q_rates,                     # roll_rate
                1.0,                                 # wheel_rate (low penalty)
                cfg.mpc_q_position,                  # base_x
                cfg.mpc_q_position,                  # base_y
                cfg.mpc_q_rates,                     # base_vel_x
                cfg.mpc_q_rates,                     # base_vel_y
            ])
            r_diag = np.array([
                cfg.mpc_r_control,                   # wheel_torque
                cfg.mpc_r_control,                   # base_x_force
                cfg.mpc_r_control,                   # base_y_force
            ])
            mpc_controller = MPCController(
                A=A,
                B=B,
                horizon=cfg.mpc_horizon,
                q_diag=q_diag,
                r_diag=r_diag,
                terminal_weight=cfg.mpc_terminal_weight,
                u_max=cfg.max_u,
                com_radius_m=cfg.mpc_com_constraint_radius_m,
                angle_max_rad=cfg.crash_angle_rad,
                verbose=cfg.mpc_verbose,
            )
            print("\n=== MPC CONTROLLER INITIALIZED ===")
            print(f"Horizon: {cfg.mpc_horizon} steps ({cfg.mpc_horizon / cfg.control_hz * 1000:.1f} ms)")
            print(f"Q diagonal: angles={cfg.mpc_q_angles}, rates={cfg.mpc_q_rates}, pos={cfg.mpc_q_position}")
            print(f"R diagonal: control={cfg.mpc_r_control}")
            print(f"Terminal weight: {cfg.mpc_terminal_weight:.2f}")
            print(
                "Target shaping: "
                f"run_rate_gain={cfg.mpc_target_rate_gain:.2f} "
                f"terminal_rate_gain={cfg.mpc_terminal_rate_gain:.2f} "
                f"clip={cfg.mpc_target_rate_clip_rad_s:.2f}rad/s"
            )
            print(f"COM constraint radius: {cfg.mpc_com_constraint_radius_m:.3f} m")
            print(
                "Pitch anti-drift I: "
                f"gain={cfg.mpc_pitch_i_gain:.2f} "
                f"clamp={cfg.mpc_pitch_i_clamp:.3f}rad*s "
                f"deadband={np.degrees(cfg.mpc_pitch_i_deadband_rad):.3f}deg "
                f"leak={cfg.mpc_pitch_i_leak_per_s:.2f}/s"
            )
            print(
                "Pitch rescue guard: "
                f"angle_frac={cfg.mpc_pitch_guard_angle_frac:.2f} "
                f"rate={cfg.mpc_pitch_guard_rate_entry_rad_s:.2f}rad/s "
                f"kp={cfg.mpc_pitch_guard_kp:.1f} kd={cfg.mpc_pitch_guard_kd:.1f} "
                f"max_frac={cfg.mpc_pitch_guard_max_frac:.2f}"
            )
            print(
                "Roll anti-drift I: "
                f"gain={cfg.mpc_roll_i_gain:.2f} "
                f"clamp={cfg.mpc_roll_i_clamp:.3f}rad*s "
                f"deadband={np.degrees(cfg.mpc_roll_i_deadband_rad):.3f}deg "
                f"leak={cfg.mpc_roll_i_leak_per_s:.2f}/s"
            )
        except Exception as e:
            print(f"\nWarning: MPC initialization failed: {e}")
            print("Falling back to LQR control")
            mpc_controller = None

    print("\n=== LINEARIZATION ===")

    print(f"A shape: {A.shape}, B shape: {B.shape}")
    print(f"A eigenvalues: {np.linalg.eigvals(A)}")
    ctrb_rank = _controllability_rank(A, B)
    print(f"Controllability rank: {ctrb_rank}/{A.shape[0]}")
    if ctrb_rank < A.shape[0]:
        print("Warning: model has uncontrollable modes (drift can remain despite tuning).")
    print("\n=== DELTA-U LQR ===")
    print(f"K_du shape: {K_du.shape}")
    print(
        "Linearization operating point: "
        f"pitch={np.degrees(cfg.linearize_pitch_rad):.2f}deg "
        f"roll={np.degrees(cfg.linearize_roll_rad):.2f}deg"
    )
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
    print(f"lock_root_attitude={cfg.lock_root_attitude}")
    print(f"seed={cfg.seed}")
    print(
        "Viewer interaction: "
        "left-double-click selects a body; hold Ctrl and drag to apply force/torque; "
        "drag without Ctrl moves the camera."
    )
    print(f"Viewer perturb helpers: planar_perturb={args.planar_perturb} drag_assist={args.drag_assist}")
    print(
        f"hardware_realistic={cfg.hardware_realistic} control_hz={cfg.control_hz:.1f} "
        f"delay_steps={cfg.control_delay_steps} wheel_ticks={cfg.wheel_encoder_ticks_per_rev}"
    )
    print(
        f"dynamics_dt_model={model.opt.timestep:.6f}s "
        f"design_dt_control={control_dt:.6f}s "
        f"(control_steps={control_steps})"
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
    print(f"Wheel spin thresholds (runtime): budget={wheel_budget_speed:.2f}rad/s hard={wheel_hard_speed:.2f}rad/s")
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
    print(
        "Sensor frontend: "
        f"requested={cfg.sensor_source} resolved={sensor_source} "
        f"mujoco_available={mujoco_sensor_available} "
        f"sample_hz={cfg.sensor_hz:.1f} delay_steps={cfg.sensor_delay_steps}"
    )
    if cfg.sensor_source == "mujoco" and sensor_source != "mujoco":
        print("Warning: requested --sensor-source mujoco but required named sensors were missing; using direct fallback.")
    print(
        "Sensor limits: "
        f"angle_clip={np.degrees(cfg.imu_angle_clip_rad):.1f}deg "
        f"rate_clip={cfg.imu_rate_clip_rad_s:.2f}rad/s "
        f"wheel_clip={cfg.wheel_rate_clip_rad_s:.2f}rad/s"
    )
    print(
        "Sensor LPF: "
        f"angle={cfg.imu_angle_lpf_hz:.1f}Hz rate={cfg.imu_rate_lpf_hz:.1f}Hz "
        f"wheel={cfg.wheel_rate_lpf_hz:.1f}Hz base_pos={cfg.base_pos_lpf_hz:.1f}Hz base_vel={cfg.base_vel_lpf_hz:.1f}Hz"
    )
    print(f"XML ctrlrange low: {XML_CTRL_LOW}")
    print(f"XML ctrlrange high: {XML_CTRL_HIGH}")
    print("\n=== RESIDUAL POLICY ===")
    print(f"residual_model={cfg.residual_model_path}")
    print(f"residual_scale={cfg.residual_scale:.3f}")
    print(f"residual_max_abs_u={cfg.residual_max_abs_u}")
    print(
        "residual_gate: "
        f"tilt={np.degrees(cfg.residual_gate_tilt_rad):.2f}deg "
        f"rate={cfg.residual_gate_rate_rad_s:.2f}rad/s"
    )
    print(f"residual_status={residual_policy.status}")
    print("\n=== ADAPTIVE DISTURBANCE OBSERVER ===")
    print(f"dob_enabled={cfg.dob_enabled}")
    if cfg.dob_enabled:
        dob_cutoff_hz_eff = float(cfg.dob_gain / (2.0 * np.pi)) if cfg.dob_gain > 0.0 else 0.0
        print(
            "dob_params: "
            f"gain={cfg.dob_gain:.2f}/s cutoff~{dob_cutoff_hz_eff:.2f}Hz "
            f"leak={cfg.dob_leak_per_s:.2f}/s "
            f"max_abs={cfg.dob_max_abs_u}"
        )
    print(f"gain_schedule_enabled={cfg.gain_schedule_enabled}")
    if cfg.gain_schedule_enabled:
        print(
            "gain_schedule: "
            f"min={cfg.gain_schedule_min:.2f} max={cfg.gain_schedule_max:.2f} "
            f"ref={cfg.gain_schedule_disturbance_ref:.3f} "
            f"rate={cfg.gain_schedule_rate_per_s:.2f}/s "
            f"weights={cfg.gain_schedule_weights}"
        )
    print("\n=== ONLINE ID + ADAPTIVE LQR ===")
    print(f"online_id_enabled={cfg.online_id_enabled}")
    if cfg.online_id_enabled:
        if adaptive_scheduler is None:
            print("online_id_status=disabled_for_current_mode")
        else:
            print(
                "online_id_params: "
                f"forget={cfg.online_id_forgetting:.6f} "
                f"recompute_every={cfg.online_id_recompute_every} "
                f"min_updates={cfg.online_id_min_updates} "
                f"g_scale=[{cfg.online_id_gravity_scale_min:.2f},{cfg.online_id_gravity_scale_max:.2f}] "
                f"i_inv_scale=[{cfg.online_id_inertia_inv_scale_min:.2f},{cfg.online_id_inertia_inv_scale_max:.2f}] "
                f"blend={cfg.online_id_gain_blend_alpha:.2f}"
            )
    print(f"Initial Y-direction tilt (roll): {args.initial_y_tilt_deg:.2f} deg")
    print(
        "Payload: "
        f"mass={payload_mass_kg:.3f}kg support_radius={cfg.payload_support_radius_m:.3f}m "
        f"com_fail_steps={cfg.payload_com_fail_steps}"
    )
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
    if cfg.use_mpc and cfg.gain_schedule_enabled:
        print("Note: gain scheduling scales LQR terms only; MPC path currently uses DOB feed-forward without LQR gain scaling.")

    reset_state(model, data, ids.q_pitch, ids.q_roll, pitch_eq=0.0, roll_eq=initial_roll_rad)
    if cfg.lock_root_attitude:
        enforce_planar_root_attitude(model, data, ids)
    upright_blend = 0.0
    despin_gain = 0.25
    rng = np.random.default_rng(cfg.seed)
    sensor_frontend_state = create_sensor_frontend_state(cfg)
    reset_sensor_frontend_state(sensor_frontend_state)

    queue_len = 1 if not cfg.hardware_realistic else cfg.control_delay_steps + 1
    (
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
    ) = reset_controller_buffers(NX, NU, queue_len)
    dob_hat = np.zeros(NU, dtype=float)
    dob_raw = np.zeros(NU, dtype=float)
    dob_prev_x_est = None
    gain_schedule_scale_state = 1.0

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
    wheel_speed_abs_sum = 0.0
    wheel_speed_abs_max = 0.0
    wheel_speed_pos_peak = 0.0
    wheel_speed_neg_peak = 0.0
    wheel_speed_samples = 0
    wheel_over_budget_pos_steps = 0
    wheel_over_budget_neg_steps = 0
    wheel_over_hard_pos_steps = 0
    wheel_over_hard_neg_steps = 0
    wheel_hard_same_dir_request_count = 0
    wheel_hard_safe_output_count = 0
    residual_applied_count = 0
    residual_clipped_count = 0
    residual_gate_blocked_count = 0
    residual_max_abs = np.zeros(NU, dtype=float)
    dob_disturbance_level_max = 0.0
    gain_schedule_scale_max = 1.0
    gain_schedule_scale_accum = 0.0
    com_planar_dist = compute_robot_com_distance_xy(model, data, ids.base_y_body_id)
    max_com_planar_dist = com_planar_dist
    com_over_support_steps = 0
    com_fail_streak = 0
    com_overload_failures = 0
    pitch_recovery_deadline_step = None
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
                "term_dob_comp_rw",
                "term_dob_comp_bx",
                "term_dob_comp_by",
                "term_residual_rw",
                "term_residual_bx",
                "term_residual_by",
                "dob_hat_rw",
                "dob_hat_bx",
                "dob_hat_by",
                "dob_raw_rw",
                "dob_raw_bx",
                "dob_raw_by",
                "gain_schedule_scale",
                "disturbance_level",
                "wheel_rate",
                "wheel_rate_abs",
                "wheel_budget_speed",
                "wheel_hard_speed",
                "wheel_over_budget",
                "wheel_over_hard",
                "high_spin_active",
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
                "crash_reason",
                "controller_family",
                "balance_phase",
                "pitch",
                "roll",
                "pitch_rate",
                "roll_rate",
                "wheel_rate",
                "wheel_rate_abs",
                "wheel_budget_speed",
                "wheel_hard_speed",
                "wheel_over_budget",
                "wheel_over_hard",
                "high_spin_active",
                "base_x",
                "base_y",
                "u_rw",
                "u_bx",
                "u_by",
                "term_dob_comp_rw",
                "term_dob_comp_bx",
                "term_dob_comp_by",
                "dob_hat_rw",
                "dob_hat_bx",
                "dob_hat_by",
                "gain_schedule_scale",
                "disturbance_level",
                "payload_mass_kg",
                "com_planar_dist_m",
                "com_over_support",
            ],
        )
        trace_events_writer.writeheader()

    # 5) Closed-loop runtime: estimate -> control -> clamp -> simulate -> render.
    with mujoco.viewer.launch_passive(model, data) as viewer:
        with viewer.lock():
            # Hide MuJoCo's default perturb box/force markers; we render a directional arrow instead.
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTOBJ] = 0
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = 0
            viewer.user_scn.ngeom = 0
        while viewer.is_running():
            step_count += 1
            drag_anchor_body = -1
            drag_force_world = np.zeros(3, dtype=float)
            drag_anchor_world = None
            # Pull GUI events/perturbations before control+step so dragging is applied immediately.
            viewer.sync()
            if np.any(prev_script_force):
                data.xfrc_applied[push_body_id, :3] -= prev_script_force
                prev_script_force[:] = 0.0
            selected_body = int(viewer.perturb.select)
            drag_wrench_world = np.zeros(6, dtype=float)
            if 0 < selected_body < model.nbody:
                drag_wrench_world = np.array(data.xfrc_applied[selected_body, :6], dtype=float)
            # Route mouse drag wrench to root body so dragging any clicked part moves the whole robot assembly.
            data.xfrc_applied[:, :] = 0.0
            if args.planar_perturb:
                drag_wrench_world[2] = 0.0
            if np.any(np.abs(drag_wrench_world) > 1e-9):
                data.xfrc_applied[ids.base_x_body_id, :6] = drag_wrench_world
                drag_anchor_body = selected_body
                drag_force_world = np.array(drag_wrench_world[:3], dtype=float)
                if 0 < selected_body < model.nbody:
                    local_hit = np.array(viewer.perturb.localpos, dtype=float)
                    if np.all(np.isfinite(local_hit)):
                        rot = data.xmat[selected_body].reshape(3, 3)
                        drag_anchor_world = np.array(data.xpos[selected_body], dtype=float) + rot @ local_hit
            if cfg.wheel_only:
                enforce_wheel_only_constraints(model, data, ids, lock_root_attitude=cfg.lock_root_attitude)
            else:
                if cfg.lock_root_attitude:
                    enforce_planar_root_attitude(model, data, ids)

            x_true = get_true_state(data, ids)
            com_planar_dist = compute_robot_com_distance_xy(model, data, ids.base_y_body_id)
            max_com_planar_dist = max(max_com_planar_dist, com_planar_dist)

            if not np.all(np.isfinite(x_true)):
                print(f"\nNumerical instability at step {step_count}; resetting state.")
                reset_state(model, data, ids.q_pitch, ids.q_roll, pitch_eq=0.0, roll_eq=initial_roll_rad)
                if cfg.lock_root_attitude:
                    enforce_planar_root_attitude(model, data, ids)
                com_fail_streak = 0
                (
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
                ) = reset_controller_buffers(NX, NU, queue_len)
                dob_hat[:] = 0.0
                dob_raw[:] = 0.0
                dob_prev_x_est = None
                adaptive_prev_x_est = None
                gain_schedule_scale_state = 1.0
                pitch_recovery_deadline_step = None
                reset_sensor_frontend_state(sensor_frontend_state)
                continue

            # Estimator predictor runs at simulation step; control gains are designed on lifted control-step model.
            x_pred = A_step @ x_est + B_step @ u_eff_applied
            x_est = x_pred

            if step_count % control_steps == 0:
                control_updates += 1
                x_est = estimator_measurement_update(
                    cfg,
                    x_true,
                    x_pred,
                    C,
                    L,
                    rng,
                    wheel_lsb,
                    data=data,
                    sensor_ids=sensor_ids,
                    sensor_source=sensor_source,
                    sensor_state=sensor_frontend_state,
                    sim_time_s=float(data.time),
                    control_dt=control_dt,
                )
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
                disturbance_level = 0.0
                if cfg.dob_enabled:
                    dob_hat, dob_raw = update_disturbance_observer(
                        cfg=cfg,
                        A=A,
                        B=B,
                        B_pinv=B_pinv,
                        x_prev=dob_prev_x_est,
                        u_prev=u_eff_applied,
                        x_curr=x_est,
                        dob_hat=dob_hat,
                        control_dt=control_dt,
                    )
                    if cfg.gain_schedule_enabled:
                        gain_schedule_scale_state, disturbance_level = update_gain_schedule(
                            cfg=cfg,
                            dob_hat=dob_hat,
                            gain_schedule_state=gain_schedule_scale_state,
                            control_dt=control_dt,
                        )
                    else:
                        gain_schedule_scale_state = 1.0
                        disturbance_level = float(np.linalg.norm(cfg.gain_schedule_weights * dob_hat))
                else:
                    dob_hat[:] = 0.0
                    dob_raw[:] = 0.0
                    gain_schedule_scale_state = 1.0
                if adaptive_scheduler is not None:
                    K_du, k_updated = adaptive_scheduler.maybe_update_gain(
                        x_prev=adaptive_prev_x_est,
                        u_prev=u_eff_applied,
                        x_curr=x_est,
                        k_current=K_du,
                    )
                    if cfg.online_id_verbose and k_updated:
                        s = adaptive_scheduler.stats
                        print(
                            f"[online-id] update={control_updates} "
                            f"g_scale={s.gravity_scale:.3f} "
                            f"i_inv_scale={s.inertia_inv_scale:.3f} "
                            f"rls_updates={s.rls_updates} "
                            f"gain_recomputes={s.gain_recomputes}"
                        )
                (
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
                    wheel_momentum_bias_int=wheel_momentum_bias_int,
                    balance_phase=balance_phase,
                    recovery_time_s=recovery_time_s,
                    high_spin_active=high_spin_active,
                    control_dt=control_dt,
                    K_du=K_du,
                    K_wheel_only=K_wheel_only,
                    K_paper_pitch=K_paper_pitch,
                    du_hits=du_hits,
                    sat_hits=sat_hits,
                    dob_compensation=dob_hat,
                    gain_schedule_scale=gain_schedule_scale_state,
                    disturbance_level=disturbance_level,
                    mpc_controller=mpc_controller,
                )
                dob_prev_x_est = x_est.copy()
                adaptive_prev_x_est = x_est.copy()
                dob_disturbance_level_max = max(dob_disturbance_level_max, disturbance_level)
                gain_schedule_scale_max = max(gain_schedule_scale_max, gain_schedule_scale_state)
                gain_schedule_scale_accum += gain_schedule_scale_state
                residual_step = residual_policy.step(
                    x_est=x_est,
                    x_true=x_true,
                    u_nominal=u_cmd,
                    u_eff_applied=u_eff_applied,
                )
                residual_delta = residual_step.delta_u
                residual_gate_blocked_count += int(residual_step.gate_blocked)
                if residual_step.applied:
                    residual_applied_count += 1
                    residual_clipped_count += int(residual_step.clipped)
                    residual_max_abs = np.maximum(residual_max_abs, np.abs(residual_delta))
                    u_cmd = u_cmd + residual_delta
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
                            "term_dob_comp_rw": float(control_terms["term_disturbance_comp"][0]),
                            "term_dob_comp_bx": float(control_terms["term_disturbance_comp"][1]),
                            "term_dob_comp_by": float(control_terms["term_disturbance_comp"][2]),
                            "term_residual_rw": float(residual_delta[0]),
                            "term_residual_bx": float(residual_delta[1]),
                            "term_residual_by": float(residual_delta[2]),
                            "dob_hat_rw": float(dob_hat[0]),
                            "dob_hat_bx": float(dob_hat[1]),
                            "dob_hat_by": float(dob_hat[2]),
                            "dob_raw_rw": float(dob_raw[0]),
                            "dob_raw_bx": float(dob_raw[1]),
                            "dob_raw_by": float(dob_raw[2]),
                            "gain_schedule_scale": float(control_terms["gain_schedule_scale"][0]),
                            "disturbance_level": float(control_terms["disturbance_level"][0]),
                            "wheel_rate": float(x_true[4]),
                            "wheel_rate_abs": float(abs(x_true[4])),
                            "wheel_budget_speed": float(wheel_budget_speed),
                            "wheel_hard_speed": float(wheel_hard_speed),
                            "wheel_over_budget": int(abs(float(x_true[4])) > wheel_budget_speed),
                            "wheel_over_hard": int(abs(float(x_true[4])) > wheel_hard_speed),
                            "high_spin_active": int(high_spin_active),
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
                            "wheel_rate_abs": float(abs(x_true[4])),
                            "wheel_budget_speed": float(wheel_budget_speed),
                            "wheel_hard_speed": float(wheel_hard_speed),
                            "wheel_over_budget": int(abs(float(x_true[4])) > wheel_budget_speed),
                            "wheel_over_hard": int(abs(float(x_true[4])) > wheel_hard_speed),
                            "high_spin_active": int(high_spin_active),
                            "base_x": float(x_true[5]),
                            "base_y": float(x_true[6]),
                            "u_rw": float(u_cmd[0]),
                            "u_bx": float(u_cmd[1]),
                            "u_by": float(u_cmd[2]),
                            "term_dob_comp_rw": float(control_terms["term_disturbance_comp"][0]),
                            "term_dob_comp_bx": float(control_terms["term_disturbance_comp"][1]),
                            "term_dob_comp_by": float(control_terms["term_disturbance_comp"][2]),
                            "dob_hat_rw": float(dob_hat[0]),
                            "dob_hat_bx": float(dob_hat[1]),
                            "dob_hat_by": float(dob_hat[2]),
                            "gain_schedule_scale": float(control_terms["gain_schedule_scale"][0]),
                            "disturbance_level": float(control_terms["disturbance_level"][0]),
                            "payload_mass_kg": float(payload_mass_kg),
                            "com_planar_dist_m": float(com_planar_dist),
                            "com_over_support": int(com_planar_dist > cfg.payload_support_radius_m),
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
            if abs(wheel_speed) >= wheel_hard_speed and abs(float(u_drag[0])) > 1e-9 and np.sign(float(u_drag[0])) == np.sign(
                wheel_speed
            ):
                wheel_hard_same_dir_request_count += 1
            wheel_cmd = wheel_command_with_limits(cfg, wheel_speed, float(u_drag[0]))
            if abs(wheel_speed) >= wheel_hard_speed:
                if abs(wheel_cmd) <= 1e-9 or (abs(wheel_speed) > 1e-9 and np.sign(wheel_cmd) != np.sign(wheel_speed)):
                    wheel_hard_safe_output_count += 1
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
                enforce_wheel_only_constraints(model, data, ids, lock_root_attitude=cfg.lock_root_attitude)
            else:
                if cfg.lock_root_attitude:
                    enforce_planar_root_attitude(model, data, ids)
            wheel_speed_after = float(data.qvel[ids.v_rw])
            wheel_speed_abs_after = abs(wheel_speed_after)
            wheel_speed_abs_sum += wheel_speed_abs_after
            wheel_speed_abs_max = max(wheel_speed_abs_max, wheel_speed_abs_after)
            wheel_speed_pos_peak = max(wheel_speed_pos_peak, wheel_speed_after)
            wheel_speed_neg_peak = min(wheel_speed_neg_peak, wheel_speed_after)
            wheel_speed_samples += 1
            if wheel_speed_abs_after > wheel_budget_speed:
                if wheel_speed_after >= 0.0:
                    wheel_over_budget_pos_steps += 1
                else:
                    wheel_over_budget_neg_steps += 1
            if wheel_speed_abs_after > wheel_hard_speed:
                if wheel_speed_after >= 0.0:
                    wheel_over_hard_pos_steps += 1
                else:
                    wheel_over_hard_neg_steps += 1
            speed_limit_hits[0] += int(abs(data.qvel[ids.v_rw]) > cfg.max_wheel_speed_rad_s)
            speed_limit_hits[1] += int(abs(data.qvel[ids.v_pitch]) > cfg.max_pitch_roll_rate_rad_s)
            speed_limit_hits[2] += int(abs(data.qvel[ids.v_roll]) > cfg.max_pitch_roll_rate_rad_s)
            speed_limit_hits[3] += int(abs(data.qvel[ids.v_base_x]) > cfg.max_base_speed_m_s)
            speed_limit_hits[4] += int(abs(data.qvel[ids.v_base_y]) > cfg.max_base_speed_m_s)
            com_planar_dist = compute_robot_com_distance_xy(model, data, ids.base_y_body_id)
            max_com_planar_dist = max(max_com_planar_dist, com_planar_dist)
            if com_planar_dist > cfg.payload_support_radius_m:
                com_over_support_steps += 1
                com_fail_streak += 1
            else:
                com_fail_streak = 0
            _render_drag_force_arrow(viewer, data, drag_anchor_body, drag_force_world, anchor_world=drag_anchor_world)
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
                    f"u_bx={u_eff_applied[1]:7.2f} u_by={u_eff_applied[2]:7.2f} "
                    f"com_xy={com_planar_dist:6.3f}m"
                )

            pitch_rate_now = float(data.qvel[ids.v_pitch])
            roll_rate_now = float(data.qvel[ids.v_roll])
            wheel_rate_now = float(data.qvel[ids.v_rw])

            pitch_failed = abs(pitch) >= cfg.crash_angle_rad
            roll_failed = abs(roll) >= cfg.crash_angle_rad
            pitch_crash_confirmed = False
            pitch_crash_reason = "pitch_tilt"
            if pitch_failed:
                pitch_diverging = (abs(pitch_rate_now) > 1e-9) and (np.sign(pitch_rate_now) == np.sign(pitch))
                if (not cfg.crash_divergence_gate_enabled) or pitch_diverging:
                    pitch_crash_confirmed = True
                    pitch_crash_reason = "pitch_tilt_diverging" if cfg.crash_divergence_gate_enabled else "pitch_tilt"
                    pitch_recovery_deadline_step = None
                elif cfg.crash_recovery_window_steps <= 0:
                    pitch_crash_confirmed = True
                    pitch_crash_reason = "pitch_tilt_recovery_disabled"
                    pitch_recovery_deadline_step = None
                else:
                    if pitch_recovery_deadline_step is None:
                        pitch_recovery_deadline_step = step_count + cfg.crash_recovery_window_steps
                    if step_count >= pitch_recovery_deadline_step:
                        pitch_crash_confirmed = True
                        pitch_crash_reason = "pitch_tilt_recovery_timeout"
                        pitch_recovery_deadline_step = None
            else:
                pitch_recovery_deadline_step = None

            tilt_failed = pitch_crash_confirmed or roll_failed
            com_failed = com_fail_streak >= cfg.payload_com_fail_steps
            if tilt_failed or com_failed:
                crash_count += 1
                com_overload_failures += int(com_failed)
                pitch_rate_at_crash = pitch_rate_now
                roll_rate_at_crash = roll_rate_now
                wheel_rate_at_crash = wheel_rate_now
                crash_reason = "com_overload" if com_failed else (pitch_crash_reason if pitch_crash_confirmed else "roll_tilt")
                if trace_events_writer is not None:
                    trace_events_writer.writerow(
                        {
                            "step": step_count,
                            "sim_time_s": float(data.time),
                            "event": "crash",
                            "crash_reason": crash_reason,
                            "controller_family": cfg.controller_family,
                            "balance_phase": balance_phase,
                            "pitch": float(pitch),
                            "roll": float(roll),
                            "pitch_rate": pitch_rate_at_crash,
                            "roll_rate": roll_rate_at_crash,
                            "wheel_rate": wheel_rate_at_crash,
                            "wheel_rate_abs": float(abs(wheel_rate_at_crash)),
                            "wheel_budget_speed": float(wheel_budget_speed),
                            "wheel_hard_speed": float(wheel_hard_speed),
                            "wheel_over_budget": int(abs(wheel_rate_at_crash) > wheel_budget_speed),
                            "wheel_over_hard": int(abs(wheel_rate_at_crash) > wheel_hard_speed),
                            "high_spin_active": int(high_spin_active),
                            "base_x": float(data.qpos[ids.q_base_x]),
                            "base_y": float(data.qpos[ids.q_base_y]),
                            "u_rw": float(u_eff_applied[0]),
                            "u_bx": float(u_eff_applied[1]),
                            "u_by": float(u_eff_applied[2]),
                            "payload_mass_kg": float(payload_mass_kg),
                            "com_planar_dist_m": float(com_planar_dist),
                            "com_over_support": int(com_planar_dist > cfg.payload_support_radius_m),
                        }
                    )
                print(
                    f"\nCRASH #{crash_count} at step {step_count}: "
                    f"pitch={np.degrees(pitch):.2f}deg roll={np.degrees(roll):.2f}deg "
                    f"pitch_rate={pitch_rate_at_crash:.3f}rad/s "
                    f"roll_rate={roll_rate_at_crash:.3f}rad/s "
                    f"wheel_rate={wheel_rate_at_crash:.2f}rad/s "
                    f"com_xy={com_planar_dist:.3f}m "
                    f"reason={crash_reason}"
                )
                print(f"CRASH_PITCH_RATE_RAD_S={pitch_rate_at_crash:+.3f}")
                if cfg.stop_on_crash:
                    break
                reset_state(model, data, ids.q_pitch, ids.q_roll, pitch_eq=0.0, roll_eq=initial_roll_rad)
                if cfg.lock_root_attitude:
                    enforce_planar_root_attitude(model, data, ids)
                com_fail_streak = 0
                (
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
                ) = reset_controller_buffers(NX, NU, queue_len)
                dob_hat[:] = 0.0
                dob_raw[:] = 0.0
                dob_prev_x_est = None
                adaptive_prev_x_est = None
                gain_schedule_scale_state = 1.0
                pitch_recovery_deadline_step = None
                reset_sensor_frontend_state(sensor_frontend_state)
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
    print(f"Payload mass: {payload_mass_kg:.3f}kg")
    print(f"Max COM planar distance: {max_com_planar_dist:.3f}m")
    print(f"COM over-support ratio: {com_over_support_steps / max(step_count, 1):.3f}")
    print(f"COM overload crash count: {com_overload_failures}")
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
    wheel_sample_denom = max(wheel_speed_samples, 1)
    wheel_over_budget_steps_total = wheel_over_budget_pos_steps + wheel_over_budget_neg_steps
    wheel_over_hard_steps_total = wheel_over_hard_pos_steps + wheel_over_hard_neg_steps
    print(
        f"Wheel speed peaks [max_pos,min_neg,max_abs]: "
        f"{wheel_speed_pos_peak:.2f}, {wheel_speed_neg_peak:.2f}, {wheel_speed_abs_max:.2f} rad/s"
    )
    print(f"Wheel mean |speed|: {wheel_speed_abs_sum / wheel_sample_denom:.2f} rad/s")
    print(
        "Wheel over-budget ratio [total,pos,neg]: "
        f"{wheel_over_budget_steps_total / wheel_sample_denom:.3f}, "
        f"{wheel_over_budget_pos_steps / wheel_sample_denom:.3f}, "
        f"{wheel_over_budget_neg_steps / wheel_sample_denom:.3f}"
    )
    print(
        "Wheel over-hard ratio [total,pos,neg]: "
        f"{wheel_over_hard_steps_total / wheel_sample_denom:.3f}, "
        f"{wheel_over_hard_pos_steps / wheel_sample_denom:.3f}, "
        f"{wheel_over_hard_neg_steps / wheel_sample_denom:.3f}"
    )
    print(f"Wheel hard-zone same-direction request count: {wheel_hard_same_dir_request_count}")
    print(f"Wheel hard-zone safe-output count: {wheel_hard_safe_output_count}")
    if cfg.dob_enabled:
        print(f"DOb max disturbance level: {dob_disturbance_level_max:.4f}")
        print(f"DOb final estimate [rw,bx,by]: {dob_hat}")
    if cfg.gain_schedule_enabled:
        print(f"Gain schedule max scale: {gain_schedule_scale_max:.3f}")
        print(f"Gain schedule mean scale: {gain_schedule_scale_accum / denom:.3f}")
    if adaptive_scheduler is not None:
        s = adaptive_scheduler.stats
        print(
            "Online ID stats: "
            f"rls_updates={s.rls_updates} "
            f"skipped={s.rls_skipped} "
            f"gain_recomputes={s.gain_recomputes} "
            f"gain_failures={s.gain_recompute_failures}"
        )
        print(
            "Online ID estimates: "
            f"gravity_scale={s.gravity_scale:.3f} "
            f"inertia_inv_scale={s.inertia_inv_scale:.3f} "
            f"innovation_rms={s.innovation_rms:.4f}"
        )
    print(f"Residual applied rate [updates]: {residual_applied_count / denom:.3f}")
    print(f"Residual clipped count: {residual_clipped_count}")
    print(f"Residual gate-blocked count: {residual_gate_blocked_count}")
    print(f"Residual max |delta_u| [rw,bx,by]: {residual_max_abs}")


if __name__ == "__main__":
    main()


