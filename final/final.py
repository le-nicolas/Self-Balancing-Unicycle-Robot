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
    compute_robot_com_distance_xy,
    enforce_wheel_only_constraints,
    estimator_measurement_update,
    get_true_state,
    lookup_model_ids,
    reset_state,
    set_payload_mass,
)
from control_core import (
    apply_control_delay,
    apply_upright_postprocess,
    base_commands_with_limits,
    compute_control_command,
    reset_controller_buffers,
    wheel_command_with_limits,
)
from residual_model import ResidualPolicy

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
    payload_mass_kg = set_payload_mass(model, data, ids, cfg.payload_mass_kg)
    push_body_id = {
        "stick": ids.stick_body_id,
        "base_y": ids.base_y_body_id,
        "base_x": ids.base_x_body_id,
        "payload": ids.payload_body_id,
    }[args.push_body]
    reset_state(model, data, ids.q_pitch, ids.q_roll, pitch_eq=0.0, roll_eq=initial_roll_rad)

    # 3) Linearize XML-defined dynamics about upright and build controller gains.
    nx = 2 * model.nv + model.na
    nu = model.nu
    A_full = np.zeros((nx, nx))
    B_full = np.zeros((nx, nu))
    mujoco.mjd_transitionFD(model, data, 1e-6, True, A_full, B_full, None, None)

    idx = [
        ids.q_pitch,
        ids.q_roll,
        model.nv + ids.v_pitch,
        model.nv + ids.v_roll,
        model.nv + ids.v_rw,
        ids.q_base_x,
        ids.q_base_y,
        model.nv + ids.v_base_x,
        model.nv + ids.v_base_y,
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
                u_max=cfg.max_u,
                com_radius_m=cfg.mpc_com_constraint_radius_m,
                angle_max_rad=cfg.crash_angle_rad,
                verbose=cfg.mpc_verbose,
            )
            print("\n=== MPC CONTROLLER INITIALIZED ===")
            print(f"Horizon: {cfg.mpc_horizon} steps ({cfg.mpc_horizon / cfg.control_hz * 1000:.1f} ms)")
            print(f"Q diagonal: angles={cfg.mpc_q_angles}, rates={cfg.mpc_q_rates}, pos={cfg.mpc_q_position}")
            print(f"R diagonal: control={cfg.mpc_r_control}")
            print(f"COM constraint radius: {cfg.mpc_com_constraint_radius_m:.3f} m")
        except Exception as e:
            print(f"\nWarning: MPC initialization failed: {e}")
            print("Falling back to LQR control")
            mpc_controller = None

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
    com_planar_dist = compute_robot_com_distance_xy(model, data, ids.base_y_body_id)
    max_com_planar_dist = com_planar_dist
    com_over_support_steps = 0
    com_fail_streak = 0
    com_overload_failures = 0
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
                "term_residual_rw",
                "term_residual_bx",
                "term_residual_by",
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
                "payload_mass_kg",
                "com_planar_dist_m",
                "com_over_support",
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
            com_planar_dist = compute_robot_com_distance_xy(model, data, ids.base_y_body_id)
            max_com_planar_dist = max(max_com_planar_dist, com_planar_dist)

            if not np.all(np.isfinite(x_true)):
                print(f"\nNumerical instability at step {step_count}; resetting state.")
                reset_state(model, data, ids.q_pitch, ids.q_roll, pitch_eq=0.0, roll_eq=initial_roll_rad)
                com_fail_streak = 0
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
                    mpc_controller=mpc_controller,
                )
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
                            "term_residual_rw": float(residual_delta[0]),
                            "term_residual_bx": float(residual_delta[1]),
                            "term_residual_by": float(residual_delta[2]),
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
                enforce_wheel_only_constraints(model, data, ids)
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

            tilt_failed = abs(pitch) >= cfg.crash_angle_rad or abs(roll) >= cfg.crash_angle_rad
            com_failed = com_fail_streak >= cfg.payload_com_fail_steps
            if tilt_failed or com_failed:
                crash_count += 1
                com_overload_failures += int(com_failed)
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
                            "wheel_rate_abs": float(abs(data.qvel[ids.v_rw])),
                            "wheel_budget_speed": float(wheel_budget_speed),
                            "wheel_hard_speed": float(wheel_hard_speed),
                            "wheel_over_budget": int(abs(float(data.qvel[ids.v_rw])) > wheel_budget_speed),
                            "wheel_over_hard": int(abs(float(data.qvel[ids.v_rw])) > wheel_hard_speed),
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
                    f"com_xy={com_planar_dist:.3f}m reason={'com_overload' if com_failed else 'tilt'}"
                )
                if cfg.stop_on_crash:
                    break
                reset_state(model, data, ids.q_pitch, ids.q_roll, pitch_eq=0.0, roll_eq=initial_roll_rad)
                com_fail_streak = 0
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
    print(f"Residual applied rate [updates]: {residual_applied_count / denom:.3f}")
    print(f"Residual clipped count: {residual_clipped_count}")
    print(f"Residual gate-blocked count: {residual_gate_blocked_count}")
    print(f"Residual max |delta_u| [rw,bx,by]: {residual_max_abs}")


if __name__ == "__main__":
    main()


