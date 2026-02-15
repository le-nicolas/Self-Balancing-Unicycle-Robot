import argparse
import hashlib
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from scipy.linalg import solve_discrete_are

import final as runtime


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export controller/estimator params as deterministic C header.")
    parser.add_argument("--mode", choices=["smooth", "robust"], default="smooth")
    parser.add_argument(
        "--real-hardware",
        action="store_true",
        help="Extra-conservative bring-up profile for physical hardware (forces strict limits and stop-on-crash).",
    )
    parser.add_argument("--hardware-safe", action="store_true", help="Use conservative real-hardware startup limits.")
    parser.add_argument("--out", type=str, default="final/firmware/controller_params.h")

    # Match final.py runtime defaults and semantics.
    parser.add_argument("--easy-mode", action="store_true")
    parser.add_argument("--stop-on-crash", action="store_true")
    parser.add_argument("--wheel-only", action="store_true")
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
    parser.add_argument("--enable-base-integrator", action="store_true")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--crash-angle-deg", type=float, default=25.0)
    parser.add_argument("--disturbance-mag", type=float, default=None)
    parser.add_argument("--disturbance-interval", type=int, default=None)
    parser.add_argument("--control-hz", type=float, default=250.0)
    parser.add_argument("--control-delay-steps", type=int, default=1)
    parser.add_argument("--wheel-encoder-ticks", type=int, default=2048)
    parser.add_argument("--imu-angle-noise-deg", type=float, default=0.25)
    parser.add_argument("--imu-rate-noise", type=float, default=0.02)
    parser.add_argument("--wheel-rate-noise", type=float, default=0.01)
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
    return parser


def _fmt_num(x: float) -> str:
    return f"{float(x):.9e}f"


def _fmt_array_1d(name: str, arr: np.ndarray) -> str:
    vals = ", ".join(_fmt_num(v) for v in arr)
    return f"static const float {name}[{arr.shape[0]}] = {{{vals}}};\n"


def _fmt_array_2d(name: str, arr: np.ndarray) -> str:
    rows = []
    for row in arr:
        rows.append("    {" + ", ".join(_fmt_num(v) for v in row) + "}")
    return f"static const float {name}[{arr.shape[0]}][{arr.shape[1]}] = {{\n" + ",\n".join(rows) + "\n};\n"


def _jid(model: mujoco.MjModel, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)


def _aid(model: mujoco.MjModel, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)


def linearize_model(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
    jid_pitch = _jid(model, "stick_pitch")
    jid_roll = _jid(model, "stick_roll")
    jid_rw = _jid(model, "wheel_spin")
    jid_base_x = _jid(model, "base_x_slide")
    jid_base_y = _jid(model, "base_y_slide")

    q_pitch = model.jnt_qposadr[jid_pitch]
    q_roll = model.jnt_qposadr[jid_roll]
    q_base_x = model.jnt_qposadr[jid_base_x]
    q_base_y = model.jnt_qposadr[jid_base_y]

    v_pitch = model.jnt_dofadr[jid_pitch]
    v_roll = model.jnt_dofadr[jid_roll]
    v_rw = model.jnt_dofadr[jid_rw]
    v_base_x = model.jnt_dofadr[jid_base_x]
    v_base_y = model.jnt_dofadr[jid_base_y]

    aid_rw = _aid(model, "wheel_spin")
    aid_base_x = _aid(model, "base_x_force")
    aid_base_y = _aid(model, "base_y_force")

    runtime.reset_state(model, data, q_pitch, q_roll)

    nx = model.nq + model.nv
    nu = model.nu
    a_full = np.zeros((nx, nx))
    b_full = np.zeros((nx, nu))
    mujoco.mjd_transitionFD(model, data, 1e-6, True, a_full, b_full, None, None)

    idx = [
        q_pitch,
        q_roll,
        model.nq + v_pitch,
        model.nq + v_roll,
        model.nq + v_rw,
        q_base_x,
        q_base_y,
        model.nq + v_base_x,
        model.nq + v_base_y,
    ]
    a = a_full[np.ix_(idx, idx)]
    b = b_full[np.ix_(idx, [aid_rw, aid_base_x, aid_base_y])]
    return a, b


def compute_export_bundle(args: argparse.Namespace) -> dict[str, Any]:
    cfg = runtime.build_config(args)
    xml_path = Path(__file__).with_name("final.xml")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    a, b = linearize_model(model, data)
    nx = a.shape[0]
    nu = b.shape[1]

    a_aug = np.block([[a, b], [np.zeros((nu, nx)), np.eye(nu)]])
    b_aug = np.vstack([b, np.eye(nu)])
    q_aug = np.block([[cfg.qx, np.zeros((nx, nu))], [np.zeros((nu, nx)), cfg.qu]])
    p_aug = solve_discrete_are(a_aug, b_aug, q_aug, cfg.r_du)
    k_du = np.linalg.inv(b_aug.T @ p_aug @ b_aug + cfg.r_du) @ (b_aug.T @ p_aug @ a_aug)

    c = runtime.build_partial_measurement_matrix()
    control_steps = 1 if not cfg.hardware_realistic else max(1, int(round(1.0 / (model.opt.timestep * cfg.control_hz))))
    control_dt = control_steps * model.opt.timestep
    wheel_lsb = (2.0 * np.pi) / (cfg.wheel_encoder_ticks_per_rev * control_dt)
    qn = np.diag([1e-4, 1e-4, 1e-4, 1e-4, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5])
    wheel_var = (wheel_lsb**2) / 12.0 + cfg.wheel_encoder_rate_noise_std_rad_s**2
    rn = np.diag(
        [
            cfg.imu_angle_noise_std_rad**2,
            cfg.imu_angle_noise_std_rad**2,
            cfg.imu_rate_noise_std_rad_s**2,
            cfg.imu_rate_noise_std_rad_s**2,
            wheel_var,
        ]
    )
    l = runtime.build_kalman_gain(a, qn, c, rn)

    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b)) and np.all(np.isfinite(k_du)) and np.all(np.isfinite(l))):
        raise RuntimeError("Non-finite values encountered in exported matrices.")

    act_ids = np.array([_aid(model, "wheel_spin"), _aid(model, "base_x_force"), _aid(model, "base_y_force")], dtype=int)
    xml_low = model.actuator_ctrlrange[act_ids, 0]
    xml_high = model.actuator_ctrlrange[act_ids, 1]

    xml_sha = hashlib.sha256(xml_path.read_bytes()).hexdigest()
    sig_hasher = hashlib.sha256()
    for arr in (a, b, c, k_du, l, cfg.max_u, cfg.max_du, cfg.qx, cfg.qu, cfg.r_du):
        sig_hasher.update(np.asarray(arr, dtype=np.float64).tobytes())
    sig_hasher.update(str(cfg).encode("utf-8"))

    return {
        "mode": args.mode,
        "cfg": cfg,
        "xml_path": xml_path,
        "xml_sha256": xml_sha,
        "generation_signature": sig_hasher.hexdigest(),
        "a": a,
        "b": b,
        "c": c,
        "k_du": k_du,
        "l": l,
        "qn": qn,
        "rn": rn,
        "control_steps": int(control_steps),
        "control_dt": float(control_dt),
        "wheel_lsb": float(wheel_lsb),
        "xml_ctrl_low": xml_low,
        "xml_ctrl_high": xml_high,
    }


def render_header(bundle: dict[str, Any], args: argparse.Namespace) -> str:
    cfg = bundle["cfg"]
    a = bundle["a"]
    b = bundle["b"]
    c = bundle["c"]
    k_du = bundle["k_du"]
    l = bundle["l"]

    lines = []
    lines.append("/*")
    lines.append(" * Deterministic controller parameter export for firmware.")
    lines.append(f" * profile_mode: {bundle['mode']}")
    lines.append(f" * source_xml: {bundle['xml_path'].as_posix()}")
    lines.append(f" * xml_sha256: {bundle['xml_sha256']}")
    lines.append(f" * generation_signature_sha256: {bundle['generation_signature']}")
    lines.append(f" * seed_metadata: {int(args.seed)}")
    lines.append(" */")
    lines.append("")
    lines.append("#ifndef CONTROLLER_PARAMS_H")
    lines.append("#define CONTROLLER_PARAMS_H")
    lines.append("")
    lines.append("#include <stdint.h>")
    lines.append("")
    lines.append(f"#define CTRL_PROFILE_MODE \"{bundle['mode']}\"")
    lines.append(f"#define CTRL_REAL_HARDWARE_PROFILE {1 if cfg.real_hardware_profile else 0}")
    lines.append(f"#define CTRL_HARDWARE_SAFE {1 if cfg.hardware_safe else 0}")
    lines.append(f"#define CTRL_WHEEL_ONLY {1 if cfg.wheel_only else 0}")
    lines.append(f"#define CTRL_ALLOW_BASE_MOTION {1 if cfg.allow_base_motion else 0}")
    lines.append(f"#define CTRL_XML_SHA256 \"{bundle['xml_sha256']}\"")
    lines.append(f"#define CTRL_GENERATION_SIGNATURE_SHA256 \"{bundle['generation_signature']}\"")
    lines.append(f"#define CTRL_NX {a.shape[0]}")
    lines.append(f"#define CTRL_NU {b.shape[1]}")
    lines.append(f"#define CTRL_NY {c.shape[0]}")
    lines.append(f"#define CTRL_DT_SIM {_fmt_num(bundle['control_dt'] / max(bundle['control_steps'], 1))}")
    lines.append(f"#define CTRL_CONTROL_STEPS {bundle['control_steps']}")
    lines.append(f"#define CTRL_HZ {_fmt_num(cfg.control_hz)}")
    lines.append(f"#define CTRL_DELAY_STEPS {int(cfg.control_delay_steps)}")
    lines.append(f"#define CTRL_HARDWARE_REALISTIC {1 if cfg.hardware_realistic else 0}")
    lines.append("")
    lines.append(_fmt_array_2d("CTRL_A", a))
    lines.append(_fmt_array_2d("CTRL_B", b))
    lines.append(_fmt_array_2d("CTRL_C", c))
    lines.append(_fmt_array_2d("CTRL_K_DU", k_du))
    lines.append(_fmt_array_2d("CTRL_L", l))
    lines.append(_fmt_array_1d("CTRL_MAX_U", cfg.max_u))
    lines.append(_fmt_array_1d("CTRL_MAX_DU", cfg.max_du))
    lines.append(_fmt_array_2d("CTRL_QX", cfg.qx))
    lines.append(_fmt_array_2d("CTRL_QU", cfg.qu))
    lines.append(_fmt_array_2d("CTRL_R_DU", cfg.r_du))
    lines.append(_fmt_array_1d("CTRL_XML_CTRL_LOW", bundle["xml_ctrl_low"]))
    lines.append(_fmt_array_1d("CTRL_XML_CTRL_HIGH", bundle["xml_ctrl_high"]))
    lines.append("")
    lines.append(f"static const float CTRL_KI_BASE = {_fmt_num(cfg.ki_base)};")
    lines.append(f"static const float CTRL_U_BLEED = {_fmt_num(cfg.u_bleed)};")
    lines.append(f"static const float CTRL_X_REF = {_fmt_num(cfg.x_ref)};")
    lines.append(f"static const float CTRL_Y_REF = {_fmt_num(cfg.y_ref)};")
    lines.append(f"static const float CTRL_INT_CLAMP = {_fmt_num(cfg.int_clamp)};")
    lines.append(f"static const float CTRL_CRASH_ANGLE_RAD = {_fmt_num(cfg.crash_angle_rad)};")
    lines.append(f"static const float CTRL_UPRIGHT_ANGLE_THRESH = {_fmt_num(cfg.upright_angle_thresh)};")
    lines.append(f"static const float CTRL_UPRIGHT_VEL_THRESH = {_fmt_num(cfg.upright_vel_thresh)};")
    lines.append(f"static const float CTRL_UPRIGHT_POS_THRESH = {_fmt_num(cfg.upright_pos_thresh)};")
    lines.append("")
    lines.append(f"static const float CTRL_IMU_ANGLE_NOISE_STD_RAD = {_fmt_num(cfg.imu_angle_noise_std_rad)};")
    lines.append(f"static const float CTRL_IMU_RATE_NOISE_STD_RAD_S = {_fmt_num(cfg.imu_rate_noise_std_rad_s)};")
    lines.append(
        f"static const float CTRL_WHEEL_RATE_NOISE_STD_RAD_S = {_fmt_num(cfg.wheel_encoder_rate_noise_std_rad_s)};"
    )
    lines.append(
        f"static const float CTRL_WHEEL_ENCODER_TICKS_PER_REV = {_fmt_num(float(cfg.wheel_encoder_ticks_per_rev))};"
    )
    lines.append(f"static const float CTRL_CONTROL_DT = {_fmt_num(bundle['control_dt'])};")
    lines.append(f"static const float CTRL_WHEEL_RATE_LSB_RAD_S = {_fmt_num(bundle['wheel_lsb'])};")
    lines.append("")
    lines.append(f"static const float CTRL_MAX_WHEEL_SPEED_RAD_S = {_fmt_num(cfg.max_wheel_speed_rad_s)};")
    lines.append(f"static const float CTRL_MAX_PITCH_ROLL_RATE_RAD_S = {_fmt_num(cfg.max_pitch_roll_rate_rad_s)};")
    lines.append(f"static const float CTRL_MAX_BASE_SPEED_M_S = {_fmt_num(cfg.max_base_speed_m_s)};")
    lines.append(f"static const float CTRL_WHEEL_DERATE_START_FRAC = {_fmt_num(cfg.wheel_torque_derate_start)};")
    lines.append(f"static const float CTRL_BASE_DERATE_START_FRAC = {_fmt_num(cfg.base_torque_derate_start)};")
    lines.append(f"static const float CTRL_BASE_FORCE_SOFT_LIMIT = {_fmt_num(cfg.base_force_soft_limit)};")
    lines.append(f"static const float CTRL_BASE_SPEED_SOFT_LIMIT_FRAC = {_fmt_num(cfg.base_speed_soft_limit_frac)};")
    lines.append(f"static const float CTRL_WHEEL_TORQUE_LIMIT_NM = {_fmt_num(cfg.wheel_torque_limit_nm)};")
    lines.append(
        f"static const uint32_t CTRL_CMD_TIMEOUT_US = {50000 if cfg.real_hardware_profile else 100000}u;"
    )
    lines.append(
        f"static const uint32_t CTRL_TILT_TRIP_COUNT = {12 if cfg.real_hardware_profile else 8}u;"
    )
    lines.append("")
    lines.append(f"static const float CTRL_WHEEL_MOTOR_KV_RPM_PER_V = {_fmt_num(cfg.wheel_motor_kv_rpm_per_v)};")
    lines.append(f"static const float CTRL_WHEEL_MOTOR_RESISTANCE_OHM = {_fmt_num(cfg.wheel_motor_resistance_ohm)};")
    lines.append(f"static const float CTRL_WHEEL_CURRENT_LIMIT_A = {_fmt_num(cfg.wheel_current_limit_a)};")
    lines.append(f"static const float CTRL_BUS_VOLTAGE_V = {_fmt_num(cfg.bus_voltage_v)};")
    lines.append(f"static const float CTRL_WHEEL_GEAR_RATIO = {_fmt_num(cfg.wheel_gear_ratio)};")
    lines.append(f"static const float CTRL_DRIVE_EFFICIENCY = {_fmt_num(cfg.drive_efficiency)};")
    lines.append("")
    lines.append("#endif /* CONTROLLER_PARAMS_H */")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = build_arg_parser().parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bundle = compute_export_bundle(args)
    text = render_header(bundle, args)
    out_path.write_text(text, encoding="utf-8")
    print(f"Wrote firmware header: {out_path}")


if __name__ == "__main__":
    main()
