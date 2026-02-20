from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.linalg import solve_discrete_are

from runtime_config import RuntimeConfig


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
    payload_body_id: int
    payload_geom_id: int


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
        payload_body_id=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "payload"),
        payload_geom_id=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "payload_geom"),
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




def reset_state(model, data, q_pitch, q_roll, pitch_eq=0.0, roll_eq=0.0):
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    # Fix invalid quaternion in freejoint (indices 3:7)
    data.qpos[3] = 1.0  # w component of quaternion (identity rotation)
    data.qpos[4:7] = 0.0  # x, y, z components
    data.qpos[q_pitch] = pitch_eq
    data.qpos[q_roll] = roll_eq
    mujoco.mj_forward(model, data)


def set_payload_mass(model: mujoco.MjModel, data: mujoco.MjData, ids: ModelIds, payload_mass_kg: float) -> float:
    """Set runtime payload mass while keeping a valid positive-definite inertia tensor."""
    if ids.payload_body_id < 0 or ids.payload_geom_id < 0:
        return 0.0
    mass_target = float(max(payload_mass_kg, 0.0))
    mass_runtime = max(mass_target, 1e-6)
    sx, sy, sz = model.geom_size[ids.payload_geom_id, :3]
    ixx = (mass_runtime / 3.0) * (sy * sy + sz * sz)
    iyy = (mass_runtime / 3.0) * (sx * sx + sz * sz)
    izz = (mass_runtime / 3.0) * (sx * sx + sy * sy)
    model.body_mass[ids.payload_body_id] = mass_runtime
    model.body_inertia[ids.payload_body_id, :] = np.array([ixx, iyy, izz], dtype=float)
    mujoco.mj_setConst(model, data)
    mujoco.mj_forward(model, data)
    return mass_target


def compute_robot_com_distance_xy(model: mujoco.MjModel, data: mujoco.MjData, support_body_id: int) -> float:
    """Planar COM distance from support-body origin."""
    masses = model.body_mass[1:]
    total_mass = float(np.sum(masses))
    if total_mass <= 1e-12:
        return 0.0
    com_xy = (masses[:, None] * data.xipos[1:, :2]).sum(axis=0) / total_mass
    support_xy = data.xpos[support_body_id, :2]
    return float(np.linalg.norm(com_xy - support_xy))


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



