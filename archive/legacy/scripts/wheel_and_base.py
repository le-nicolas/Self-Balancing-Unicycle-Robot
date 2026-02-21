from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np


def wrap_to_pi(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def rate_limit(target: float, previous: float, max_rate: float, dt: float) -> float:
    max_step = max_rate * dt
    return previous + float(np.clip(target - previous, -max_step, max_step))


# ----------------------------
# Load model
# ----------------------------
model = mujoco.MjModel.from_xml_path(str(Path(__file__).resolve().parents[1] / "models" / "reactionwheel_basemotor.xml"))
data = mujoco.MjData(model)
dt = model.opt.timestep

# ----------------------------
# Resolve IDs once (robust to XML reorder)
# ----------------------------
j_stick = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "stick_hinge")
j_wheel = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "wheel_spin")
a_wheel = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wheel_motor")
a_base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "base_motor")

if min(j_stick, j_wheel, a_wheel, a_base) < 0:
    raise RuntimeError("Expected joints/actuators were not found in reactionwheel_basemotor.xml.")

qpos_stick = model.jnt_qposadr[j_stick]
dof_stick = model.jnt_dofadr[j_stick]
dof_wheel = model.jnt_dofadr[j_wheel]

# ----------------------------
# Controller gains
# ----------------------------
# Wheel (fast inner stabilizer)
K_RW_THETA = 620.0
K_RW_THETA_DOT = 95.0
K_RW_SPEED = 16.0

# Base motor (slower assist + wheel momentum dumping)
K_BASE_THETA = 90.0
K_BASE_THETA_DOT = 24.0
K_BASE_DUMP = 10.0

# Wheel speed handling
SOFT_WHEEL_SPEED = 120.0
HARD_WHEEL_SPEED = 220.0

# Torque slew-rate limits (more realistic motor response)
MAX_DTAU_WHEEL = 16000.0  # N*m/s
MAX_DTAU_BASE = 3000.0    # N*m/s

# ----------------------------
# Actuator limits (from XML)
# ----------------------------
MAX_TAU_WHEEL = float(np.max(np.abs(model.actuator_ctrlrange[a_wheel])))
MAX_TAU_BASE = float(np.max(np.abs(model.actuator_ctrlrange[a_base])))

# ----------------------------
# Initial conditions
# ----------------------------
data.qpos[qpos_stick] = np.deg2rad(8.0)
data.qvel[dof_stick] = 0.0
data.qvel[dof_wheel] = 0.0
mujoco.mj_forward(model, data)

tau_wheel_prev = 0.0
tau_base_prev = 0.0
last_print_time = -1.0

# ----------------------------
# Simulation loop
# ----------------------------
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        theta = wrap_to_pi(data.qpos[qpos_stick])
        theta_dot = data.qvel[dof_stick]
        wheel_speed = data.qvel[dof_wheel]

        # Feedforward torque needed to hold current stick state against model bias forces.
        tau_bias_stick = data.qfrc_bias[dof_stick]

        # Fast wheel stabilization torque.
        tau_wheel_cmd = -(
            K_RW_THETA * theta
            + K_RW_THETA_DOT * theta_dot
            + K_RW_SPEED * wheel_speed
        )

        # Soft-derate wheel torque as speed approaches an unsafe region.
        speed_abs = abs(wheel_speed)
        if speed_abs > SOFT_WHEEL_SPEED:
            derate_ratio = np.clip(
                (speed_abs - SOFT_WHEEL_SPEED) / (HARD_WHEEL_SPEED - SOFT_WHEEL_SPEED),
                0.0,
                1.0,
            )
            tau_wheel_cmd *= 1.0 - 0.8 * derate_ratio

        # Base motor: bias compensation + posture assist + momentum dumping.
        tau_base_cmd = (
            tau_bias_stick
            - K_BASE_THETA * theta
            - K_BASE_THETA_DOT * theta_dot
            + K_BASE_DUMP * wheel_speed
        )

        # Saturate commands to actuator limits.
        tau_wheel_cmd = float(np.clip(tau_wheel_cmd, -MAX_TAU_WHEEL, MAX_TAU_WHEEL))
        tau_base_cmd = float(np.clip(tau_base_cmd, -MAX_TAU_BASE, MAX_TAU_BASE))

        # Rate-limit torque changes (motor realism and less chatter).
        tau_wheel = rate_limit(tau_wheel_cmd, tau_wheel_prev, MAX_DTAU_WHEEL, dt)
        tau_base = rate_limit(tau_base_cmd, tau_base_prev, MAX_DTAU_BASE, dt)

        data.ctrl[a_wheel] = tau_wheel
        data.ctrl[a_base] = tau_base

        tau_wheel_prev = tau_wheel
        tau_base_prev = tau_base

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(dt)

        if data.time - last_print_time >= 1.0:
            last_print_time = data.time
            print(
                f"t={data.time:6.2f}s  theta={np.degrees(theta):7.2f} deg  "
                f"theta_dot={theta_dot:8.3f}  wheel={wheel_speed:8.2f} rad/s  "
                f"tau_w={tau_wheel:8.2f}  tau_b={tau_base:7.2f}"
            )

