"""Debug script to verify gain matrices and state extraction."""
from pathlib import Path
import numpy as np
from unconstrained_runtime import UnconstrainedWheelBotRuntime

xml_path = Path(__file__).with_name("final.xml")
runtime = UnconstrainedWheelBotRuntime(
    xml_path=xml_path,
    payload_mass_kg=0.4,
    mode="balanced",
    seed=0,
)

print("=== Control Gain Shapes ===")
print(f"k_roll shape: {runtime.k_roll.shape}")
print(f"k_pitch shape: {runtime.k_pitch.shape}")

print("\n=== State Vector ===")
x = runtime._state_vector()
print(f"Full state shape: {x.shape}")
print(f"Full state: {x}")
print(f"  [0] roll={x[0]:.6f}, [1] pitch={x[1]:.6f}")
print(f"  [2] roll_rate={x[2]:.6f}, [3] pitch_rate={x[3]:.6f}")
print(f"  [4] wheel_spin={x[4]:.6f}")
print(f"  [5] wheel_x_angle={x[5]:.6f}, [6] wheel_y_angle={x[6]:.6f}")
print(f"  [7] wheel_x_rate={x[7]:.6f}, [8] wheel_y_rate={x[8]:.6f}")

print("\n=== Roll Control ===")
x_roll = np.array([x[0], x[2], x[4]], dtype=float)
print(f"x_roll: {x_roll}")
u_roll = -(runtime.k_roll @ x_roll)
print(f"u_roll (raw): {u_roll}, shape: {u_roll.shape}")
u_roll_clipped = np.clip(u_roll, -runtime.cfg.max_u[0], runtime.cfg.max_u[0])
print(f"u_roll (clipped): {u_roll_clipped}")

print("\n=== Pitch Control ===")
x_pitch = np.array([x[1], x[3], x[5], x[6], x[7], x[8]], dtype=float)
print(f"x_pitch: {x_pitch}")
u_pitch = -(runtime.k_pitch @ x_pitch)
print(f"u_pitch (raw): {u_pitch}, shape: {u_pitch.shape}")
u_pitch_clipped = np.clip(u_pitch, -runtime.cfg.max_u[1:], runtime.cfg.max_u[1:])
print(f"u_pitch (clipped): {u_pitch_clipped}")

print("\n=== Combined Control ===")
u = np.array([u_roll_clipped[0], u_pitch_clipped[0], u_pitch_clipped[1]], dtype=float)
u[1] = 0.0  # Disable X drive
print(f"Final u: {u}")

print("\n=== Gain Matrix Preview ===")
print("k_roll:\n", runtime.k_roll)
print("\nk_pitch:\n", runtime.k_pitch)

print("\n=== Config ===")
print(f"max_u: {runtime.cfg.max_u}")
print(f"Q diag shape (for reference): rows from original unified Q")
print(f"mode: {runtime.cfg.mode}")
