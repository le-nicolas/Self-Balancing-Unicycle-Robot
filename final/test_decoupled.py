"""Quick test of decoupled control."""
from pathlib import Path
from unconstrained_runtime import UnconstrainedWheelBotRuntime

xml_path = Path(__file__).with_name("final.xml")
runtime = UnconstrainedWheelBotRuntime(
    xml_path=xml_path,
    payload_mass_kg=0.4,
    mode="balanced",
    seed=0,
)

print("Testing decoupled control...")
print("Running for 5 seconds...\n")

success = True
for i in range(2500):  # 5 seconds at dt=0.002
    runtime.step()
    
    if runtime.failed:
        s = runtime.get_state()
        print(f"FAILED at t={s['elapsed_s']:.3f}s")
        print(f"  Roll: {s['roll_rad']*180/3.14159:.1f}°, Pitch: {s['pitch_rad']*180/3.14159:.1f}°")
        print(f"  u = {s['u_rw']:.1f}, {s['u_dx']:.1f}, {s['u_dy']:.1f}")
        success = False
        break
    
    if (i + 1) % 500 == 0:
        s = runtime.get_state()
        print(f"t={s['elapsed_s']:.2f}s: roll={s['roll_rad']*180/3.14159:6.2f}° pitch={s['pitch_rad']*180/3.14159:6.2f}°")

if success:
    s = runtime.get_state()
    print(f"\nSUCCESS! Stabilized for {s['elapsed_s']:.2f}s")
    print(f"Final: roll={s['roll_rad']*180/3.14159:6.2f}°, pitch={s['pitch_rad']*180/3.14159:6.2f}°")
