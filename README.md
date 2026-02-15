# Reaction Wheel Balancer (Non-Technical Guide)

good for modeling and deriving firmware parameters, but treat it as pre-deployment logic, not final safety-certified hardware control.



https://github.com/user-attachments/assets/586ee1d3-beb5-460e-8ca5-f6c0afe3e5ce

<img width="836" height="696" alt="Screenshot 2026-02-15 134403" src="https://github.com/user-attachments/assets/bfab47e0-a868-42be-bca1-90a2e9d7474f" />



This project simulates a self-balancing robot in MuJoCo.

In simple terms:
- the robot body wants to fall,
- a spinning wheel inside creates balancing torque,
- the base can move under the body to help catch the fall,
- software repeats this correction loop many times per second.

## What This Is

This is a digital twin of a balancing robot with:
- a base body / wheel area,
- a stick-like upper body,
- a reaction wheel,
- and a controller that tries to keep it upright.

You can run it, push it, switch modes, and inspect how close the behavior is to real life constraints.

## Mechanical Design (Why the Geometry Can Balance)

### Main parts
- **Base body / wheel area**: lower structure that contacts terrain.
- **Base X/Y slide joints**: allows base translation to assist stability.
- **Stick body**: upright mass that needs stabilization.
- **Reaction wheel**: spinning rotor used to generate corrective torque.

### Why this geometry works
- The stick has mass above the base, so gravity creates a tipping tendency.
- The reaction wheel can rapidly apply opposite torque to fight tilt.
- The sliding base can shift under the robot when wheel momentum gets high.
- So balancing is done with two tools:
  1. rotational correction (reaction wheel),
  2. translational correction (base movement).

### Visual-only vs physics-critical geometry
In `final/final.xml`:
- **Physics-critical geometry** affects mass/inertia/contact.
- **Visual-only geometry** has `mass="0"` and collision disabled (`contype="0"`, `conaffinity="0"`).

This separation keeps visuals flexible without accidentally changing control dynamics.

## Physics in Simple Terms (Light Math)

Key equations used by this kind of balancing system:

- Torque/acceleration relationship:
  - `tau = I * alpha`
- Wheel angular momentum:
  - `L = I_w * omega_w`
- Equal/opposite torque effect:
  - accelerating the wheel one direction pushes the body the opposite direction.

Small-angle intuition for inverted pendulum behavior:
- `theta_ddot ≈ (g/l) * theta + (tau_control / I_body)`

Where:
- `theta` is tilt angle,
- `g` is gravity,
- `l` is center-of-mass height,
- `tau_control` is commanded corrective torque.

Contact and friction from terrain also matter because base translation depends on grip.

## Control Law (What the Software Does)

Each control update cycle in `final/final.py` does:
1. Read noisy sensor-like measurements.
2. Estimate state using a Kalman-style correction.
3. Compute control increment with delta-u LQR.
4. Apply safety shaping and command limits.
5. Send actuator commands and step simulation.

Core control form:
- `du = -K * [x; u_prev]`
- `u = clip(u_prev + du)`

Meaning:
- the controller adjusts commands incrementally,
- commands are clipped to safe bounds,
- wheel/base actuation are coordinated.

Safety logic includes:
- wheel speed budget and high-spin latch behavior,
- base authority gating,
- actuator saturation/rate limiting,
- crash-angle handling.

## Why It Is Close to Real Life

The simulator includes practical constraints, not just ideal physics:
- control loop frequency (`control_hz`) separate from physics timestep,
- command delay queue (`control_delay_steps`),
- sensor noise (IMU + wheel encoder + base estimate noise),
- actuator saturation and motor envelope limits,
- crash thresholds and stop-on-crash behavior.

So tuning is done under imperfect sensing/actuation assumptions.

## ESP32 Relevance

The project is structured to map cleanly to embedded control flow on ESP32-like systems:
- fixed-rate estimator + controller outer loop,
- deterministic guardrails and fault handling,
- parameter export parity between simulation and firmware.

Important files:
- `final/firmware/README.md`
- `final/export_firmware_params.py`
- `final/test_export_parity.py`

This makes it easier to move from sim behavior to firmware implementation with fewer surprises.

## Run It on Your Laptop / PC

Requirements:
- Python 3.10+
- Graphics/display environment for MuJoCo viewer

Install dependencies:

```bash
pip install -r requirements.txt
```

Run default mode:

```bash
python final/final.py --mode smooth
```

Run robust mode:

```bash
python final/final.py --mode robust --stability-profile low-spin-robust
```

Run controller families:

```bash
python final/final.py --mode smooth --controller-family current
python final/final.py --mode smooth --controller-family hybrid_modern
python final/final.py --mode smooth --controller-family paper_split_baseline
```

Play with scripted push disturbance:

```bash
python final/final.py --mode smooth --push-x 4 --push-start-s 1.0 --push-duration-s 0.15
```

Try hardware-like constraints:

```bash
python final/final.py --mode robust --real-hardware
```

Useful checks:

```bash
python final/test_export_parity.py
python final/export_firmware_params.py --mode smooth
```

## Modernization Notes

- The runtime now supports controller families:
  - `current` (backward-compatible baseline),
  - `hybrid_modern` (preferred modernized controller),
  - `paper_split_baseline` (literature-style comparator).
- `hybrid_modern` is not a paper copy. It keeps the model-based core and adds explicit interpretable terms.
- Optional explainability logging:
  - `--log-control-terms`
  - `--control-terms-csv <path>`

## Recent Update (February 2026)

- Added controller family selection in runtime:
  - `--controller-family {current,hybrid_modern,paper_split_baseline}`
- Added interpretable control-term decomposition and optional CSV logging:
  - `term_lqr_core`, `term_roll_stability`, `term_pitch_stability`, `term_despin`, `term_base_hold`, `term_safety_shaping`
  - `--log-control-terms`, `--control-terms-csv`
- Added literature-style comparator path (`paper_split_baseline`) for fair benchmark comparison.
- Extended evaluator/benchmark pipeline with controller-family support, composite score reporting, and promotion decision fields.
- Added significance-aware promotion logic (paired bootstrap) for `hybrid_modern` vs `current`.
- Added CI-friendly benchmark profiles and script entry points:
  - `python final/benchmark.py --benchmark-profile fast_pr`
  - `python final/benchmark.py --benchmark-profile nightly_long`
  - `python final/benchmark_fast_pr.py`
  - `python final/benchmark_nightly_long.py`

## Benchmark Protocol

- Main benchmark script: `final/benchmark.py`
- Families are benchmarked side-by-side via `--controller-families`.
- Composite score + hard gates are computed for each candidate.
- Promotion is decided for `hybrid_modern` vs `current` baseline with:
  - hard safety gates,
  - paired bootstrap significance test (`--significance-alpha`).

Fast PR benchmark:

```bash
python final/benchmark.py --benchmark-profile fast_pr
```

Nightly long benchmark:

```bash
python final/benchmark.py --benchmark-profile nightly_long
```

Release campaign (protocol-locked, cross-variant/domain, corrected significance):

```bash
python final/benchmark.py --release-campaign --multiple-comparison-correction holm --model-variants nominal,inertia_plus,friction_low,com_shift --domain-rand-profile default,rand_light,rand_medium
```

Strict simulation hardware-readiness gate (JSON + Markdown):

```bash
python final/benchmark.py --release-campaign --readiness-report --readiness-sign-window-steps 25 --readiness-replay-min-consistency 0.60 --readiness-replay-max-nrmse 0.75
```

Hardware replay alignment (optional):

```bash
python final/benchmark.py --benchmark-profile nightly_long --hardware-trace-path path/to/hardware_trace.csv
python final/final.py --mode smooth --trace-events-csv final/results/runtime_trace.csv
```

Readiness artifacts are emitted under `final/results/` as:
- `readiness_<timestamp>.json`
- `readiness_<timestamp>.md`

## Known Limitations

- This is still a simulation, not a certified safety controller.
- Contact, friction, and sensor models are approximations.
- Real hardware still needs calibration, staged testing, and safety interlocks.

## Project Files (Quick Map)

- Main runtime/controller: `final/final.py`
- Mechanical/scene model: `final/final.xml`
- Firmware notes: `final/firmware/README.md`
- Non-technical final-folder copy: `final/README.md`
