# Final Folder Guide (For Non-Technical Readers)

good for modeling and deriving firmware parameters, but treat it as pre-deployment logic, not final safety-certified hardware control.

This folder is the "main demo" of the project.

It simulates a self-balancing robot in MuJoCo and shows how software can keep a robot upright using a reaction wheel and a moving base.

## 1) What This Is

In simple terms, this robot is like a broom trying to stand upright on your hand:
- the **stick** wants to fall,
- the **reaction wheel** inside can twist the body back,
- the **base** can slide to catch the fall,
- the controller repeats this many times per second.

Key files:
- Simulation + controller: `final/final.py`
- Mechanical model (geometry, joints, actuators): `final/final.xml`
- ESP32 firmware notes: `final/firmware/README.md`

## 2) Mechanical Design (Why the Geometry Helps It Balance)

### Main parts
- **Base body / wheel area**: lower body that contacts terrain.
- **Base X/Y slide joints**: lets the base shift under the robot.
- **Stick body**: upright mass that must be stabilized.
- **Reaction wheel**: spinning rotor that creates corrective torque.

### Why this geometry works
- The stick has mass above the base, so gravity creates a tipping moment.
- The reaction wheel can create fast opposite torque without moving the whole base first.
- If wheel speed gets too high, the base can move to reduce required wheel momentum.
- That gives two balancing tools:
  1. rotational correction (reaction wheel),
  2. translational correction (base motion).

### Geometry types in the XML
- **Dynamics-critical geometry**: affects mass/inertia/contact.
- **Visual-only geometry**: for appearance only (`mass="0"`, no collision contribution).

This separation is intentional: you can improve look/shape without silently breaking control physics.

## 3) Physics in Plain English (with Light Math)

These are the key ideas behind balancing:

- Torque drives angular acceleration:
  - `tau = I * alpha`
- Angular momentum of wheel:
  - `L = I_w * omega_w`
- Equal/opposite torque effect:
  - accelerating wheel one way pushes body the other way.

For small tilt angles, an inverted pendulum can be approximated as:
- `theta_ddot ≈ (g/l) * theta + (tau_control / I_body)`

Where:
- `theta` = tilt angle,
- `g` = gravity,
- `l` = center-of-mass height,
- `tau_control` = motor-generated correction torque.

Terrain friction matters too: if contact is poor, base motion helps less.

## 4) Control Law (How Software Actually Stabilizes It)

Every control update in `final/final.py` does:
1. Measure noisy sensor channels (IMU/encoder-like signals).
2. Estimate current state (Kalman correction).
3. Compute command increment using delta-u LQR.
4. Apply saturation/rate limits and safety logic.
5. Send commands to MuJoCo actuators.

Core control form:
- `du = -K * [x; u_prev]`
- `u = clip(u_prev + du)`

This means it does not jump directly to giant commands; it adjusts smoothly while respecting limits.

Safety layers include:
- wheel speed budget and high-spin recovery latch,
- base authority gating,
- actuator slew/torque limits,
- crash-angle stop logic.

## 5) Why This Is Close to Real Life

The simulation intentionally includes real-world effects:
- control loop rate separate from physics timestep,
- command delay queue,
- IMU/wheel/base sensor noise,
- motor envelope and saturation limits,
- crash thresholds and stop-on-crash behavior.

So the controller is not tuned on a perfect/noiseless world.

## 6) ESP32 Relevance

This project is structured so simulation logic maps to embedded firmware flow:
- fixed-rate estimator/controller loop,
- actuator guardrails,
- deterministic exported parameters.

Important files:
- `final/export_firmware_params.py` (exports controller params)
- `final/test_export_parity.py` (checks export/runtime consistency)
- `final/firmware/README.md` (integration notes for embedded side)

For ESP32 bring-up, the same safety mindset is used: bounded commands, watchdog-like checks, and conservative modes.

## 7) Run It on Your Laptop / PC

Requirements:
- Python 3.10+
- GPU/display environment for MuJoCo viewer

Install:

```bash
pip install -r requirements.txt
```

Run default smooth mode:

```bash
python final/final.py --mode smooth
```

Run robust profile:

```bash
python final/final.py --mode robust --stability-profile low-spin-robust
```

Run controller families:

```bash
python final/final.py --mode smooth --controller-family current
python final/final.py --mode smooth --controller-family hybrid_modern
python final/final.py --mode smooth --controller-family paper_split_baseline
```

Try push disturbance:

```bash
python final/final.py --mode smooth --push-x 4 --push-start-s 1.0 --push-duration-s 0.15
```

Try hardware-like constraints:

```bash
python final/final.py --mode robust --real-hardware
```

## 8) Why This Is Modern

- It uses a hybrid architecture: model-based core with explicit safety shaping.
- It includes a literature-style comparator (`paper_split_baseline`) for fair evaluation.
- It supports interpretable term logging (`--log-control-terms`) so each command can be traced to named terms.

## 9) Recent Update (February 2026)

- Added runtime controller family selector:
  - `--controller-family {current,hybrid_modern,paper_split_baseline}`
- Added explainability logging for control terms with optional CSV output:
  - `--log-control-terms`
  - `--control-terms-csv <path>`
- Added benchmark coverage for modern, literature-style, and legacy controller families.
- Added composite-score + hard-gate + significance-based promotion logic.
- Added benchmark profiles for operations:
  - `fast_pr` for quick deterministic checks
  - `nightly_long` for full stress validation

## 10) Benchmark Protocol

- Run with `final/benchmark.py`.
- Default benchmark compares multiple controller families and operating modes.
- Uses:
  - hard stability/saturation gates,
  - composite score,
  - paired bootstrap significance for promotion decisions.

Fast PR profile:

```bash
python final/benchmark.py --benchmark-profile fast_pr
```

Nightly long profile:

```bash
python final/benchmark.py --benchmark-profile nightly_long
```

Release campaign:

```bash
python final/benchmark.py --release-campaign --multiple-comparison-correction holm
```

Strict simulation hardware-readiness gate:

```bash
python final/benchmark.py --release-campaign --readiness-report --readiness-sign-window-steps 25 --readiness-replay-min-consistency 0.60 --readiness-replay-max-nrmse 0.75
```

Optional hardware replay:

```bash
python final/benchmark.py --benchmark-profile nightly_long --hardware-trace-path path/to/hardware_trace.csv
python final/final.py --mode smooth --trace-events-csv final/results/runtime_trace.csv
```

Readiness outputs:
- `final/results/readiness_<timestamp>.json`
- `final/results/readiness_<timestamp>.md`

## 11) Known Limits

- This is still a simulation, not a certified hardware safety system.
- Contact/friction/sensor models are approximations.
- Real hardware requires calibration, safety interlocks, and staged testing.

---

If you only read one file first, read `final/README.md` (this file), then open `final/final.py` and `final/final.xml` side-by-side.
