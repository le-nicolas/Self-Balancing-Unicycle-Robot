# Final Folder Guide (For Non-Technical Readers)

good for modeling and deriving firmware parameters, but treat it as pre-deployment logic, not final safety-certified hardware control.

This folder is the "main demo" of the project.

It simulates a self-balancing robot in MuJoCo and shows how software can keep a robot upright using a reaction wheel and a moving base.

## Scope and Claim Boundaries

- This folder reports simulation results only, under the exact benchmark settings and artifacts in `final/results/`.
- It does not claim certified hardware safety, formal stability guarantees, or universal superiority over prior physical robots.
- Literature comparisons are for context and reproducibility, not direct apples-to-apples hardware claims.

## Implemented Architecture (Current)

- State estimation: linear Kalman correction from noisy IMU/encoder-like channels.
- Control core: delta-u LQR with runtime safety shaping and mode-specific terms.
- Optional residual correction: offline-trained PyTorch model adds bounded `delta_u` on top of nominal control (`u = u_nominal + delta_u`), with runtime tilt/rate gating.
- Safety policies: wheel-speed budget/high-spin latch, strict hard-speed same-direction suppression, base-authority gating, actuator clipping/rate limits, crash-angle stop logic.

## Not Implemented (Do Not Infer)

- Disturbance-observer-based control law (DOB) with explicit online disturbance compensation.
- Control-Lyapunov-function QP controller (CLF-QP), including closed-form CLF-QP update rules.
- Formal Lyapunov proof for the full runtime controller.

## Metrics Snapshot (2026-02-16)

Canonical source:
- `final/results/benchmark_20260216_011044_summary.txt`
- `final/results/benchmark_20260216_011044.csv`
- `final/results/benchmark_20260216_011044_protocol.json`

| Controller (mode_default / nominal / default) | Survival | Crash rate | Composite score |
|---|---:|---:|---:|
| `baseline_robust_hinf_like` | 1.000 | 0.000 | 75.457 |
| `hybrid_modern` | 1.000 | 0.000 | 74.460 |
| `current` | 1.000 | 0.000 | 73.993 |
| `paper_split_baseline` | 1.000 | 0.000 | 72.913 |
| `baseline_mpc` | 0.625 | 0.375 | 37.837 |

## Reproducibility

Benchmark command used for this snapshot:

```bash
python final/benchmark.py --benchmark-profile fast_pr --episodes 8 --steps 3000 --trials 0 --controller-families current,hybrid_modern,paper_split_baseline,baseline_mpc,baseline_robust_hinf_like --model-variants nominal --domain-rand-profile default --compare-modes default-vs-low-spin-robust --primary-objective balanced
```

Expected outputs:
- `final/results/benchmark_<timestamp>_summary.txt`
- `final/results/benchmark_<timestamp>.csv`
- `final/results/benchmark_<timestamp>_protocol.json`

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
4. Optionally add residual correction from an offline PyTorch model.
5. Apply saturation/rate limits and safety logic.
6. Send commands to MuJoCo actuators.

Core control form:
- `du = -K * [x; u_prev]`
- `u = clip(u_prev + du)`

This means it does not jump directly to giant commands; it adjusts smoothly while respecting limits.

Residual correction behavior (optional):
- It does not replace the controller.
- It is offline-trained only (no online learning in runtime loop).
- It is clipped and gated by tilt/rate thresholds.
- If `delta_u = 0`, behavior falls back to the original controller path.

Safety layers include:
- wheel speed budget and high-spin recovery latch,
- strict hard-zone protection that suppresses same-direction wheel torque near hard speed,
- base authority gating,
- actuator slew/torque limits,
- crash-angle stop logic.

## 5) Realism Assumptions in Simulation

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

Optional residual dependency:

```bash
pip install torch
```

Run default smooth mode:

```bash
python final/final.py --mode smooth
```

Run with top payload mass:

```bash
python final/final.py --mode smooth --payload-mass 0.8
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

Run with optional residual correction:

```bash
python final/final.py --mode smooth --residual-model path/to/residual.pt --residual-scale 0.20
```

## 8) Implemented Runtime Features

- It uses a hybrid architecture: model-based core with explicit safety shaping.
- It includes a literature-style comparator (`paper_split_baseline`) for fair evaluation.
- It supports interpretable term logging (`--log-control-terms`) so each command can be traced to named terms.
- It now logs wheel-spin telemetry for tuning: signed peak speed, mean absolute speed, over-budget/over-hard ratios (total and sign-split), and hard-zone suppression counters.

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

## 11) Historical Paper Comparison (Nuanced)

### Intention of this comparison

- The intent is to show **simulation-only advancement** through reproducibility, stress testing, and transparent benchmark artifacts.
- The intent is **not** to dismiss physical robot work; fabrication tolerances, sensor drift, and integration constraints are real and matter.
- Therefore, we compare with nuance: stronger simulation evaluation does not automatically mean universally better real-world control.

### Reference paper (physical robot, about a decade earlier)

- Source paper:
  - J. Lee, S. Han, J. Lee, "Decoupled Dynamic Control for Pitch and Roll Axes of the Unicycle Robot," IEEE TIE, Vol. 60, No. 9, September 2013, DOI: `10.1109/TIE.2012.2208431`.
- Local file used for review:
  - `C:/Users/User/Downloads/onebot.pdf`
- Extracted text artifact used in this repo:
  - `final/results/onebot_extracted.txt`

Paper-reported findings (hardware experiments):
- Decoupled control architecture:
  - roll: fuzzy-sliding mode
  - pitch: LQR
- Trajectory tests:
  - ramp input over `3.3 m`
  - ladder input over `1.1 m` forward/back
  - parabolic input up to `2.8 m`
- Reported angle error ranges:
  - ramp: roll about `+/-1 deg`, pitch about `+/-2 deg`
  - ladder: roll about `+/-2 deg`, pitch about `+/-4 deg`
  - parabolic: mostly around `+/-1 deg` (roll), `+/-2 deg` (pitch), with temporary pitch increase above `4 deg` around `30 s`
- Reported limitation: velocity remained below `0.1 m/s` due sensor speed limits.

### This repo findings (simulation campaign)

As of the benchmark run on `2026-02-16`:
- Command:
  - `python final/benchmark.py --benchmark-profile fast_pr --episodes 8 --steps 3000 --trials 0 --controller-families current,hybrid_modern,paper_split_baseline,baseline_mpc,baseline_robust_hinf_like --model-variants nominal --domain-rand-profile default --compare-modes default-vs-low-spin-robust --primary-objective balanced`
- Artifacts:
  - `final/results/benchmark_20260216_011044_summary.txt`
  - `final/results/benchmark_20260216_011044.csv`
  - `final/results/benchmark_20260216_011044_protocol.json`

Key simulated baseline results (nominal/default):
- `hybrid_modern`: survival `1.000`, crash rate `0.000`, composite score `74.460`
- `current`: survival `1.000`, crash rate `0.000`, composite score `73.993`
- `baseline_robust_hinf_like`: survival `1.000`, crash rate `0.000`, composite score `75.457`
- `paper_split_baseline`: survival `1.000`, crash rate `0.000`, composite score `72.913`
- `baseline_mpc`: survival `0.625`, crash rate `0.375`

Simulation protocol highlights:
- control timing and delay (`250 Hz`, `1` step delay)
- injected sensor noise
- periodic disturbances
- multi-family comparator framework and reproducible artifacts

### Statistical visual graph (paper context vs simulation)

![Paper vs simulation statistical comparison](results/paper_vs_sim_comparison.png)

Graph inputs:
- CSV: `final/results/benchmark_20260216_011044.csv`
- Plot script: `final/plot_paper_vs_sim.py`
- Re-generate:
```bash
python final/plot_paper_vs_sim.py --csv final/results/benchmark_20260216_011044.csv --out final/results/paper_vs_sim_comparison.png
```

### Nuanced claim boundary

Reasonable claim:
- This project advances **simulation evaluation rigor** (repeatability, explicit stress conditions, comparator breadth, and artifact traceability).
- This project demonstrates stable balancing performance in simulation under configured noise, delay, and disturbance settings.

Not yet a justified claim:
- "Universally better control performance than the 2013 hardware robot" (not directly apples-to-apples without matched tasks/constraints).

To make the comparison tighter in future work:
- add a dedicated simulation benchmark that exactly matches the paper's trajectory tasks (`3.3 m` ramp, `1.1 m` ladder, `2.8 m` parabolic),
- include matched speed limits and comparable sensor assumptions,
- report the same tracking/error metrics side-by-side.

## 12) Known Limits

- This is still a simulation, not a certified hardware safety system.
- Contact/friction/sensor models are approximations.
- Real hardware requires calibration, safety interlocks, and staged testing.

---

If you only read one file first, read `final/README.md` (this file), then open `final/final.py` and `final/final.xml` side-by-side.
