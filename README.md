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

## Scope and Claim Boundaries

- This repository claims simulation results only, under the exact benchmark settings and artifacts in `final/results/`.
- It does not claim certified hardware safety, formal stability guarantees, or universal superiority over prior physical robots.
- Comparisons to literature are for context and reproducibility, not direct apples-to-apples hardware claims.

## Implemented Architecture (Current)

- State estimation: linear Kalman correction from noisy sensor-like channels.
- Control core: delta-u LQR with mode-dependent shaping terms.
- Runtime safety shaping: torque/rate limits, wheel-speed budget/high-spin latch, base-authority gating, crash-angle logic.

## Not Implemented (Do Not Infer)

- Disturbance-observer-based control law (DOB) with explicit disturbance-state compensation.
- Control-Lyapunov-function QP controller (CLF-QP), including closed-form CLF-QP updates.
- Formal Lyapunov proof of stability for the full runtime controller.

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

## 2013 Paper Context and Statistical Visual Comparison

Reference paper (hardware implementation):
- J. Lee, S. Han, J. Lee, "Decoupled Dynamic Control for Pitch and Roll Axes of the Unicycle Robot," IEEE Transactions on Industrial Electronics, Vol. 60, No. 9, September 2013.
- DOI: `10.1109/TIE.2012.2208431`
- Local review source: `C:/Users/User/Downloads/onebot.pdf`

Comparison intent:
- Reproduce balancing objectives in simulation and extend evaluation rigor under explicit noise, delay, disturbances, comparator families, and repeatable artifacts.
- Preserve nuance: this is not a strict apples-to-apples hardware replacement claim.

Statistical visual summary (paper context vs this simulation benchmark):

![Paper vs simulation statistical comparison](final/results/paper_vs_sim_comparison.png)

Figure source and generator:
- CSV: `final/results/benchmark_20260216_011044.csv`
- Script: `final/plot_paper_vs_sim.py`
- Re-generate:
```bash
python final/plot_paper_vs_sim.py --csv final/results/benchmark_20260216_011044.csv --out final/results/paper_vs_sim_comparison.png
```

## Reproducibility

Benchmark command used for the snapshot:

```bash
python final/benchmark.py --benchmark-profile fast_pr --episodes 8 --steps 3000 --trials 0 --controller-families current,hybrid_modern,paper_split_baseline,baseline_mpc,baseline_robust_hinf_like --model-variants nominal --domain-rand-profile default --compare-modes default-vs-low-spin-robust --primary-objective balanced
```

Expected outputs:
- `final/results/benchmark_<timestamp>_summary.txt`
- `final/results/benchmark_<timestamp>.csv`
- `final/results/benchmark_<timestamp>_protocol.json`

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

## Realism Assumptions in Simulation

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

## Runtime Controller Modes

- The runtime now supports controller families:
  - `current` (backward-compatible baseline),
  - `hybrid_modern` (alternative tuned controller),
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
