# Final Balancer: Build Story, Problems, and Reproducible Guide

This folder is the current MuJoCo implementation of our wheel-on-stick balancer.
The goal of this README is to explain:

1. What `final.py` and `final.xml` do.
2. How we evolved the design.
3. What failed along the way and how we fixed it.
4. How to reproduce the work and rebuild the system from scratch.

## 1) What Is In This Folder

| File | Purpose |
|---|---|
| `final/final.xml` | Mechanical and physics model (bodies, joints, inertia, contacts, actuators). |
| `final/final.py` | Runtime loop: parse config, linearize model, estimate state, compute control, apply safety, simulate/render. |
| `final/control_core.py` | Main control law and safety shaping (LQR path and MPC path). |
| `final/runtime_config.py` | All runtime parameters and CLI flags. |
| `final/config.yaml` | Central tuning file for wheel/base saturation policy and other runtime overrides. |
| `final/adaptive_id.py` | Online RLS system-ID and adaptive LQR gain scheduler. |
| `final/runtime_model.py` | Model IDs, reset logic, estimator, COM distance checks, root-attitude clamp helpers. |
| `final/mpc_controller.py` | Constrained MPC solver (OSQP/scipy fallback). |
| `final/controller_eval.py` | Headless evaluator used by benchmarking/tuning scripts. |
| `final/benchmark.py` | Reproducible stress benchmark and artifact writer. |
| `final/analyze_crash_coupling.py` | Replays one episode and prints pre-crash disturbance-event wheel/pitch trends for coupling diagnosis. |
| `final/find_lqr_envelope.py` | Binary-searches disturbance XY magnitude where per-cycle pitch ratchet crosses a target threshold. |
| `final/sim2real_sensitivity.py` | Ranks model/domain sensitivity from benchmark CSV baselines. |
| `final/build_residual_dataset.py` | Builds supervised residual dataset from paired baseline/teacher traces. |
| `final/train_residual_model.py` | Trains residual MLP checkpoint compatible with runtime loader. |

## 2) How `final.xml` and `final.py` Fit Together

`final.xml` is the source of truth for robot physics.
`final.py` reads it and builds a controller around that exact model.

Control state used in runtime:

`x = [pitch, roll, pitch_rate, roll_rate, wheel_rate, base_x, base_y, base_vx, base_vy]`

Control command:

`u = [wheel_spin, base_x_force, base_y_force]`

Runtime flow in `final/final.py`:

1. Parse CLI and build `RuntimeConfig`.
2. Load `final.xml` and resolve IDs by name.
3. Linearize dynamics around upright (`mjd_transitionFD`).
4. Build controller gains (LQR delta-u, optional MPC setup).
5. Build estimator (Kalman correction on noisy channels).
6. Run loop: estimate -> control -> clamp/derate/safety -> step MuJoCo -> detect crash.
7. Log metrics/artifacts.

## 2.1) Architecture Diagram

```text
                    +-----------------------------+
                    |      runtime_config.py      |
                    |  CLI flags -> RuntimeConfig |
                    +-------------+---------------+
                                  |
                                  v
                    +-----------------------------+
                    |          final.py           |
                    | orchestrates runtime loop   |
                    +------+------+---------------+
                           |      |
             load model    |      | control call
                           |      v
                           |   +---------------------------+
                           |   |      control_core.py      |
                           |   | LQR/MPC + safety shaping  |
                           |   +------+--------------------+
                           |          |
                           |          | (if --use-mpc)
                           |          v
                           |   +---------------------------+
                           |   |     mpc_controller.py     |
                           |   | constrained QP optimizer  |
                           |   +---------------------------+
                           |
                           v
              +-----------------------------+
              |         final.xml           |
              | plant: joints/acts/contact  |
              +--------------+--------------+
                             |
                             v
              +-----------------------------+
              |      runtime_model.py       |
              | IDs, reset, estimator, COM  |
              +--------------+--------------+
                             |
                             v
              +-----------------------------+
              | MuJoCo step + state update  |
              +--------------+--------------+
                             |
                             v
              +-----------------------------+
              | logs/metrics/results output  |
              +-----------------------------+

Benchmark path (headless):
benchmark.py -> controller_eval.py -> same plant/control stack -> final/results/*
```

## 3) How We Got To The Current Design

This is the practical progression we followed.

1. Mechanical model first.
Implemented the free-body base, 2-DOF stick (`stick_pitch`, `stick_roll`), reaction wheel joint (`wheel_spin`), terrain, and motor actuators in `final/final.xml`.

2. Baseline controller around linearized dynamics.
Added finite-difference linearization + delta-u LQR in `final/final.py` and `final/control_core.py`.

3. Estimation and hardware-like runtime behavior.
Added IMU/encoder noise, control delay, quantization, and Kalman update in `final/runtime_model.py`.

4. Runtime safety layers.
Added saturation handling, wheel momentum budget/hard-zone behavior, command-rate limits, and crash detection in `final/control_core.py` and `final/final.py`.

5. Reproducible evaluation.
Added `final/controller_eval.py` and `final/benchmark.py` for deterministic multi-episode testing and artifact generation in `final/results/`.

6. MPC path with hard constraints.
Added `final/mpc_controller.py` and integration into `final/final.py`/`final/control_core.py` for constrained control (input/angle/COM bounds).

7. Drift and tilt hardening.
Added root attitude clamping, robust DARE fallbacks, terminal cost/rate-target shaping, pitch/roll anti-drift integrators, and pitch rescue logic.

## 3.1) Why We Added Realistic IMU/Sensor Modeling

The previous estimator path was intentionally idealized for fast controller iteration.
That was useful early, but it can hide problems that appear immediately on hardware.

We added four realism layers in `runtime_model.py` and `final.xml`:

1. Bias drift (random walk) on IMU/encoder channels.
2. Explicit sensor sample/hold and configurable measurement latency.
3. Per-channel saturation and low-pass filtering.
4. MuJoCo named sensor path (`sensordata`) with direct-state fallback.

Why this matters:

1. It exposes slow drift modes earlier.
Without bias and latency, a controller can look perfect in sim but fail from gradual tilt/wheel drift in longer runs.
2. It makes estimator confidence more realistic.
The Kalman update now sees noisier, delayed, and clipped measurements, which is closer to real embedded behavior.
3. It improves sim-to-real relevance.
You tune against the failure modes you will actually see on hardware: phase lag, bias accumulation, and sensor ceilings.
4. It makes benchmark claims stronger.
Stability under non-ideal sensing is a more meaningful robustness result than stability under near-perfect state access.

General behavior impact you should expect:

1. Slightly slower and less sharp recovery.
2. More long-horizon drift pressure if gains were tuned for ideal sensing.
3. Better realism, better transfer value, and clearer tuning tradeoffs.

## 4) Problems We Hit and How We Solved Them

| Problem observed | Root cause | Fix applied | Where |
|---|---|---|---|
| `solve_discrete_are` failures (`ill-conditioned`, `no finite solution`) | Numerical sensitivity in some linearized/augmented cases | Added robust DARE wrapper with regularization sweep and robust linear solve fallback | `final/final.py`, `final/controller_eval.py` |
| Robot kept falling sideways | Roll control channel was effectively wrong for this architecture | Remapped `base_y_force` to `stick_roll` to restore direct roll authority | `final/final.xml` |
| Stick looked like it tilted while base orientation drifted unrealistically | Uncontrolled free-joint attitude mode | Added `lock_root_attitude` path to clamp free-joint quaternion/angular rates upright | `final/runtime_model.py`, `final/final.py`, `final/controller_eval.py`, `final/runtime_config.py` |
| Wheel looked visually detached from ground | Visual-only geometry dimensions did not match physical contact cylinder | Made physical contact geometry explicit (`drive_wheel_phys`) and aligned visual sidewall radius | `final/final.xml` |
| Occasional self-collision artifacts | Base and stick could collide at nominal assembly | Added contact exclusion between `base_y` and `stick` | `final/final.xml` |
| MPC solve failures caused unstable behavior | Optimizer can fail/infeasible on some steps | Added runtime fallback to clipped one-step linear feedback | `final/control_core.py` |
| MPC was stable but could drift into persistent backward pitch mode | Slow model mismatch and insufficient late-stage recovery | Added terminal/reference shaping, pitch I-term, pitch rescue guard, and base-x pitch support blending | `final/runtime_config.py`, `final/mpc_controller.py`, `final/control_core.py`, `final/final.py` |
| Repeated disturbances could ratchet wheel speed bias and reduce recovery authority | Wheel momentum remained pre-biased between disturbance events | Added hold-phase wheel momentum bias integrator that uses small base-x bias to bleed wheel speed toward zero | `final/control_core.py`, `final/controller_eval.py` |
| Crash logs were hard to interpret | Failures were grouped as generic tilt | Added explicit `pitch_tilt` vs `roll_tilt` vs `com_overload` reason logging | `final/final.py` |

## 4.1) Troubleshooting Matrix

Use this matrix when behavior regresses or a run "looks wrong".

| Symptom | Quick checks | Likely cause | Action |
|---|---|---|---|
| Falls backward repeatedly (pitch) | Inspect runtime logs for rising `pitch` with near-zero `roll` | Pitch drift accumulation or weak late recovery | Increase pitch hardening: `--mpc-pitch-i-gain`, `--mpc-pitch-i-clamp`, `--mpc-pitch-guard-kp`, `--mpc-pitch-guard-kd`, lower `--mpc-pitch-guard-angle-frac` |
| Tilts sideways (roll) | Confirm crash reason is `roll_tilt` and check `u_by` activity | Roll authority/sign mismatch | Verify actuator wiring in XML (`base_y_force` mapping), tune roll gains/limits, confirm `--allow-base-motion` is enabled |
| Stick tilts but whole robot pose looks wrong | Check if free-joint attitude drifts in long runs | Uncontrolled root attitude mode | Keep `--lock-root-attitude` enabled (default) |
| Wheel appears not touching ground visually | Compare physical wheel radius and visual sidewall radius | Visual geometry mismatch | Align visual geoms to physical contact geom in `final/final.xml` (`drive_wheel_phys`, sidewall sizes) |
| `solve_discrete_are` throws ill-conditioned/no-finite-solution errors | Confirm where failure happens (`Controller` vs `Paper pitch`) | Numerical conditioning near marginal modes | Use robust DARE path (already default), avoid forcing fragile controller family until model/regime is valid |
| MPC "fails" intermittently or acts like commands drop | Check whether fallback path is active | QP solve/infeasibility on some steps | Keep runtime fallback enabled (default), tune horizon/costs and constraints (`--mpc-horizon`, `--mpc-q-*`, `--mpc-r-control`) |
| High crash count only in long runs | Compare short-run vs long-run metrics | Slow drift reaching hard limits | Use long headless benchmark, analyze trace terms, prioritize drift compensation over short-run smoothness |
| COM overload crashes | Check `com_xy` vs payload support radius in logs | COM boundary exceeded under load/drift | Tune base support behavior, reduce payload or increase support radius (`--payload-support-radius-m`) for experiments |
| Base commands saturate at limits | Monitor `u_bx`, `u_by` and sat metrics | Gains too aggressive or actuator limits too tight | Reduce aggressive gains or increase penalties (`--mpc-r-control`), re-tune `max_u`/`max_du` profiles carefully |
| Viewer looks stable but benchmark fails | Compare same mode/flags between viewer and headless run | Configuration mismatch | Re-run benchmark with explicit flags and fixed seed, then replay with same config in viewer |

## 5) Reproduce The Current System

### 5.1 Environment Setup

```powershell
cd C:\Users\User\Mujoco-series
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install osqp matplotlib pytest
```

Optional:

1. `pip install torch` if you want residual model support.

Runtime tuning:

1. `final/config.yaml` is auto-loaded when present.
2. Override with `--config path/to/config.yaml`.

### 5.2 Run Interactive Viewer

```powershell
python final/final.py --mode smooth
python final/final.py --mode robust
python final/final.py --mode robust --controller-family current_dob --dob-cutoff-hz 5.0
python final/final.py --use-mpc --mode robust
```

Mouse interaction in the viewer:

1. Left-double-click a body to select it.
2. Hold `Ctrl` and drag the mouse to apply force/torque to the selected body.
3. Drag without `Ctrl` to move the camera.

Useful runtime flags for manual perturbation sessions:

1. `--planar-perturb` / `--no-planar-perturb`: keep/remove Z-force clamping during perturbation.
2. `--drag-assist` / `--no-drag-assist`: temporarily scale controller effort while mouse perturbation is active.

### 5.3 Run Reproducible Benchmark (Headless)

```powershell
python final/benchmark.py --benchmark-profile fast_pr --episodes 12 --steps 6000 --trials 0 --controller-families current,current_dob,hybrid_modern,paper_split_baseline,baseline_mpc,baseline_robust_hinf_like --model-variants nominal --domain-rand-profile default --compare-modes default-vs-low-spin-robust --primary-objective balanced
```

Outputs are written to `final/results/`:

1. `benchmark_<timestamp>.csv`
2. `benchmark_<timestamp>_summary.txt`
3. `benchmark_<timestamp>_protocol.json`
4. `benchmark_<timestamp>_release_bundle.json`

### 5.4 Capture Detailed Runtime Logs

```powershell
python final/final.py --use-mpc --mode robust --log-control-terms --control-terms-csv final/results/control_terms_debug.csv --trace-events-csv final/results/runtime_trace_debug.csv
```

### 5.5 Sim-to-Real Discrepancy Workflow

Use this process after exporting `controller_params.h` and running hardware bring-up:

1. Generate firmware params with conservative limits.
```powershell
python final/export_firmware_params.py --mode robust --real-hardware --out final/firmware/controller_params.h
```
2. Confirm parity lock before flashing.
```powershell
python -m pytest final/test_export_parity.py -q
python -m pytest final/test_firmware_scenario_parity.py -q
```
3. Run benchmark with model/domain stress variants and hardware replay trace.
```powershell
python final/benchmark.py --benchmark-profile fast_pr --episodes 8 --steps 3000 --trials 0 --controller-families current --model-variants nominal,inertia_plus,inertia_minus,friction_low,friction_high,com_shift --domain-rand-profile default,rand_light,rand_medium,rand_heavy --compare-modes default-vs-low-spin-robust --primary-objective balanced --hardware-trace-path path/to/hardware_trace.csv
```
4. Generate sensitivity ranking from the benchmark CSV.
```powershell
python final/sim2real_sensitivity.py --csv final/results/benchmark_<timestamp>.csv
```
5. Record findings in:
`docs/SIM_TO_REAL_DISCREPANCY_LOG.md`

Full runbook:
`docs/SIM_TO_REAL_WORKFLOW.md`

### 5.6 Residual Model Workflow

Runtime behavior:

1. Residual model input is `x_est[9] + u_eff_applied[3] + u_nominal[3]` (15 features).
2. Output is additive `delta_u[3]`.
3. `--residual-scale` multiplies residual amplitude before final per-axis clipping.

Data/build/train path:

1. Capture paired baseline and teacher traces (same seed/disturbance):
```powershell
python final/final.py --mode robust --controller-family current --seed 12345 --trace-events-csv final/results/trace_baseline.csv
python final/final.py --mode robust --controller-family hybrid_modern --seed 12345 --trace-events-csv final/results/trace_teacher.csv
```
2. Build dataset:
```powershell
python final/build_residual_dataset.py --baseline-trace final/results/trace_baseline.csv --teacher-trace final/results/trace_teacher.csv --out final/results/residual_dataset.npz
```
3. Train checkpoint:
```powershell
python final/train_residual_model.py --dataset final/results/residual_dataset.npz --out final/results/residual.pt --epochs 50 --batch-size 1024 --hidden-dim 32 --hidden-layers 2
```
4. Evaluate:
```powershell
python final/final.py --mode robust --residual-model final/results/residual.pt --residual-scale 0.20 --log-control-terms --trace-events-csv final/results/trace_residual_eval.csv
```

Detailed guide:
`docs/RESIDUAL_MODEL_GUIDE.md`

### 5.7 Adaptive Disturbance Observer + Gain Scheduling

This feature is optional and can run on both LQR and MPC paths:

1. Disturbance observer estimates unknown additive input disturbances in actuator space (`rw`, `bx`, `by`).
2. Compensation term subtracts that estimate from the command.
3. `--dob-cutoff-hz` is the primary Q-filter tuning knob (typical starting range: 3-8 Hz).
4. Gain scheduling scales stabilization gains up under large estimated disturbance and back down in calm state (LQR shaping path).

Example:

```powershell
python final/final.py --mode robust --controller-family current_dob --dob-cutoff-hz 5.0 --dob-leak-per-s 0.6 --dob-max-rw 6.0 --dob-max-bx 1.0 --dob-max-by 1.0 --enable-gain-scheduling --gain-sched-min 1.0 --gain-sched-max 1.8 --gain-sched-ref 2.0 --gain-sched-rate-per-s 3.0

# MPC + DOB feed-forward
python final/final.py --mode robust --use-mpc --enable-dob --dob-cutoff-hz 5.0 --dob-leak-per-s 0.6 --dob-max-rw 6.0 --dob-max-bx 1.0 --dob-max-by 1.0
```

For tuning visibility, run with:

```powershell
python final/final.py --mode robust --controller-family current_dob --dob-cutoff-hz 5.0 --enable-gain-scheduling --log-control-terms --trace-events-csv final/results/runtime_trace_dob.csv
```

### 5.8 Online ID + Adaptive LQR Gain Scheduling

Optional runtime adaptation for payload/plant drift:

1. RLS estimates effective pitch stiffness and control authority from telemetry.
2. Adapted `A/B` rows are rebuilt from those estimates.
3. `K_du` is recomputed every N control updates with blend/rate limits.

Example:

```powershell
python final/final.py --mode robust --enable-online-id --online-id-recompute-every 25 --online-id-min-updates 60
```

### 5.9 Crash-Coupling Disturbance Analysis

Use this to replay one episode and inspect the six disturbance events before crash (or end-of-episode if no crash):

```powershell
# 1-based episode index (default indexing in throughput_ablation JSON)
python final/analyze_crash_coupling.py --episode-index 20 --window-start 4600 --window-end 5600 --events 6

# 0-based index if your notes label "episode 20" as the 21st seed
python final/analyze_crash_coupling.py --episode-index 20 --zero-based-index --window-start 4600 --window-end 5600 --events 6
```

Output includes:

1. Disturbance-event rows (`step,wheel_speed,pitch,pitch_rate,force_xyz`).
2. Linear trend slopes for wheel speed and pitch.
3. `coupling_confirmed=1` when wheel speed trends negative while pitch trends positive across selected events.

### 5.10 LQR Envelope Boundary Search

Use binary search to estimate the largest disturbance XY magnitude where per-cycle pitch ratchet meets a target:

```powershell
python final/find_lqr_envelope.py --episode-indices 20 --disturbance-interval 200 --xy-min 0 --xy-max 12 --target-ratchet-deg-per-cycle 0.0
```

Key outputs:

1. `ratchet_mean` (deg/cycle): mean increase in `|pitch|` between disturbance events.
2. `slope100_mean` (deg/s): mean 100-step end-of-window slope of `|pitch|`.
3. `boundary_xy_approx`: estimated disturbance magnitude boundary for the selected target.

## 6) Rebuild From Scratch (Recommended Path)

If someone wants to recreate this project from zero, use this exact sequence.

1. Build a minimal MuJoCo plant in XML.
Include a ground plane, base free-joint, stick pitch/roll joints, reaction wheel joint, and three actuators.
Success check: model loads and steps without NaNs.

2. Implement runtime skeleton.
Load model, resolve names, read true state, apply raw controls.
Success check: actuator commands match expected joints/signs.

3. Add linearization and baseline LQR.
Linearize around upright, build delta-u controller, clip commands.
Success check: recovers from small initial tilt.

4. Add estimator and hardware realism.
Inject sensor noise + control delay + quantization, then Kalman correction.
Success check: no unstable drift from estimator mismatch alone.

5. Add safety envelopes.
Wheel speed budget/hard zone, actuator derates, COM support checks, crash logic.
Success check: no runaway command/wheel behavior.

6. Add evaluation harness.
Run many seeds/episodes headlessly and persist artifacts for comparison.
Success check: benchmark outputs are deterministic and comparable across changes.

7. Add MPC and harden difficult modes.
Use constrained MPC with runtime fallback, then add targeted anti-drift logic for observed failure modes.
Success check: better crash behavior under stress tests, not just easier nominal runs.

## 7) Practical Tuning Workflow (Pitch-First)

When pitch is your dominant failure mode, tune in this order:

1. Verify model/control wiring first.
Confirm `wheel_spin`, `base_x_force`, `base_y_force` map to intended joints.

2. Stabilize optimization behavior.
Tune `--mpc-horizon`, `--mpc-q-angles`, `--mpc-q-rates`, `--mpc-r-control`, `--mpc-terminal-weight`.

3. Shape pitch transients.
Tune `--mpc-target-rate-gain`, `--mpc-terminal-rate-gain`, `--mpc-target-rate-clip`.

4. Compensate slow drift.
Tune `--mpc-pitch-i-gain`, `--mpc-pitch-i-clamp`, `--mpc-pitch-i-deadband-deg`, `--mpc-pitch-i-leak-per-s`.

5. Add emergency behavior.
Tune `--mpc-pitch-guard-angle-frac`, `--mpc-pitch-guard-rate`, `--mpc-pitch-guard-kp`, `--mpc-pitch-guard-kd`.

6. Validate with both nominal and stressed runs.
Use scripted pushes and benchmark seeds, not only one clean viewer run.

### 7.1 Current Pitch-Urgency Defaults

Current fast-failure prevention tuning focuses on earlier pitch correction:

1. RW `max_du` is kept at baseline defaults (no extra delta-u widening).
2. LQR pitch-angle weight (`Qx[0,0]`) is increased by ~2.5x so pitch growth is treated as urgent much earlier.
3. Crash console output includes `pitch_rate`, `roll_rate`, and `wheel_rate` at failure time to distinguish gain vs authority-limit issues.

## 8) Important Boundaries

1. This is simulation-first engineering work, not a hardware safety certification.
2. Some long-horizon drift behavior can be plant-limited with the current geometry/actuation.
3. Benchmark conclusions are valid for the tested settings and artifacts in `final/results/`.
4. Viewer behavior and headless benchmark behavior should be compared, not assumed identical without checking logs.

## 9) Useful Commands

```powershell
# Show all runtime flags
python final/final.py --help

# MPC robust run
python final/final.py --use-mpc --mode robust

# Quick baseline benchmark
python final/benchmark.py --benchmark-profile fast_pr --episodes 8 --steps 3000 --trials 0

# Export firmware-friendly controller header
python final/export_firmware_params.py --mode robust --out final/firmware/controller_params.h

# Rank sensitivity factors from a benchmark CSV
python final/sim2real_sensitivity.py --csv final/results/benchmark_<timestamp>.csv

# Build and train residual model from paired traces
python final/build_residual_dataset.py --baseline-trace final/results/trace_baseline.csv --teacher-trace final/results/trace_teacher.csv --out final/results/residual_dataset.npz
python final/train_residual_model.py --dataset final/results/residual_dataset.npz --out final/results/residual.pt
```

---

If you are extending this project, keep the same discipline:
change one layer at a time, benchmark after each change, and log enough data to prove the change helped.
