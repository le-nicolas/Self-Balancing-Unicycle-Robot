# Self-Balancing Unicycle Robot

![Status](https://img.shields.io/badge/status-active_research-orange)
![Scope](https://img.shields.io/badge/scope-simulation_only-blue)
![Maintained](https://img.shields.io/badge/maintained_path-final%2F-green)
![Evidence](https://img.shields.io/badge/claims-artifact_backed-brightgreen)

**Project Status (Quick Read):**
- Maturity: active development and benchmarking, not a frozen release.
- Primary code path: `final/`.
- Ground truth for claims: saved benchmark artifacts in `final/results/`.
- CI path: `.github/workflows/ci.yml` runs parity tests plus a fast benchmark smoke profile on push/PR.

This repository is a simulation-first control project for a balancing robot with a reaction wheel and a moving base.

It is written for two audiences:
- non-technical readers who want to understand what this project proves
- technical readers who want to reproduce, inspect, and extend the control stack

## 0) Repository Layout

- `final/`: active simulator, controller runtime, firmware export, and benchmark tooling.
- `meshes/`: shared geometry assets.
- `docs/`: workflow and analysis documentation.
- `archive/`: legacy prototypes and historical logs moved out of the root.
- `archive/legacy/scripts/README.md`: per-script intent map for archived iteration files.
- `archive/investigations/model_integrity/`: one-off investigation scripts from the 2026-02-21 structural bug analysis.
- `web/`: browser viewer side quest.
- `wheel-base-controller-release/`: standalone minimal wheel-and-base package snapshot.

## 0.5) Current Status and Transparency (as of 2026-02-21)

- This repo is an active research/dev sandbox, not a frozen release.
- The only actively maintained control stack is under `final/`.
- The repository root is intentionally minimal; runnable/project scripts live under `final/` and archived investigations live under `archive/`.
- Historical files/prototypes are retained in `archive/` for traceability, not as recommended entry points.
- Legacy script intents are indexed in `archive/legacy/scripts/README.md` so canonical vs historical files are explicit.
- The benchmark table in this README (`2026-02-21` snapshot) is a reproducible reference point, not a claim that all newer stress cases are solved.
- Recent fixed-seed stress diagnostics (`seed=42`, 80 episodes, high disturbance profile) are stored in:
  - `final/results/throughput_ablation_seed42.json`
  - `final/results/throughput_ablation_seed42_extra.json`
- Current dominant failure mode in those stress diagnostics is trajectory/phase recovery throughput under repeated disturbances, not a simple control-sign inversion.
- Optional features (`--enable-dob`, `--enable-gain-scheduling`, residual model path) are experimental and should be treated as such unless benchmarked with saved artifacts.
- Development and review are AI-assisted; benchmark artifacts and reproducible scripts are the source of truth for project claims.

## 1) Non-Technical Overview

### What this project does



https://github.com/user-attachments/assets/ca2fff7d-20d3-4f47-b21a-2c3a8b4247ea



<img width="836" height="696" alt="Screenshot 2026-02-15 134403" src="https://github.com/user-attachments/assets/9bb79145-424a-46d5-af1c-76f9d9f833f4" />



Imagine trying to balance a broom upright on your hand.
- The broom (robot body) wants to fall.
- A fast spinning wheel inside the robot pushes back against that fall.
- The base can also move under the robot to help catch it.
- Software repeats this correction loop many times per second.

This project builds that behavior in simulation using MuJoCo.

### Why simulation-only matters here

The goal is not just to make one demo run.  
The goal is to make results repeatable, testable, and comparable under controlled conditions:
- sensor noise
- control delay
- disturbances
- multiple controller families
- repeatable benchmark artifacts

### What we are claiming

- We can reproduce balancing objectives in simulation with explicit robustness tests.
- We provide reproducible benchmark artifacts under `final/results/`.
- We provide transparent comparison scripts and data.

### What we are not claiming

- We are not claiming certified hardware safety.
- We are not claiming this automatically beats every physical robot implementation.
- Literature comparison is for context and reproducibility, not direct hardware replacement claims.

### Historical context: 2013 hardware paper

Reference paper:
- J. Lee, S. Han, J. Lee, "Decoupled Dynamic Control for Pitch and Roll Axes of the Unicycle Robot," IEEE Transactions on Industrial Electronics, Vol. 60, No. 9, September 2013
- DOI: `10.1109/TIE.2012.2208431`
- Local review source: `C:/Users/User/Downloads/onebot.pdf`

Paper summary:
- Real hardware unicycle robot
- Decoupled control design (roll and pitch handled separately)
- Reported trajectory-following experiments with bounded angle errors

### Statistical visual comparison (paper context vs this simulation run)

![Paper vs simulation statistical comparison](final/results/paper_vs_sim_comparison.png)

Figure source:
- CSV: `final/results/benchmark_20260216_011044.csv`
- Plot script: `final/plot_paper_vs_sim.py`

Re-generate:
```bash
python final/plot_paper_vs_sim.py --csv final/results/benchmark_20260216_011044.csv --out final/results/paper_vs_sim_comparison.png
```

## 2) Technical Reference

### 2.1 System architecture

- Physics model: `final/final.xml`
- Runtime controller and simulator loop: `final/final.py`
- Evaluator/benchmark harness: `final/controller_eval.py`, `final/benchmark.py`

Runtime stack:
- state estimation: linear Kalman correction from noisy channels
- control core: delta-u LQR with controller-family-specific shaping
- optional online ID + adaptive LQR scheduling: RLS estimates effective pitch stiffness/control authority and periodically refreshes `K`
- optional adaptive disturbance observer + gain scheduling: estimate unknown additive disturbances and scale stabilizing gains in real time
- optional residual correction: offline PyTorch model adds bounded `delta_u` to nominal command
- safety shaping: saturation and rate limits, wheel-speed budget/high-spin logic, strict hard-speed same-direction suppression, base-authority gating, crash-angle handling (centrally tunable in `final/config.yaml`)

Why recent IMU/sensor realism changes were added:
- avoid overfitting to idealized state feedback
- surface long-horizon drift/failure modes earlier (bias + delay + clipping effects)
- make benchmark robustness claims stronger under non-ideal sensing
- improve sim-to-real relevance for future embedded deployment

General effect:
- recovery may look slightly slower/less sharp
- controllers tuned on ideal sensing may need retuning
- behavior is more realistic and more representative of hardware constraints

### 2.2 Controller families in benchmark

- `current`
- `current_dob` (`current` + DOB feed-forward compensation)
- `hybrid_modern`
- `paper_split_baseline`
- `baseline_mpc` (reference constrained-MPC baseline, intentionally kept without low-spin hardening add-ons)
- `baseline_robust_hinf_like`

### 2.3 Benchmark snapshot (2026-02-21)

Canonical artifacts:
- `final/results/benchmark_20260221_100424_summary.txt`
- `final/results/benchmark_20260221_100424.csv`
- `final/results/benchmark_20260221_100424_protocol.json`

Mode shown below: `mode_default / nominal / default`

| Controller | Survival | Crash rate | Composite score |
|---|---:|---:|---:|
| `baseline_mpc` | 1.000 | 0.000 | 86.793 |
| `current` | 1.000 | 0.000 | 81.862 |
| `hybrid_modern` | 1.000 | 0.000 | 80.727 |
| `baseline_robust_hinf_like` | 1.000 | 0.000 | 80.232 |
| `paper_split_baseline` | 1.000 | 0.000 | 69.169 |

`baseline_mpc` is still a comparator baseline, not the default deployment path.
Historical note: an earlier artifact (`final/results/benchmark_20260216_011044.csv`, generated 2026-02-16) recorded `baseline_mpc` crash rate `0.375`; current tuned defaults in this snapshot are improved.
Longer and harsher profiles can still separate controllers even when this fast smoke profile passes for all families.

### 2.4 Reproducibility

Benchmark command used for this snapshot:
```bash
python final/benchmark.py --benchmark-profile fast_pr --episodes 8 --steps 2000 --trials 0 --controller-families current,current_dob,hybrid_modern,paper_split_baseline,baseline_mpc,baseline_robust_hinf_like --model-variants nominal --domain-rand-profile default --compare-modes default-vs-low-spin-robust --primary-objective balanced
```

Expected outputs:
- `final/results/benchmark_<timestamp>_summary.txt`
- `final/results/benchmark_<timestamp>.csv`
- `final/results/benchmark_<timestamp>_protocol.json`

### 2.5 Run the simulator

Install:
```bash
pip install -r requirements.txt
```

Optional for residual correction:
```bash
pip install torch
```

Viewer run:
```bash
python final/final.py --mode smooth
```

Payload overload run (physical payload attached on top of the stick):
```bash
python final/final.py --mode smooth --payload-mass 0.8
```

Robust profile:
```bash
python final/final.py --mode robust --stability-profile low-spin-robust
```

Adaptive online-ID profile (payload variation hardening):
```bash
python final/final.py --mode robust --enable-online-id --online-id-recompute-every 25 --online-id-min-updates 60
```

Controller family selection:
```bash
python final/final.py --mode smooth --controller-family current
python final/final.py --mode smooth --controller-family current_dob --dob-cutoff-hz 5.0
python final/final.py --mode smooth --controller-family hybrid_modern
python final/final.py --mode smooth --controller-family paper_split_baseline
```

Optional residual correction run:
```bash
python final/final.py --mode smooth --residual-model path/to/residual.pt --residual-scale 0.20
```
Training and deployment details:
- `docs/RESIDUAL_MODEL_GUIDE.md`
- `final/build_residual_dataset.py`
- `final/train_residual_model.py`

Central runtime tuning file:
- `final/config.yaml` (auto-loaded when present; override with `--config path/to/config.yaml`)

Web side quest (Brave, live runtime-connected):
```bash
python web/server.py --port 8090
```
Open `http://localhost:8090/web/` and use the payload mass controls.

Wheel-spin diagnostics:
- runtime summary now reports signed wheel speed peaks, mean absolute wheel speed, over-budget/over-hard ratios (total and sign-split), and hard-zone suppression counters for tuning.
- `--log-control-terms` and `--trace-events-csv` include wheel-rate/budget/hard telemetry fields.

### 2.6 Firmware relevance

This simulation is structured to map to embedded workflows:
- fixed-rate control pattern
- bounded actuator guardrails
- parameter export parity checks

Files:
- `final/firmware/README.md`
- `final/export_firmware_params.py`
- `final/test_export_parity.py`
- `final/test_firmware_scenario_parity.py`
- `docs/SIM_TO_REAL_WORKFLOW.md`
- `docs/SIM_TO_REAL_DISCREPANCY_LOG.md`
- `final/sim2real_sensitivity.py`

### 2.7 Project map

- Main runtime: `final/final.py`
- MuJoCo model: `final/final.xml`
- Online ID scheduler: `final/adaptive_id.py`
- Evaluator: `final/controller_eval.py`
- Benchmark driver: `final/benchmark.py`
- Sim-to-real sensitivity summarizer: `final/sim2real_sensitivity.py`
- Paper-vs-sim graph generator: `final/plot_paper_vs_sim.py`
- Final folder guide: `final/README.md`
- Documentation index: `docs/README.md`
- Legacy archive index: `archive/README.md`

### 2.8 Limitations

- Simulation results only
- No certified hardware safety claim
- Contact/friction/sensor models are approximations
- Real hardware still requires calibration, staged testing, and safety interlocks
