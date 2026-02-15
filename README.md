# MuJoCo Reaction Wheel Balancer

This repository is a simulation-first control project for a balancing robot with a reaction wheel and a moving base.

It is written for two audiences:
- non-technical readers who want to understand what this project proves
- technical readers who want to reproduce, inspect, and extend the control stack

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
- safety shaping: saturation and rate limits, wheel-speed budget, high-spin logic, base-authority gating, crash-angle handling

### 2.2 Controller families in benchmark

- `current`
- `hybrid_modern`
- `paper_split_baseline`
- `baseline_mpc`
- `baseline_robust_hinf_like`

### 2.3 Benchmark snapshot (2026-02-16)

Canonical artifacts:
- `final/results/benchmark_20260216_011044_summary.txt`
- `final/results/benchmark_20260216_011044.csv`
- `final/results/benchmark_20260216_011044_protocol.json`

Mode shown below: `mode_default / nominal / default`

| Controller | Survival | Crash rate | Composite score |
|---|---:|---:|---:|
| `baseline_robust_hinf_like` | 1.000 | 0.000 | 75.457 |
| `hybrid_modern` | 1.000 | 0.000 | 74.460 |
| `current` | 1.000 | 0.000 | 73.993 |
| `paper_split_baseline` | 1.000 | 0.000 | 72.913 |
| `baseline_mpc` | 0.625 | 0.375 | 37.837 |

### 2.4 Reproducibility

Benchmark command used for this snapshot:
```bash
python final/benchmark.py --benchmark-profile fast_pr --episodes 8 --steps 3000 --trials 0 --controller-families current,hybrid_modern,paper_split_baseline,baseline_mpc,baseline_robust_hinf_like --model-variants nominal --domain-rand-profile default --compare-modes default-vs-low-spin-robust --primary-objective balanced
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

Viewer run:
```bash
python final/final.py --mode smooth
```

Robust profile:
```bash
python final/final.py --mode robust --stability-profile low-spin-robust
```

Controller family selection:
```bash
python final/final.py --mode smooth --controller-family current
python final/final.py --mode smooth --controller-family hybrid_modern
python final/final.py --mode smooth --controller-family paper_split_baseline
```

### 2.6 Firmware relevance

This simulation is structured to map to embedded workflows:
- fixed-rate control pattern
- bounded actuator guardrails
- parameter export parity checks

Files:
- `final/firmware/README.md`
- `final/export_firmware_params.py`
- `final/test_export_parity.py`

### 2.7 Project map

- Main runtime: `final/final.py`
- MuJoCo model: `final/final.xml`
- Evaluator: `final/controller_eval.py`
- Benchmark driver: `final/benchmark.py`
- Paper-vs-sim graph generator: `final/plot_paper_vs_sim.py`
- Final folder guide: `final/README.md`

### 2.8 Limitations

- Simulation results only
- No certified hardware safety claim
- Contact/friction/sensor models are approximations
- Real hardware still requires calibration, staged testing, and safety interlocks
