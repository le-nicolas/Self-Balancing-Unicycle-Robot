# Reality Check Summary (Metrics-First)

This file tracks measurable evidence for robustness and sim-to-real readiness using saved artifacts.

## Artifact sources

- Fast benchmark snapshot (`2026-02-21`): `final/results/benchmark_20260221_100424.csv`
- Historical MPC snapshot (`2026-02-16`): `final/results/benchmark_20260216_011044.csv`
- Sensitivity reference run (`2026-02-15`): `final/results/benchmark_20260215_215203.csv`
- Hardware-readiness report (`2026-02-15`): `final/results/readiness_20260215_220928.json`

## 1) Controller robustness snapshot

Scope: baseline rows, `mode_default / nominal / default`, benchmark artifact `benchmark_20260221_100424.csv`.

| Controller family | Survival rate | Crash rate | Composite score |
|---|---:|---:|---:|
| `baseline_mpc` | 1.000 | 0.000 | 86.793 |
| `current` | 1.000 | 0.000 | 81.862 |
| `hybrid_modern` | 1.000 | 0.000 | 80.727 |
| `baseline_robust_hinf_like` | 1.000 | 0.000 | 80.232 |
| `paper_split_baseline` | 1.000 | 0.000 | 69.169 |

## 2) MPC status (historical underperformance vs current tuning)

| Artifact date | Artifact | `baseline_mpc` survival | `baseline_mpc` crash | `baseline_mpc` score |
|---|---|---:|---:|---:|
| 2026-02-16 | `benchmark_20260216_011044.csv` | 0.625 | 0.375 | 37.837 |
| 2026-02-21 | `benchmark_20260221_100424.csv` | 1.000 | 0.000 | 86.793 |

Interpretation:
- The earlier 0.375 crash-rate snapshot is real and preserved for traceability.
- Current defaults no longer show that failure on the fast benchmark profile.
- `baseline_mpc` remains a comparator baseline, not the recommended deployment family.

## 3) Sim-to-real gap proxy table (quantified)

Method:
- Use nominal/default rows as anchor per `(mode, family)`.
- Measure deltas under model/domain stress factors.
- Aggregate using `final/sim2real_sensitivity.py`.

Source:
- `benchmark_20260215_215203.csv`
- Generated from `final/sim2real_sensitivity.py` on 2026-02-21 (command in section 5)

| Factor | Samples | Mean abs Δ score | Mean abs Δ worst tilt (deg) | Mean sensitivity index |
|---|---:|---:|---:|---:|
| `inertia_plus` | 10 | 6.5865 | 0.0660 | 6.6360 |
| `rand_light` | 10 | 2.4938 | 0.0424 | 2.5256 |

Interpretation:
- In this reference run, inertia mismatch drives larger performance drift than light domain randomization.

## 4) Hardware replay/readiness metrics

Source: `readiness_20260215_220928.json` and `benchmark_20260215_220928.csv`.

| Metric | Value | Target / expectation | Status |
|---|---:|---:|---|
| Deterministic rerun Δ score | 0.0 | 0.0 | Pass |
| Deterministic rerun Δ survival | 0.0 | 0.0 | Pass |
| `readiness_sign_sanity_pass` | 0 / 6 rows | 6 / 6 rows | Fail |
| `readiness_replay_pass` | 0 / 6 rows | 6 / 6 rows | Fail |
| `readiness_overall_pass` | 0 / 6 rows | 6 / 6 rows | Fail |
| `sim_real_consistency_mean` | n/a (no hardware trace replay) | >= 0.60 | Not measured |
| `sim_real_traj_nrmse_mean` | n/a (no hardware trace replay) | <= 0.75 | Not measured |

Current gap to close:
- This repo has quantified simulation robustness and sensitivity, but still needs hardware-trace replay metrics to quantify true sim-to-real fidelity.

## 5) Repro commands

```powershell
# Fast benchmark snapshot
python final/benchmark.py --benchmark-profile fast_pr --episodes 8 --steps 2000 --trials 0 --controller-families current,hybrid_modern,paper_split_baseline,baseline_mpc,baseline_robust_hinf_like --model-variants nominal --domain-rand-profile default --compare-modes default-vs-low-spin-robust --primary-objective balanced

# Sensitivity summary from a benchmark CSV that includes variant/domain factors
python final/sim2real_sensitivity.py --csv final/results/benchmark_20260215_215203.csv --out-prefix final/results/sim2real_sensitivity_20260215_215203_ref

# Readiness report run (requires hardware trace input to fully measure replay metrics)
python final/benchmark.py --release-campaign --episodes 1 --trials 0 --steps 120 --controller-families current,hybrid_modern,paper_split_baseline --model-variants nominal --domain-rand-profile default --compare-modes default-vs-low-spin-robust --primary-objective balanced --readiness-report --hardware-trace-path path/to/hardware_trace.csv
```
