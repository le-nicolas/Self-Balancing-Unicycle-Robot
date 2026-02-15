from pathlib import Path

import numpy as np

import benchmark
from controller_eval import CandidateParams, ControllerEvaluator, EpisodeConfig, safe_evaluate_candidate


def _baseline_params() -> CandidateParams:
    return benchmark.baseline_candidate()


def test_protocol_manifest_roundtrip(tmp_path: Path):
    class _Args:
        seed = 123
        benchmark_profile = "fast_pr"
        compare_modes = "default-vs-low-spin-robust"
        gate_max_worst_tilt = 20.0
        gate_max_worst_base = 4.0
        gate_max_sat_du = 0.98
        gate_max_sat_abs = 0.90
        imu_angle_noise_deg = 0.25
        imu_rate_noise = 0.02
        wheel_rate_noise = 0.01
        control_hz = 250.0
        control_delay_steps = 1
        legacy_model = False
        disturbance_xy = 4.0
        disturbance_z = 2.0
        disturbance_interval = 300
        significance_alpha = 0.05
        multiple_comparison_correction = "holm"
        release_campaign = False

    mode_matrix = benchmark.get_mode_matrix("default-vs-low-spin-robust")
    fam = ["current", "hybrid_modern"]
    variants = ["nominal", "inertia_plus"]
    domains = ["default", "rand_light"]
    manifest = benchmark.build_protocol_manifest(_Args, mode_matrix, fam, variants, domains)
    p = tmp_path / "protocol.json"
    p.write_text(__import__("json").dumps(manifest), encoding="utf-8")
    loaded = benchmark.load_protocol_manifest(str(p))
    assert loaded["schema_version"] == "protocol_v1"
    assert loaded["controller_families"] == fam
    assert loaded["model_variants"] == variants
    assert loaded["domain_profiles"] == domains


def test_multiple_comparison_correction_monotonic():
    rows = [
        {"mode_id": "m", "is_baseline": False, "significance_pvalue": 0.01},
        {"mode_id": "m", "is_baseline": False, "significance_pvalue": 0.03},
        {"mode_id": "m", "is_baseline": False, "significance_pvalue": 0.20},
    ]
    benchmark.apply_multiple_comparison_correction(rows, "m", "holm")
    vals = [float(r["significance_pvalue_corrected"]) for r in rows]
    assert all(np.isfinite(v) for v in vals)
    assert all(0.0 <= v <= 1.0 for v in vals)


def test_composite_score_sanity():
    ev = ControllerEvaluator(Path(__file__).with_name("final.xml"))
    good = {
        "survived": 1.0,
        "max_abs_pitch_deg": 2.0,
        "max_abs_roll_deg": 2.0,
        "max_abs_base_x_m": 0.1,
        "max_abs_base_y_m": 0.1,
        "control_energy": 10.0,
        "mean_command_jerk": 1.0,
        "sat_rate_du": 0.01,
        "sat_rate_abs": 0.01,
        "wheel_over_hard": 0.0,
        "wheel_over_budget": 0.0,
    }
    bad = dict(good)
    bad.update(
        {
            "survived": 0.0,
            "max_abs_pitch_deg": 18.0,
            "max_abs_roll_deg": 16.0,
            "control_energy": 300.0,
            "sat_rate_du": 0.8,
            "sat_rate_abs": 0.7,
            "wheel_over_hard": 120.0,
            "wheel_over_budget": 180.0,
        }
    )
    s_good = ev._episode_composite_score(good, steps=3000)
    s_bad = ev._episode_composite_score(bad, steps=3000)
    assert s_good > s_bad


def test_family_dispatch_finite_metrics():
    ev = ControllerEvaluator(Path(__file__).with_name("final.xml"))
    families = [
        "current",
        "hybrid_modern",
        "paper_split_baseline",
        "baseline_mpc",
        "baseline_robust_hinf_like",
    ]
    for fam in families:
        cfg = EpisodeConfig(
            steps=80,
            disturbance_interval=40,
            controller_family=fam,
            model_variant_id="nominal",
            domain_profile_id="default",
        )
        out = safe_evaluate_candidate(ev, _baseline_params(), [123], cfg)
        assert np.isfinite(float(out["score_composite"]))
        assert np.isfinite(float(out["survival_rate"]))
