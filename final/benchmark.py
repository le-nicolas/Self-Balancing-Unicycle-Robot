import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
import time
from typing import Dict, List, Tuple

import numpy as np

from controller_eval import CandidateParams, ControllerEvaluator, EpisodeConfig, safe_evaluate_candidate


R_DU_BOUNDS = (0.01, 50.0)
R_DU_RW_BOUNDS = (0.1, 50.0)
Q_SCALE_BOUNDS = (0.2, 5.0)
QU_SCALE_BOUNDS = (0.2, 5.0)
KI_BASE_BOUNDS = (0.0, 0.40)
U_BLEED_BOUNDS = (0.85, 0.999)
MAX_DU_RW_BOUNDS = (1.0, 80.0)
MAX_DU_BASE_BOUNDS = (0.1, 20.0)
NOVELTY_MIN_DISTANCE = 0.12
NOVELTY_MAX_ATTEMPTS = 300
DEFAULT_CONTROLLER_FAMILIES = (
    "current,hybrid_modern,paper_split_baseline,baseline_mpc,baseline_robust_hinf_like,legacy_wheel_pid,legacy_wheel_lqr,legacy_run_pd"
)
DEFAULT_MODEL_VARIANTS = "nominal,inertia_plus,friction_low,com_shift"
DEFAULT_DOMAIN_PROFILES = "default,rand_light,rand_medium"
PROTOCOL_SCHEMA_VERSION = "protocol_v1"


def parse_args():
    parser = argparse.ArgumentParser(description="Fast robustness benchmark + random-search tuner.")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--benchmark-profile",
        choices=["fast_pr", "nightly_long"],
        default="nightly_long",
        help="Runtime profile for CI/ops use.",
    )
    parser.add_argument(
        "--controller-families",
        type=str,
        default=DEFAULT_CONTROLLER_FAMILIES,
        help="Comma-separated controller families to benchmark.",
    )
    parser.add_argument("--significance-alpha", type=float, default=0.05)
    parser.add_argument("--protocol-manifest", type=str, default=None)
    parser.add_argument("--model-variants", type=str, default=DEFAULT_MODEL_VARIANTS)
    parser.add_argument("--domain-rand-profile", type=str, default=DEFAULT_DOMAIN_PROFILES)
    parser.add_argument("--release-campaign", action="store_true")
    parser.add_argument(
        "--multiple-comparison-correction",
        choices=["holm", "bonferroni", "none"],
        default="holm",
    )
    parser.add_argument("--hardware-trace-path", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--outdir", type=str, default="final/results")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument(
        "--compare-modes",
        choices=["default-vs-low-spin-robust", "stable-demo-vs-low-spin-robust", "all"],
        default="default-vs-low-spin-robust",
    )
    parser.add_argument(
        "--primary-objective",
        choices=["crash-rate", "wheel-spin", "balanced"],
        default="crash-rate",
    )
    parser.add_argument("--stress-level", choices=["fast", "medium", "long"], default="long")
    parser.add_argument("--init-angle-deg", type=float, default=4.0)
    parser.add_argument("--init-base-pos-m", type=float, default=0.15)
    parser.add_argument("--disturbance-xy", type=float, default=4.0)
    parser.add_argument("--disturbance-z", type=float, default=2.0)
    parser.add_argument("--disturbance-interval", type=int, default=300)
    parser.add_argument("--gate-max-worst-tilt", type=float, default=20.0)
    parser.add_argument("--gate-max-worst-base", type=float, default=4.0)
    parser.add_argument("--gate-max-sat-du", type=float, default=0.98)
    parser.add_argument("--gate-max-sat-abs", type=float, default=0.90)
    parser.add_argument("--control-hz", type=float, default=250.0)
    parser.add_argument("--control-delay-steps", type=int, default=1)
    parser.add_argument("--wheel-encoder-ticks", type=int, default=2048)
    parser.add_argument("--imu-angle-noise-deg", type=float, default=0.25)
    parser.add_argument("--imu-rate-noise", type=float, default=0.02)
    parser.add_argument("--wheel-rate-noise", type=float, default=0.01)
    parser.add_argument("--legacy-model", action="store_true", help="Disable hardware-realistic timing/noise model.")
    return parser.parse_args()


def parse_csv_list(raw: str) -> List[str]:
    vals = [v.strip() for v in str(raw).split(",") if v.strip()]
    return vals


def apply_stress_defaults(args):
    profile_defaults = {
        "fast_pr": {"episodes": 8, "trials": 20, "steps": 2000},
        "nightly_long": {"episodes": 40, "trials": 200, "steps": 6000},
    }[args.benchmark_profile]
    stress_defaults = {
        "fast": {"episodes": 12, "trials": 60, "steps": 3000},
        "medium": {"episodes": 24, "trials": 120, "steps": 4000},
        "long": {"episodes": 40, "trials": 200, "steps": 6000},
    }[args.stress_level]
    defaults = profile_defaults if args.benchmark_profile == "fast_pr" else stress_defaults
    episodes = int(args.episodes) if args.episodes is not None else defaults["episodes"]
    trials = int(args.trials) if args.trials is not None else defaults["trials"]
    steps = int(args.steps) if args.steps is not None else defaults["steps"]
    return episodes, trials, steps


def parse_controller_families(raw: str) -> List[str]:
    families = [f.strip() for f in str(raw).split(",") if f.strip()]
    if not families:
        return ["current"]
    unique = []
    for fam in families:
        if fam not in unique:
            unique.append(fam)
    return unique


def build_protocol_manifest(args, mode_matrix, controller_families, model_variants, domain_profiles) -> Dict[str, object]:
    return {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "seed": int(args.seed),
        "benchmark_profile": str(args.benchmark_profile),
        "compare_modes": str(args.compare_modes),
        "mode_matrix": [list(m) for m in mode_matrix],
        "controller_families": list(controller_families),
        "model_variants": list(model_variants),
        "domain_profiles": list(domain_profiles),
        "release_campaign": bool(args.release_campaign),
        "gates": {
            "max_worst_tilt": float(args.gate_max_worst_tilt),
            "max_worst_base": float(args.gate_max_worst_base),
            "max_sat_du": float(args.gate_max_sat_du),
            "max_sat_abs": float(args.gate_max_sat_abs),
        },
        "noise": {
            "imu_angle_noise_deg": float(args.imu_angle_noise_deg),
            "imu_rate_noise": float(args.imu_rate_noise),
            "wheel_rate_noise": float(args.wheel_rate_noise),
        },
        "timing": {
            "control_hz": float(args.control_hz),
            "control_delay_steps": int(args.control_delay_steps),
            "hardware_realistic": bool(not args.legacy_model),
        },
        "disturbance": {
            "xy": float(args.disturbance_xy),
            "z": float(args.disturbance_z),
            "interval": int(args.disturbance_interval),
        },
        "statistics": {
            "alpha": float(args.significance_alpha),
            "multiple_comparison_correction": str(args.multiple_comparison_correction),
        },
    }


def load_protocol_manifest(path: str) -> Dict[str, object]:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if str(data.get("schema_version", "")) != PROTOCOL_SCHEMA_VERSION:
        raise ValueError(f"Unsupported protocol schema: {data.get('schema_version')}")
    return data


def apply_multiple_comparison_correction(
    rows: List[Dict[str, object]],
    mode_id: str,
    method: str,
) -> None:
    target = [r for r in rows if str(r.get("mode_id")) == mode_id and (not bool(r.get("is_baseline", False)))]
    pvals = np.array([float(r.get("significance_pvalue", np.nan)) for r in target], dtype=float)
    finite_idx = np.where(np.isfinite(pvals))[0]
    if finite_idx.size == 0:
        return
    m = finite_idx.size
    corrected = np.full_like(pvals, np.nan, dtype=float)
    if method == "none":
        corrected[finite_idx] = pvals[finite_idx]
    elif method == "bonferroni":
        corrected[finite_idx] = np.minimum(1.0, pvals[finite_idx] * m)
    else:  # holm
        order = finite_idx[np.argsort(pvals[finite_idx])]
        running_max = 0.0
        for rank, idx in enumerate(order):
            adj = (m - rank) * pvals[idx]
            running_max = max(running_max, adj)
            corrected[idx] = min(1.0, running_max)
    for i, row in enumerate(target):
        row["significance_pvalue_corrected"] = float(corrected[i]) if np.isfinite(corrected[i]) else np.nan


def validate_row_metrics(row: Dict[str, object]) -> None:
    numeric_keys = [
        "survival_rate",
        "crash_rate",
        "score_composite",
        "worst_pitch_deg",
        "worst_roll_deg",
        "mean_control_energy",
        "mean_sat_rate_du",
        "mean_sat_rate_abs",
    ]
    for k in numeric_keys:
        v = float(row.get(k, np.nan))
        if not np.isfinite(v):
            row["failure_reason"] = str(row.get("failure_reason", "")) + ("+" if row.get("failure_reason") else "") + "metric_nonfinite"
            row["accepted_gate"] = False


def get_mode_matrix(compare_modes: str):
    if compare_modes == "default-vs-low-spin-robust":
        return [
            ("mode_default", "default", "default"),
            ("mode_low_spin_robust", "default", "low-spin-robust"),
        ]
    if compare_modes == "stable-demo-vs-low-spin-robust":
        return [
            ("mode_stable_demo", "stable-demo", "default"),
            ("mode_low_spin_robust", "default", "low-spin-robust"),
        ]
    return [
        ("mode_default_default", "default", "default"),
        ("mode_default_low_spin_robust", "default", "low-spin-robust"),
        ("mode_stable_demo_default", "stable-demo", "default"),
        ("mode_stable_demo_low_spin_robust", "stable-demo", "low-spin-robust"),
    ]


def log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def sample_candidate(rng: np.random.Generator) -> CandidateParams:
    return CandidateParams(
        r_du_rw=log_uniform(rng, *R_DU_RW_BOUNDS),
        r_du_bx=log_uniform(rng, *R_DU_BOUNDS),
        r_du_by=log_uniform(rng, *R_DU_BOUNDS),
        q_ang_scale=float(rng.uniform(*Q_SCALE_BOUNDS)),
        q_rate_scale=float(rng.uniform(*Q_SCALE_BOUNDS)),
        q_rw_scale=float(rng.uniform(*Q_SCALE_BOUNDS)),
        q_base_scale=float(rng.uniform(*Q_SCALE_BOUNDS)),
        q_vel_scale=float(rng.uniform(*Q_SCALE_BOUNDS)),
        qu_scale=float(rng.uniform(*QU_SCALE_BOUNDS)),
        ki_base=float(rng.uniform(*KI_BASE_BOUNDS)),
        u_bleed=float(rng.uniform(*U_BLEED_BOUNDS)),
        max_du_rw=float(rng.uniform(*MAX_DU_RW_BOUNDS)),
        max_du_bx=float(rng.uniform(*MAX_DU_BASE_BOUNDS)),
        max_du_by=float(rng.uniform(*MAX_DU_BASE_BOUNDS)),
    )


def _to_unit(params: CandidateParams) -> np.ndarray:
    # Log-space channels are mapped in log-domain to preserve effective distance.
    lrrw = (np.log(params.r_du_rw) - np.log(R_DU_RW_BOUNDS[0])) / (
        np.log(R_DU_RW_BOUNDS[1]) - np.log(R_DU_RW_BOUNDS[0])
    )
    lrbx = (np.log(params.r_du_bx) - np.log(R_DU_BOUNDS[0])) / (np.log(R_DU_BOUNDS[1]) - np.log(R_DU_BOUNDS[0]))
    lrby = (np.log(params.r_du_by) - np.log(R_DU_BOUNDS[0])) / (np.log(R_DU_BOUNDS[1]) - np.log(R_DU_BOUNDS[0]))
    qang = (params.q_ang_scale - Q_SCALE_BOUNDS[0]) / (Q_SCALE_BOUNDS[1] - Q_SCALE_BOUNDS[0])
    qrate = (params.q_rate_scale - Q_SCALE_BOUNDS[0]) / (Q_SCALE_BOUNDS[1] - Q_SCALE_BOUNDS[0])
    qrw = (params.q_rw_scale - Q_SCALE_BOUNDS[0]) / (Q_SCALE_BOUNDS[1] - Q_SCALE_BOUNDS[0])
    qbase = (params.q_base_scale - Q_SCALE_BOUNDS[0]) / (Q_SCALE_BOUNDS[1] - Q_SCALE_BOUNDS[0])
    qvel = (params.q_vel_scale - Q_SCALE_BOUNDS[0]) / (Q_SCALE_BOUNDS[1] - Q_SCALE_BOUNDS[0])
    qus = (params.qu_scale - QU_SCALE_BOUNDS[0]) / (QU_SCALE_BOUNDS[1] - QU_SCALE_BOUNDS[0])
    ki = (params.ki_base - KI_BASE_BOUNDS[0]) / (KI_BASE_BOUNDS[1] - KI_BASE_BOUNDS[0])
    ub = (params.u_bleed - U_BLEED_BOUNDS[0]) / (U_BLEED_BOUNDS[1] - U_BLEED_BOUNDS[0])
    mdrw = (params.max_du_rw - MAX_DU_RW_BOUNDS[0]) / (MAX_DU_RW_BOUNDS[1] - MAX_DU_RW_BOUNDS[0])
    mdx = (params.max_du_bx - MAX_DU_BASE_BOUNDS[0]) / (MAX_DU_BASE_BOUNDS[1] - MAX_DU_BASE_BOUNDS[0])
    mdy = (params.max_du_by - MAX_DU_BASE_BOUNDS[0]) / (MAX_DU_BASE_BOUNDS[1] - MAX_DU_BASE_BOUNDS[0])
    return np.array([lrrw, lrbx, lrby, qang, qrate, qrw, qbase, qvel, qus, ki, ub, mdrw, mdx, mdy], dtype=float)


def _is_novel(
    cand: CandidateParams,
    baseline_vec: np.ndarray,
    seen_vecs: List[np.ndarray],
    min_distance: float,
) -> bool:
    v = _to_unit(cand)
    if float(np.linalg.norm(v - baseline_vec)) < min_distance:
        return False
    for s in seen_vecs:
        if float(np.linalg.norm(v - s)) < min_distance:
            return False
    return True


def sample_candidate_novel(
    rng: np.random.Generator,
    baseline_params: CandidateParams,
    seen_vecs: List[np.ndarray],
    min_distance: float = NOVELTY_MIN_DISTANCE,
    max_attempts: int = NOVELTY_MAX_ATTEMPTS,
) -> Tuple[CandidateParams, np.ndarray, bool]:
    baseline_vec = _to_unit(baseline_params)
    last = sample_candidate(rng)
    for _ in range(max_attempts):
        cand = sample_candidate(rng)
        if _is_novel(cand, baseline_vec, seen_vecs, min_distance):
            return cand, _to_unit(cand), True
        last = cand
    # Fallback: return last sample if novelty budget exhausted.
    return last, _to_unit(last), False


def baseline_candidate() -> CandidateParams:
    # Aligned with current final.py smooth profile.
    return CandidateParams(
        r_du_rw=0.1378734731394442,
        r_du_bx=1.9942819169979045,
        r_du_by=5.076875253951513,
        q_ang_scale=1.8881272496398052,
        q_rate_scale=4.365896739242128,
        q_rw_scale=4.67293811764239,
        q_base_scale=6.5,
        q_vel_scale=2.4172709641758128,
        qu_scale=3.867780939456718,
        ki_base=0.22991644116687834,
        u_bleed=0.9310825140555963,
        max_du_rw=18.210732648355684,
        max_du_bx=5.886188088635976,
        max_du_by=15.0,
    )


def make_row(
    run_id: str,
    mode_id: str,
    controller_family: str,
    model_variant_id: str,
    domain_profile_id: str,
    is_baseline: bool,
    seed: int,
    episodes: int,
    steps_per_episode: int,
    preset: str,
    stability_profile: str,
    params: CandidateParams,
    metrics: Dict[str, float],
) -> Dict[str, object]:
    metrics_copy = dict(metrics)
    episode_scores = metrics_copy.pop("episode_score_list", [])
    return {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "run_id": run_id,
        "mode_id": mode_id,
        "controller_family": controller_family,
        "model_variant_id": model_variant_id,
        "domain_profile_id": domain_profile_id,
        "is_baseline": is_baseline,
        "seed": seed,
        "episodes": episodes,
        "steps_per_episode": steps_per_episode,
        "preset": preset,
        "stability_profile": stability_profile,
        "hardware_realistic": bool(metrics.get("hardware_realistic", True)),
        "control_hz": float(metrics.get("control_hz", np.nan)),
        "control_delay_steps": int(metrics.get("control_delay_steps", 0)),
        "R_du_rw": params.r_du_rw,
        "R_du_bx": params.r_du_bx,
        "R_du_by": params.r_du_by,
        "Q_ang_scale": params.q_ang_scale,
        "Q_rate_scale": params.q_rate_scale,
        "Q_rw_scale": params.q_rw_scale,
        "Q_base_scale": params.q_base_scale,
        "Q_vel_scale": params.q_vel_scale,
        "QU_scale": params.qu_scale,
        "KI_BASE": params.ki_base,
        "U_BLEED": params.u_bleed,
        "MAX_DU_rw": params.max_du_rw,
        "MAX_DU_bx": params.max_du_bx,
        "MAX_DU_by": params.max_du_by,
        **metrics_copy,
        "delta_survival_rate": np.nan,
        "delta_score_composite": np.nan,
        "delta_worst_base_norm": np.nan,
        "delta_worst_tilt": np.nan,
        "delta_mean_control_energy": np.nan,
        "delta_mean_command_jerk": np.nan,
        "delta_mean_motion_activity": np.nan,
        "delta_mean_ctrl_sign_flip_rate": np.nan,
        "delta_mean_osc_band_energy": np.nan,
        "significance_pvalue": np.nan,
        "significance_pvalue_corrected": np.nan,
        "significance_ci_low": np.nan,
        "significance_ci_high": np.nan,
        "promotion_pass": False,
        "release_verdict": False,
        "_episode_scores": episode_scores,
    }


def add_deltas(rows: List[Dict[str, object]], baseline: Dict[str, object]):
    base_survival = float(baseline["survival_rate"])
    base_score = float(baseline.get("score_composite", np.nan))
    base_worst_base = max(float(baseline["worst_base_x_m"]), float(baseline["worst_base_y_m"]))
    base_worst_tilt = max(float(baseline["worst_pitch_deg"]), float(baseline["worst_roll_deg"]))
    base_energy = float(baseline["mean_control_energy"])
    base_jerk = float(baseline["mean_command_jerk"])
    base_activity = float(baseline["mean_motion_activity"])
    base_flip_rate = float(baseline["mean_ctrl_sign_flip_rate"])
    base_osc_band = float(baseline["mean_osc_band_energy"])

    for row in rows:
        row["delta_survival_rate"] = float(row["survival_rate"]) - base_survival
        row["delta_score_composite"] = float(row.get("score_composite", np.nan)) - base_score
        row["delta_worst_base_norm"] = max(float(row["worst_base_x_m"]), float(row["worst_base_y_m"])) - base_worst_base
        row["delta_worst_tilt"] = max(float(row["worst_pitch_deg"]), float(row["worst_roll_deg"])) - base_worst_tilt
        row["delta_mean_control_energy"] = float(row["mean_control_energy"]) - base_energy
        row["delta_mean_command_jerk"] = float(row["mean_command_jerk"]) - base_jerk
        row["delta_mean_motion_activity"] = float(row["mean_motion_activity"]) - base_activity
        row["delta_mean_ctrl_sign_flip_rate"] = float(row["mean_ctrl_sign_flip_rate"]) - base_flip_rate
        row["delta_mean_osc_band_energy"] = float(row["mean_osc_band_energy"]) - base_osc_band


def sort_key(row: Dict[str, object], primary_objective: str = "crash-rate"):
    survival = float(row["survival_rate"])
    full_survival = bool(survival >= 1.0 - 1e-12)
    crash_rate = float(row.get("crash_rate", 1.0 - survival))
    worst_base = max(float(row["worst_base_x_m"]), float(row["worst_base_y_m"]))
    worst_tilt = max(float(row["worst_pitch_deg"]), float(row["worst_roll_deg"]))
    motion_activity = float(row["mean_motion_activity"])
    command_jerk = float(row["mean_command_jerk"])
    sign_flip_rate = float(row["mean_ctrl_sign_flip_rate"])
    osc_band_energy = float(row["mean_osc_band_energy"])
    sat_rate_du = float(row["mean_sat_rate_du"])
    sat_rate_abs = float(row["mean_sat_rate_abs"])
    energy = float(row["mean_control_energy"])
    failure = str(row.get("failure_reason", ""))
    wheel_over_budget = float(row.get("wheel_over_budget_mean", 0.0))
    wheel_over_hard = float(row.get("wheel_over_hard_mean", 0.0))
    score_composite = float(row.get("score_composite", np.nan))
    objective = worst_base + command_jerk
    failure_penalty = 1 if failure else 0
    if primary_objective == "crash-rate":
        return (
            -survival,
            crash_rate,
            -score_composite,
            worst_tilt,
            wheel_over_hard,
            wheel_over_budget,
            worst_base,
            command_jerk,
            sat_rate_du,
            sat_rate_abs,
            energy,
            failure_penalty,
        )
    if primary_objective == "wheel-spin":
        return (
            wheel_over_hard,
            wheel_over_budget,
            -survival,
            -score_composite,
            crash_rate,
            worst_tilt,
            worst_base,
            command_jerk,
            sat_rate_du,
            sat_rate_abs,
            energy,
            failure_penalty,
        )
    return (
        -int(full_survival),
        -score_composite,
        objective,
        worst_base,
        command_jerk,
        worst_tilt,
        sat_rate_du,
        sat_rate_abs,
        energy,
        osc_band_energy,
        sign_flip_rate,
        motion_activity,
        failure_penalty,
        -survival,
    )


def paired_bootstrap_significance(
    base_scores: List[float],
    cand_scores: List[float],
    rng: np.random.Generator,
    n_boot: int = 2000,
) -> Tuple[float, float, float, float]:
    base = np.asarray(base_scores, dtype=float)
    cand = np.asarray(cand_scores, dtype=float)
    n = min(base.size, cand.size)
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan
    d = cand[:n] - base[:n]
    obs = float(np.mean(d))
    idx = rng.integers(0, n, size=(n_boot, n))
    means = d[idx].mean(axis=1)
    ci_low = float(np.quantile(means, 0.025))
    ci_high = float(np.quantile(means, 0.975))
    p_lo = float(np.mean(means <= 0.0))
    p_hi = float(np.mean(means >= 0.0))
    p_value = float(2.0 * min(p_lo, p_hi))
    return obs, p_value, ci_low, ci_high


def write_csv(path: Path, rows: List[Dict[str, object]]):
    if not rows:
        return
    clean_rows = [{k: v for k, v in row.items() if not str(k).startswith("_")} for row in rows]
    fieldnames = list(clean_rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(clean_rows)


def write_summary(path: Path, rows: List[Dict[str, object]], primary_objective: str):
    lines = []
    lines.append(f"OBJECTIVE={primary_objective}")
    lines.append("")

    keys = sorted(
        {
            (
                str(r["mode_id"]),
                str(r.get("controller_family", "current")),
                str(r.get("model_variant_id", "nominal")),
                str(r.get("domain_profile_id", "default")),
            )
            for r in rows
        }
    )
    baseline_by_key = {}
    for mode, family, variant, domain in keys:
        base_rows = [
            r
            for r in rows
            if str(r["mode_id"]) == mode
            and str(r.get("controller_family", "current")) == family
            and str(r.get("model_variant_id", "nominal")) == variant
            and str(r.get("domain_profile_id", "default")) == domain
            and bool(r["is_baseline"])
        ]
        if base_rows:
            baseline_by_key[(mode, family, variant, domain)] = base_rows[0]
    lines.append("BASELINE BY MODE/FAMILY")
    for mode, family, variant, domain in keys:
        b = baseline_by_key.get((mode, family, variant, domain))
        if b is None:
            continue
        lines.append(
            f"{mode}/{family}/{variant}/{domain}: preset={b.get('preset')} stability_profile={b.get('stability_profile')} "
            f"survival={float(b['survival_rate']):.3f} crash_rate={float(b.get('crash_rate', np.nan)):.3f} "
            f"score={float(b.get('score_composite', np.nan)):.3f} "
            f"wheel_over_budget={float(b.get('wheel_over_budget_mean', 0.0)):.3f} "
            f"wheel_over_hard={float(b.get('wheel_over_hard_mean', 0.0)):.3f}"
        )
    lines.append("")

    lines.append("TOP ACCEPTED PER MODE/FAMILY")
    for mode, family, variant, domain in keys:
        mode_rows = [
            r
            for r in rows
            if str(r["mode_id"]) == mode
            and str(r.get("controller_family", "current")) == family
            and str(r.get("model_variant_id", "nominal")) == variant
            and str(r.get("domain_profile_id", "default")) == domain
            and not bool(r["is_baseline"])
        ]
        accepted = [r for r in mode_rows if bool(r["accepted_gate"])]
        if accepted:
            top = sorted(accepted, key=lambda r: sort_key(r, primary_objective))[0]
            lines.append(
                f"{mode}/{family}/{variant}/{domain}: {top['run_id']} survival={float(top['survival_rate']):.3f} "
                f"crash_rate={float(top.get('crash_rate', np.nan)):.3f} "
                f"score={float(top.get('score_composite', np.nan)):.3f} "
                f"worst_tilt={max(float(top['worst_pitch_deg']), float(top['worst_roll_deg'])):.3f}deg "
                f"wheel_over_budget={float(top.get('wheel_over_budget_mean', 0.0)):.3f} "
                f"wheel_over_hard={float(top.get('wheel_over_hard_mean', 0.0)):.3f} "
                f"p={float(top.get('significance_pvalue', np.nan)):.4f} "
                f"p_corr={float(top.get('significance_pvalue_corrected', np.nan)):.4f} "
                f"promotion_pass={bool(top.get('promotion_pass', False))} "
                f"release_verdict={bool(top.get('release_verdict', False))} "
                f"tier={str(top.get('confidence_tier', 'exploratory'))}"
            )
        else:
            fallback = sorted(mode_rows, key=lambda r: sort_key(r, primary_objective))[0] if mode_rows else None
            if fallback is None:
                lines.append(f"{mode}/{family}/{variant}/{domain}: no rows")
            else:
                lines.append(
                    f"{mode}/{family}/{variant}/{domain}: no accepted candidate, fallback={fallback['run_id']} "
                    f"survival={float(fallback['survival_rate']):.3f} "
                    f"reason={fallback.get('failure_reason', '')}"
                )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def maybe_plot(path: Path, rows: List[Dict[str, object]], baseline: Dict[str, object]):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warn] plot skipped: {exc}")
        return

    x = np.array([float(r["mean_control_energy"]) for r in rows])
    y = np.array([max(float(r["worst_pitch_deg"]), float(r["worst_roll_deg"])) for r in rows])
    c = np.array([float(r["survival_rate"]) for r in rows])
    accepted = np.array([bool(r["accepted_gate"]) for r in rows])

    fig, ax = plt.subplots(figsize=(9, 6))
    s1 = ax.scatter(x[~accepted], y[~accepted], c=c[~accepted], cmap="viridis", marker="x", alpha=0.7, label="Rejected")
    ax.scatter(x[accepted], y[accepted], c=c[accepted], cmap="viridis", marker="o", alpha=0.9, label="Accepted")
    fig.colorbar(s1, ax=ax, label="Survival rate")

    bx = float(baseline["mean_control_energy"])
    by = max(float(baseline["worst_pitch_deg"]), float(baseline["worst_roll_deg"]))
    ax.scatter([bx], [by], color="red", marker="*", s=220, edgecolor="black", label="Baseline")

    accepted_rows = [r for r in rows if float(r["survival_rate"]) >= 1.0 - 1e-12]
    top5 = sorted(accepted_rows, key=sort_key)[:5]
    for r in top5:
        tx = float(r["mean_control_energy"])
        ty = max(float(r["worst_pitch_deg"]), float(r["worst_roll_deg"]))
        ax.annotate(str(r["run_id"]), (tx, ty), textcoords="offset points", xytext=(4, 4), fontsize=8)

    ax.set_title("Pareto View: Stability vs Control Effort")
    ax.set_xlabel("Mean control energy")
    ax.set_ylabel("Worst-case tilt (deg)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def print_report(rows: List[Dict[str, object]], primary_objective: str):
    print(f"\n=== BENCHMARK REPORT ({primary_objective}) ===")
    keys = sorted(
        {
            (
                str(r["mode_id"]),
                str(r.get("controller_family", "current")),
                str(r.get("model_variant_id", "nominal")),
                str(r.get("domain_profile_id", "default")),
            )
            for r in rows
        }
    )
    for mode, family, variant, domain in keys:
        baseline_rows = [
            r
            for r in rows
            if str(r["mode_id"]) == mode
            and str(r.get("controller_family", "current")) == family
            and str(r.get("model_variant_id", "nominal")) == variant
            and str(r.get("domain_profile_id", "default")) == domain
            and bool(r["is_baseline"])
        ]
        if baseline_rows:
            b = baseline_rows[0]
            print(
                f"{mode}/{family}/{variant}/{domain} baseline: preset={b.get('preset')} stability={b.get('stability_profile')} "
                f"survival={float(b['survival_rate']):.3f} crash_rate={float(b.get('crash_rate', np.nan)):.3f} "
                f"score={float(b.get('score_composite', np.nan)):.3f} "
                f"wheel_budget={float(b.get('wheel_over_budget_mean', 0.0)):.3f} "
                f"wheel_hard={float(b.get('wheel_over_hard_mean', 0.0)):.3f}"
            )
        candidates = [
            r
            for r in rows
            if str(r["mode_id"]) == mode
            and str(r.get("controller_family", "current")) == family
            and str(r.get("model_variant_id", "nominal")) == variant
            and str(r.get("domain_profile_id", "default")) == domain
            and not bool(r["is_baseline"])
        ]
        accepted = [r for r in candidates if bool(r["accepted_gate"])]
        if accepted:
            top = sorted(accepted, key=lambda r: sort_key(r, primary_objective))[0]
            print(
                f"{mode}/{family}/{variant}/{domain} top accepted: {top['run_id']} survival={float(top['survival_rate']):.3f} "
                f"crash_rate={float(top.get('crash_rate', np.nan)):.3f} "
                f"score={float(top.get('score_composite', np.nan)):.3f} "
                f"worst_tilt={max(float(top['worst_pitch_deg']), float(top['worst_roll_deg'])):.3f}deg "
                f"p={float(top.get('significance_pvalue', np.nan)):.4f} "
                f"p_corr={float(top.get('significance_pvalue_corrected', np.nan)):.4f} "
                f"pass={bool(top.get('promotion_pass', False))} "
                f"release={bool(top.get('release_verdict', False))} "
                f"tier={str(top.get('confidence_tier', 'exploratory'))}"
            )
        elif candidates:
            fb = sorted(candidates, key=lambda r: sort_key(r, primary_objective))[0]
            print(
                f"{mode}/{family}/{variant}/{domain} fallback: {fb['run_id']} survival={float(fb['survival_rate']):.3f} "
                f"reason={fb.get('failure_reason', '')}"
            )


def main():
    args = parse_args()
    if args.release_campaign:
        args.benchmark_profile = "nightly_long"
        if args.trials is None:
            args.trials = 260
        if args.episodes is None:
            args.episodes = 56
        if args.steps is None:
            args.steps = 7000
    episodes, trials, steps = apply_stress_defaults(args)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_matrix = get_mode_matrix(args.compare_modes)
    controller_families = parse_controller_families(args.controller_families)
    model_variants = parse_csv_list(args.model_variants)
    domain_profiles = parse_csv_list(args.domain_rand_profile)
    if args.protocol_manifest:
        proto = load_protocol_manifest(args.protocol_manifest)
        controller_families = [str(v) for v in proto.get("controller_families", controller_families)]
        model_variants = [str(v) for v in proto.get("model_variants", model_variants)]
        domain_profiles = [str(v) for v in proto.get("domain_profiles", domain_profiles)]
        mode_matrix = [tuple(v) for v in proto.get("mode_matrix", [list(m) for m in mode_matrix])]
    model_path = Path(__file__).with_name("final.xml")
    try:
        evaluator = ControllerEvaluator(model_path)
    except Exception as exc:
        raise SystemExit(f"Failed to load/evaluate model: {exc}") from exc

    rng = np.random.default_rng(args.seed)
    episode_seeds = [int(s) for s in rng.integers(0, 2**31 - 1, size=episodes)]
    rows: List[Dict[str, object]] = []
    seen_candidate_vecs: List[np.ndarray] = []
    novelty_rejects = 0
    print(
        f"Starting benchmark: episodes={episodes}, trials={trials}, steps={steps}, seed={args.seed}",
        flush=True,
    )
    print(
        (
            "Hardware model: "
            f"enabled={not args.legacy_model}, "
            f"control_hz={args.control_hz:.1f}, "
            f"delay_steps={args.control_delay_steps}, "
            f"wheel_ticks={args.wheel_encoder_ticks}"
        ),
        flush=True,
    )
    print(
        f"Compare modes={args.compare_modes}, objective={args.primary_objective}, stress={args.stress_level}, profile={args.benchmark_profile}",
        flush=True,
    )
    print(f"Controller families={','.join(controller_families)}", flush=True)
    print(f"Model variants={','.join(model_variants)} domain_profiles={','.join(domain_profiles)}", flush=True)
    t0 = time.perf_counter()

    base_params = baseline_candidate()
    candidate_specs = [("baseline", True, base_params)]
    for i in range(trials):
        params, cand_vec, is_novel = sample_candidate_novel(
            rng,
            base_params,
            seen_candidate_vecs,
        )
        if not is_novel:
            novelty_rejects += 1
        seen_candidate_vecs.append(cand_vec)
        candidate_specs.append((f"trial_{i:03d}", False, params))

    total_evals = len(candidate_specs) * len(mode_matrix) * len(controller_families) * len(model_variants) * len(domain_profiles)
    done_evals = 0
    for mode_id, preset, stability_profile in mode_matrix:
        for model_variant in model_variants:
            for domain_profile in domain_profiles:
                for controller_family in controller_families:
                    episode_config = EpisodeConfig(
                        steps=steps,
                        disturbance_magnitude_xy=args.disturbance_xy,
                        disturbance_magnitude_z=args.disturbance_z,
                        disturbance_interval=args.disturbance_interval,
                        init_angle_deg=args.init_angle_deg,
                        init_base_pos_m=args.init_base_pos_m,
                        max_worst_tilt_deg=args.gate_max_worst_tilt,
                        max_worst_base_m=args.gate_max_worst_base,
                        max_mean_sat_rate_du=args.gate_max_sat_du,
                        max_mean_sat_rate_abs=args.gate_max_sat_abs,
                        control_hz=args.control_hz,
                        control_delay_steps=args.control_delay_steps,
                        hardware_realistic=not args.legacy_model,
                        imu_angle_noise_std_rad=float(np.radians(args.imu_angle_noise_deg)),
                        imu_rate_noise_std_rad_s=args.imu_rate_noise,
                        wheel_encoder_ticks_per_rev=args.wheel_encoder_ticks,
                        wheel_encoder_rate_noise_std_rad_s=args.wheel_rate_noise,
                        preset=preset,
                        stability_profile=stability_profile,
                        controller_family=controller_family,
                        model_variant_id=model_variant,
                        domain_profile_id=domain_profile,
                        hardware_replay=bool(args.hardware_trace_path),
                        hardware_trace_path=args.hardware_trace_path,
                    )
                    print(
                        f"[mode] {mode_id}/{controller_family}/{model_variant}/{domain_profile}: preset={preset} stability_profile={stability_profile}",
                        flush=True,
                    )
                    for run_id, is_baseline, params in candidate_specs:
                        metrics = safe_evaluate_candidate(evaluator, params, episode_seeds, episode_config)
                        row = make_row(
                            run_id=run_id,
                            mode_id=mode_id,
                            controller_family=controller_family,
                            model_variant_id=model_variant,
                            domain_profile_id=domain_profile,
                            is_baseline=is_baseline,
                            seed=args.seed,
                            episodes=episodes,
                            steps_per_episode=episode_config.steps,
                            preset=preset,
                            stability_profile=stability_profile,
                            params=params,
                            metrics=metrics,
                        )
                        validate_row_metrics(row)
                        rows.append(row)
                        done_evals += 1
                        elapsed = time.perf_counter() - t0
                        per_eval = elapsed / max(done_evals, 1)
                        remaining = max(total_evals - done_evals, 0) * per_eval
                        print(
                            f"\rProgress: {done_evals}/{total_evals} evals | elapsed {elapsed:6.1f}s | ETA {remaining:6.1f}s",
                            end="",
                            flush=True,
                        )
    print("", flush=True)

    # Delta metrics use each mode/family baseline.
    keys = sorted(
        {
            (
                str(r["mode_id"]),
                str(r.get("controller_family", "current")),
                str(r.get("model_variant_id", "nominal")),
                str(r.get("domain_profile_id", "default")),
            )
            for r in rows
        }
    )
    for mode_id, controller_family, model_variant_id, domain_profile_id in keys:
        key_rows = [
            r
            for r in rows
            if str(r["mode_id"]) == mode_id
            and str(r.get("controller_family", "current")) == controller_family
            and str(r.get("model_variant_id", "nominal")) == model_variant_id
            and str(r.get("domain_profile_id", "default")) == domain_profile_id
        ]
        baseline_rows = [r for r in key_rows if bool(r["is_baseline"])]
        if baseline_rows:
            add_deltas(key_rows, baseline_rows[0])

    # Significance + promotion are measured against the current-family baseline in the same mode.
    sig_rng = np.random.default_rng(args.seed + 7919)
    group_keys = sorted(
        {
            (
                str(r["mode_id"]),
                str(r.get("model_variant_id", "nominal")),
                str(r.get("domain_profile_id", "default")),
            )
            for r in rows
        }
    )
    for mode_id, model_variant_id, domain_profile_id in group_keys:
        baseline_by_family = {}
        for fam in ("current", "paper_split_baseline", "baseline_mpc", "baseline_robust_hinf_like"):
            baseline_by_family[fam] = next(
                (
                    r
                    for r in rows
                    if str(r["mode_id"]) == mode_id
                    and str(r.get("model_variant_id", "nominal")) == model_variant_id
                    and str(r.get("domain_profile_id", "default")) == domain_profile_id
                    and str(r.get("controller_family", "")) == fam
                    and bool(r["is_baseline"])
                ),
                None,
            )
        baseline_current = baseline_by_family.get("current")
        if baseline_current is None:
            continue
        base_scores = [float(v) for v in baseline_current.get("_episode_scores", [])]
        base_survival = float(baseline_current.get("survival_rate", 0.0))
        base_worst_tilt = max(float(baseline_current.get("worst_pitch_deg", np.inf)), float(baseline_current.get("worst_roll_deg", np.inf)))
        candidate_rows = [
            r
            for r in rows
            if str(r["mode_id"]) == mode_id
            and str(r.get("model_variant_id", "nominal")) == model_variant_id
            and str(r.get("domain_profile_id", "default")) == domain_profile_id
            and (not bool(r["is_baseline"]))
        ]
        for row in candidate_rows:
            cand_scores = [float(v) for v in row.get("_episode_scores", [])]
            _, p_value, ci_low, ci_high = paired_bootstrap_significance(base_scores, cand_scores, sig_rng)
            row["significance_pvalue"] = p_value
            row["significance_ci_low"] = ci_low
            row["significance_ci_high"] = ci_high
        apply_multiple_comparison_correction(candidate_rows, mode_id, args.multiple_comparison_correction)
        for row in candidate_rows:
            no_safety_regress = (
                float(row.get("survival_rate", 0.0)) >= base_survival
                and max(float(row.get("worst_pitch_deg", np.inf)), float(row.get("worst_roll_deg", np.inf))) <= base_worst_tilt
            )
            score_improves = bool(float(row.get("score_composite", -np.inf)) > float(baseline_current.get("score_composite", np.inf)))
            p_corr = float(row.get("significance_pvalue_corrected", np.nan))
            signif = bool(np.isfinite(p_corr) and p_corr < float(args.significance_alpha) and np.isfinite(float(row.get("significance_ci_low", np.nan))) and float(row.get("significance_ci_low", np.nan)) > 0.0)
            beats_required = True
            for fam in ("paper_split_baseline", "baseline_mpc", "baseline_robust_hinf_like"):
                b = baseline_by_family.get(fam)
                if b is None:
                    continue
                if float(row.get("score_composite", -np.inf)) <= float(b.get("score_composite", np.inf)):
                    beats_required = False
            row["promotion_pass"] = bool(
                str(row.get("controller_family", "")) == "hybrid_modern"
                and bool(row.get("accepted_gate", False))
                and no_safety_regress
                and score_improves
                and signif
                and beats_required
            )
            row["release_verdict"] = bool(row["promotion_pass"] and str(row.get("confidence_tier", "")) in ("strong", "best_in_class_candidate"))

    # Domain robustness stratification.
    for row in rows:
        family_rows = [
            r
            for r in rows
            if str(r.get("controller_family")) == str(row.get("controller_family"))
            and str(r.get("mode_id")) == str(row.get("mode_id"))
            and str(r.get("model_variant_id")) == str(row.get("model_variant_id"))
        ]
        score_vals = np.asarray([float(r.get("score_composite", np.nan)) for r in family_rows], dtype=float)
        finite = score_vals[np.isfinite(score_vals)]
        row["domain_score_p5"] = float(np.quantile(finite, 0.05)) if finite.size else np.nan
        row["domain_score_p1"] = float(np.quantile(finite, 0.01)) if finite.size else np.nan
    ranked = sorted(rows, key=lambda r: sort_key(r, args.primary_objective))

    print_report(ranked, args.primary_objective)
    print(f"\nNovelty rejects (resample budget exhausted): {novelty_rejects}")

    csv_path = outdir / f"benchmark_{ts}.csv"
    plot_path = outdir / f"benchmark_{ts}_pareto.png"
    summary_path = outdir / f"benchmark_{ts}_summary.txt"
    manifest_path = outdir / f"benchmark_{ts}_protocol.json"
    release_path = outdir / f"benchmark_{ts}_release_bundle.json"
    write_csv(csv_path, ranked)
    baseline_for_plot = next((r for r in ranked if bool(r["is_baseline"])), ranked[0])
    maybe_plot(plot_path, ranked, baseline_for_plot)
    write_summary(summary_path, ranked, args.primary_objective)
    protocol_manifest = build_protocol_manifest(
        args=args,
        mode_matrix=mode_matrix,
        controller_families=controller_families,
        model_variants=model_variants,
        domain_profiles=domain_profiles,
    )
    manifest_path.write_text(json.dumps(protocol_manifest, indent=2), encoding="utf-8")
    release_bundle = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "timestamp": ts,
        "release_campaign": bool(args.release_campaign),
        "promotion_pass_count": int(sum(1 for r in ranked if bool(r.get("promotion_pass", False)))),
        "release_verdict_count": int(sum(1 for r in ranked if bool(r.get("release_verdict", False)))),
        "top_rows": [
            {
                "run_id": r.get("run_id"),
                "mode_id": r.get("mode_id"),
                "controller_family": r.get("controller_family"),
                "model_variant_id": r.get("model_variant_id"),
                "domain_profile_id": r.get("domain_profile_id"),
                "score_composite": r.get("score_composite"),
                "survival_rate": r.get("survival_rate"),
                "promotion_pass": r.get("promotion_pass"),
                "release_verdict": r.get("release_verdict"),
                "confidence_tier": r.get("confidence_tier"),
            }
            for r in ranked[:20]
        ],
        "artifacts": {
            "csv": str(csv_path),
            "plot": str(plot_path),
            "summary": str(summary_path),
            "protocol_manifest": str(manifest_path),
        },
    }
    release_path.write_text(json.dumps(release_bundle, indent=2), encoding="utf-8")

    print("\n=== ARTIFACTS ===")
    print(f"CSV: {csv_path}")
    print(f"Plot: {plot_path}")
    print(f"Summary: {summary_path}")
    print(f"Protocol: {manifest_path}")
    print(f"Release bundle: {release_path}")


if __name__ == "__main__":
    main()
