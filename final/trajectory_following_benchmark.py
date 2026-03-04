from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from controller_eval import CandidateParams, ControllerEvaluator, EpisodeConfig


DEFAULT_FAMILIES = "current,current_dob,hybrid_modern,baseline_mpc,paper_split_baseline,baseline_robust_hinf_like"
DEFAULT_YAW_MODES = "off,bearing_pd"


def baseline_candidate() -> CandidateParams:
    return CandidateParams(
        r_du_rw=0.1378734731394442,
        r_du_bx=1.9942819169979045,
        r_du_by=3.0,
        q_ang_scale=1.8881272496398052,
        q_rate_scale=4.365896739242128,
        q_rw_scale=4.67293811764239,
        q_base_scale=5.0,
        q_vel_scale=2.4172709641758128,
        qu_scale=5.0,
        ki_base=0.22991644116687834,
        u_bleed=0.9310825140555963,
        max_du_rw=18.210732648355684,
        max_du_bx=5.886188088635976,
        max_du_by=15.0,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Straight-line trajectory following benchmark (forward/back along X) for controller families.",
    )
    parser.add_argument("--families", type=str, default=DEFAULT_FAMILIES)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--steps", type=int, default=7000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["smooth", "robust"], default="smooth")
    parser.add_argument("--preset", type=str, default="default")
    parser.add_argument("--stability-profile", type=str, default="default")
    parser.add_argument("--control-hz", type=float, default=250.0)
    parser.add_argument("--control-delay-steps", type=int, default=1)
    parser.add_argument("--legacy-model", action="store_true")
    parser.add_argument("--disturbance-xy", type=float, default=1.5)
    parser.add_argument("--disturbance-z", type=float, default=0.4)
    parser.add_argument("--disturbance-interval", type=int, default=300)
    parser.add_argument("--trajectory-profile", choices=["line_sine", "step_x"], default="line_sine")
    parser.add_argument("--trajectory-amp-m", type=float, default=0.22)
    parser.add_argument("--trajectory-step-m", type=float, default=0.18)
    parser.add_argument("--trajectory-period-s", type=float, default=6.0)
    parser.add_argument("--trajectory-warmup-s", type=float, default=1.0)
    parser.add_argument("--yaw-modes", type=str, default=DEFAULT_YAW_MODES)
    parser.add_argument("--yaw-heading-kp", type=float, default=2.2)
    parser.add_argument("--yaw-heading-kd", type=float, default=0.8)
    parser.add_argument("--yaw-lateral-pos-k", type=float, default=0.7)
    parser.add_argument("--yaw-max-force", type=float, default=0.9)
    parser.add_argument("--yaw-min-speed-m-s", type=float, default=0.03)
    parser.add_argument("--outdir", type=str, default="results")
    return parser.parse_args(argv)


def _parse_family_list(raw: str) -> list[str]:
    out: list[str] = []
    for token in str(raw).split(","):
        name = token.strip()
        if name and name not in out:
            out.append(name)
    return out if out else ["current"]


def _parse_mode_list(raw: str) -> list[str]:
    out: list[str] = []
    for token in str(raw).split(","):
        mode = token.strip()
        if mode and mode not in out:
            out.append(mode)
    return out if out else ["off"]


def _nanmean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.nanmean(arr)) if np.any(np.isfinite(arr)) else float("nan")


def _nanmax(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.nanmax(arr)) if np.any(np.isfinite(arr)) else float("nan")


def _score(summary: dict[str, float]) -> float:
    survival = float(summary["survival_rate"])
    rmse = float(summary["tracking_rmse_mean_m"])
    p95 = float(summary["tracking_p95_mean_m"])
    peak = float(summary["tracking_peak_max_m"])
    heading_mae = float(summary.get("heading_err_mae_mean_deg", np.nan))
    energy = float(summary["control_energy_mean"])
    if not np.isfinite(rmse):
        return -1e9
    score = 100.0 * survival
    score -= 240.0 * rmse
    score -= 80.0 * (p95 if np.isfinite(p95) else rmse)
    score -= 55.0 * (peak if np.isfinite(peak) else rmse)
    if np.isfinite(heading_mae):
        score -= 0.80 * heading_mae
    score -= 0.02 * (energy if np.isfinite(energy) else 0.0)
    return float(score)


def run_family(
    evaluator: ControllerEvaluator,
    family: str,
    yaw_mode: str,
    params: CandidateParams,
    episode_seeds: list[int],
    cfg_template: EpisodeConfig,
) -> dict[str, Any]:
    cfg = EpisodeConfig(**cfg_template.__dict__)
    cfg.controller_family = family
    cfg.yaw_control_mode = yaw_mode
    episodes: list[dict[str, float]] = []
    for seed in episode_seeds:
        episodes.append(evaluator.simulate_episode(params=params, episode_seed=seed, config=cfg))

    survival = float(np.mean([ep["survived"] for ep in episodes]))
    crash_rate = 1.0 - survival
    tracking_rmse_mean = _nanmean([float(ep.get("tracking_rmse_m", np.nan)) for ep in episodes])
    tracking_mae_mean = _nanmean([float(ep.get("tracking_mae_m", np.nan)) for ep in episodes])
    tracking_p95_mean = _nanmean([float(ep.get("tracking_p95_m", np.nan)) for ep in episodes])
    tracking_peak_max = _nanmax([float(ep.get("tracking_peak_m", np.nan)) for ep in episodes])
    tracking_x_mae_mean = _nanmean([float(ep.get("tracking_x_mae_m", np.nan)) for ep in episodes])
    tracking_y_mae_mean = _nanmean([float(ep.get("tracking_y_mae_m", np.nan)) for ep in episodes])
    heading_err_mae_mean = _nanmean([float(ep.get("heading_err_mae_deg", np.nan)) for ep in episodes])
    heading_err_max_max = _nanmax([float(ep.get("heading_err_max_deg", np.nan)) for ep in episodes])
    control_energy_mean = _nanmean([float(ep.get("control_energy", np.nan)) for ep in episodes])
    worst_pitch_deg = _nanmax([float(ep.get("max_abs_pitch_deg", np.nan)) for ep in episodes])
    worst_roll_deg = _nanmax([float(ep.get("max_abs_roll_deg", np.nan)) for ep in episodes])
    worst_base_x_m = _nanmax([float(ep.get("max_abs_base_x_m", np.nan)) for ep in episodes])
    worst_base_y_m = _nanmax([float(ep.get("max_abs_base_y_m", np.nan)) for ep in episodes])

    out: dict[str, Any] = {
        "controller_family": family,
        "yaw_mode": yaw_mode,
        "episodes": int(len(episodes)),
        "survival_rate": survival,
        "crash_rate": crash_rate,
        "tracking_rmse_mean_m": tracking_rmse_mean,
        "tracking_mae_mean_m": tracking_mae_mean,
        "tracking_p95_mean_m": tracking_p95_mean,
        "tracking_peak_max_m": tracking_peak_max,
        "tracking_x_mae_mean_m": tracking_x_mae_mean,
        "tracking_y_mae_mean_m": tracking_y_mae_mean,
        "heading_err_mae_mean_deg": heading_err_mae_mean,
        "heading_err_max_max_deg": heading_err_max_max,
        "control_energy_mean": control_energy_mean,
        "worst_pitch_deg": worst_pitch_deg,
        "worst_roll_deg": worst_roll_deg,
        "worst_base_x_m": worst_base_x_m,
        "worst_base_y_m": worst_base_y_m,
    }
    out["trajectory_score"] = _score(out)
    return out


def main(argv=None):
    args = parse_args(argv)
    root = Path(__file__).resolve().parent
    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = root / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    families = _parse_family_list(args.families)
    yaw_modes = _parse_mode_list(args.yaw_modes)
    seeds = [int(v) for v in np.random.default_rng(int(args.seed)).integers(0, 2**31 - 1, size=max(args.episodes, 1))]

    cfg_template = EpisodeConfig(
        steps=int(max(args.steps, 200)),
        disturbance_magnitude_xy=float(max(args.disturbance_xy, 0.0)),
        disturbance_magnitude_z=float(max(args.disturbance_z, 0.0)),
        disturbance_interval=int(max(args.disturbance_interval, 1)),
        init_angle_deg=4.0,
        init_base_pos_m=0.0,
        mode=str(args.mode),
        control_hz=float(max(args.control_hz, 1.0)),
        control_delay_steps=int(max(args.control_delay_steps, 0)),
        hardware_realistic=bool(not args.legacy_model),
        preset=str(args.preset),
        stability_profile=str(args.stability_profile),
        trajectory_profile=str(args.trajectory_profile),
        trajectory_warmup_s=float(max(args.trajectory_warmup_s, 0.0)),
        trajectory_x_step_m=float(args.trajectory_step_m),
        trajectory_x_amp_m=float(max(args.trajectory_amp_m, 0.0)),
        trajectory_period_s=float(max(args.trajectory_period_s, 0.5)),
        trajectory_x_bias_m=0.0,
        trajectory_y_bias_m=0.0,
        yaw_control_mode="off",
        yaw_heading_kp=float(max(args.yaw_heading_kp, 0.0)),
        yaw_heading_kd=float(max(args.yaw_heading_kd, 0.0)),
        yaw_lateral_pos_k=float(max(args.yaw_lateral_pos_k, 0.0)),
        yaw_max_force=float(max(args.yaw_max_force, 0.0)),
        yaw_min_speed_m_s=float(max(args.yaw_min_speed_m_s, 0.0)),
    )

    print(
        "Trajectory benchmark: "
        f"families={','.join(families)} "
        f"yaw_modes={','.join(yaw_modes)} "
        f"episodes={len(seeds)} steps={cfg_template.steps} "
        f"profile={cfg_template.trajectory_profile}",
        flush=True,
    )
    evaluator = ControllerEvaluator(root / "final.xml")
    params = baseline_candidate()
    rows: list[dict[str, Any]] = []
    total_jobs = len(families) * len(yaw_modes)
    job_idx = 0
    for yaw_mode in yaw_modes:
        for family in families:
            job_idx += 1
            print(f"[{job_idx}/{total_jobs}] evaluating family={family} yaw_mode={yaw_mode}", flush=True)
            row = run_family(
                evaluator=evaluator,
                family=family,
                yaw_mode=yaw_mode,
                params=params,
                episode_seeds=seeds,
                cfg_template=cfg_template,
            )
            rows.append(row)
            head_mae = float(row.get("heading_err_mae_mean_deg", np.nan))
            head_mae_str = f"{head_mae:.2f}deg" if np.isfinite(head_mae) else "n/a"
            print(
                "  "
                f"survival={row['survival_rate']:.3f} "
                f"rmse={row['tracking_rmse_mean_m']:.4f}m "
                f"p95={row['tracking_p95_mean_m']:.4f}m "
                f"peak={row['tracking_peak_max_m']:.4f}m "
                f"head_mae={head_mae_str} "
                f"score={row['trajectory_score']:.3f}",
                flush=True,
            )

    ranked = sorted(rows, key=lambda r: float(r["trajectory_score"]), reverse=True)
    print("\n=== TRAJECTORY RANKING ===")
    for rank, row in enumerate(ranked, start=1):
        head_mae = float(row.get("heading_err_mae_mean_deg", np.nan))
        head_mae_str = f"{head_mae:.2f}deg" if np.isfinite(head_mae) else "n/a"
        print(
            f"{rank:2d}. {row['controller_family']:>24s} yaw={str(row['yaw_mode']):<10s} "
            f"score={float(row['trajectory_score']):8.3f} "
            f"survival={float(row['survival_rate']):.3f} "
            f"rmse={float(row['tracking_rmse_mean_m']):.4f}m "
            f"peak={float(row['tracking_peak_max_m']):.4f}m "
            f"head_mae={head_mae_str}"
        )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = outdir / f"trajectory_following_{ts}.csv"
    json_path = outdir / f"trajectory_following_{ts}.json"

    fieldnames = list(ranked[0].keys()) if ranked else ["controller_family"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ranked)

    payload = {
        "timestamp": ts,
        "args": vars(args),
        "episode_seeds": seeds,
        "ranking": ranked,
        "winner": ranked[0] if ranked else None,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nArtifacts:\n- {csv_path}\n- {json_path}")


if __name__ == "__main__":
    main()
