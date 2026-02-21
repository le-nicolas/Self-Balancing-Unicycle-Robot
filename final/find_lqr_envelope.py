import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Dict, List

import numpy as np

from controller_eval import CandidateParams, ControllerEvaluator, EpisodeConfig


def baseline_candidate() -> CandidateParams:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Binary-search disturbance XY envelope where pitch ratchet rate per disturbance cycle crosses target."
    )
    parser.add_argument(
        "--ablation-json",
        type=str,
        default=str(Path(__file__).with_name("results") / "throughput_ablation_seed42.json"),
        help="Path with episode_seeds list (used when --episode-indices is set).",
    )
    parser.add_argument(
        "--episode-indices",
        type=str,
        default="20",
        help="Comma-separated 0-based episode indices into ablation episode_seeds.",
    )
    parser.add_argument("--seed-fallback", type=int, default=1074497555)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--disturbance-interval", type=int, default=200)
    parser.add_argument("--disturbance-z", type=float, default=2.0)
    parser.add_argument("--controller-family", type=str, default="current")
    parser.add_argument("--preset", type=str, default="default")
    parser.add_argument("--stability-profile", type=str, default="default")
    parser.add_argument("--control-hz", type=float, default=250.0)
    parser.add_argument("--control-delay-steps", type=int, default=1)
    parser.add_argument("--legacy-model", action="store_true")
    parser.add_argument("--xy-min", type=float, default=0.0)
    parser.add_argument("--xy-max", type=float, default=10.0)
    parser.add_argument("--target-ratchet-deg-per-cycle", type=float, default=0.0)
    parser.add_argument(
        "--baseline-subtract-zero-xy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use effective ratchet = ratchet_mean(xy) - ratchet_mean(xy=0).",
    )
    parser.add_argument("--max-iters", type=int, default=8)
    parser.add_argument("--xy-tol", type=float, default=0.10)
    return parser.parse_args()


def resolve_seeds(args: argparse.Namespace) -> tuple[List[int], int]:
    path = Path(args.ablation_json)
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        steps = int(data.get("steps", args.steps))
        seeds = [int(v) for v in data.get("episode_seeds", [])]
    else:
        steps = int(args.steps)
        seeds = []
    idxs = [int(x.strip()) for x in str(args.episode_indices).split(",") if x.strip()]
    if not seeds or not idxs:
        return [int(args.seed_fallback)], steps
    out: List[int] = []
    for idx in idxs:
        if idx < 0 or idx >= len(seeds):
            raise ValueError(f"episode index {idx} out of range for {path} (n={len(seeds)}).")
        out.append(int(seeds[idx]))
    return out, steps


def _slope_deg_per_s(step_vals: Dict[int, float], end_step: int, window: int, dt: float) -> float:
    start = max(1, end_step - window + 1)
    xs = []
    ys = []
    for s in range(start, end_step + 1):
        if s in step_vals:
            xs.append(float(s))
            ys.append(float(step_vals[s]))
    if len(xs) < 2:
        return float("nan")
    m = float(np.polyfit(np.asarray(xs), np.asarray(ys), 1)[0])  # deg / step
    return float(m / dt)  # deg / s


def episode_metrics(
    evaluator: ControllerEvaluator,
    params: CandidateParams,
    cfg: EpisodeConfig,
    seed: int,
) -> dict[str, float]:
    result = evaluator.simulate_episode(
        params=params,
        episode_seed=int(seed),
        config=cfg,
        collect_disturbance_events=True,
        collect_pitch_phase_trace=True,
    )
    events = list(result.get("disturbance_events", []))
    if len(events) < 2:
        return {
            "ratchet_mean": float("nan"),
            "ratchet_p90": float("nan"),
            "slope100_mean": float("nan"),
            "slope100_p90": float("nan"),
            "survived": float(result.get("survived", 0.0)),
        }

    pitch_abs_events = np.asarray([abs(np.degrees(float(ev["pitch"]))) for ev in events], dtype=float)
    cycle_deltas = np.diff(pitch_abs_events)
    trace = {int(x["step"]): abs(float(x["pitch_deg"])) for x in result.get("pitch_phase_trace", [])}
    slopes = []
    for ev in events:
        end_step = int(ev["step"])
        slopes.append(_slope_deg_per_s(trace, end_step=end_step, window=100, dt=float(evaluator.dt)))
    slopes = np.asarray([v for v in slopes if np.isfinite(v)], dtype=float)

    return {
        "ratchet_mean": float(np.mean(cycle_deltas)),
        "ratchet_p90": float(np.percentile(cycle_deltas, 90)),
        "slope100_mean": float(np.mean(slopes)) if slopes.size else float("nan"),
        "slope100_p90": float(np.percentile(slopes, 90)) if slopes.size else float("nan"),
        "survived": float(result.get("survived", 0.0)),
    }


def aggregate_metrics(
    evaluator: ControllerEvaluator,
    params: CandidateParams,
    cfg_base: EpisodeConfig,
    seeds: List[int],
    xy: float,
) -> dict[str, float]:
    cfg = replace(cfg_base, disturbance_magnitude_xy=float(xy))
    ms = [episode_metrics(evaluator, params, cfg, s) for s in seeds]
    return {
        "ratchet_mean": float(np.nanmean([m["ratchet_mean"] for m in ms])),
        "ratchet_p90": float(np.nanmean([m["ratchet_p90"] for m in ms])),
        "slope100_mean": float(np.nanmean([m["slope100_mean"] for m in ms])),
        "slope100_p90": float(np.nanmean([m["slope100_p90"] for m in ms])),
        "survival_rate": float(np.mean([m["survived"] for m in ms])),
    }


def main() -> None:
    args = parse_args()
    seeds, steps = resolve_seeds(args)
    evaluator = ControllerEvaluator(Path(__file__).with_name("final.xml"))
    params = baseline_candidate()
    cfg_base = EpisodeConfig(
        steps=int(steps),
        disturbance_magnitude_xy=0.0,
        disturbance_magnitude_z=float(args.disturbance_z),
        disturbance_interval=max(int(args.disturbance_interval), 1),
        controller_family=str(args.controller_family),
        preset=str(args.preset),
        stability_profile=str(args.stability_profile),
        control_hz=float(args.control_hz),
        control_delay_steps=max(int(args.control_delay_steps), 0),
        hardware_realistic=not bool(args.legacy_model),
    )

    target = float(args.target_ratchet_deg_per_cycle)
    lo = float(min(args.xy_min, args.xy_max))
    hi = float(max(args.xy_min, args.xy_max))

    cache: Dict[float, dict[str, float]] = {}

    def eval_xy(xy: float) -> dict[str, float]:
        key = float(np.round(xy, 6))
        if key not in cache:
            cache[key] = aggregate_metrics(evaluator, params, cfg_base, seeds, key)
        return cache[key]

    m_lo = eval_xy(lo)
    m_hi = eval_xy(hi)
    m_zero = eval_xy(0.0)
    baseline_ratchet = float(m_zero["ratchet_mean"]) if bool(args.baseline_subtract_zero_xy) else 0.0

    def effective(metric: dict[str, float]) -> float:
        return float(metric["ratchet_mean"] - baseline_ratchet)

    print("=== LQR Envelope Search ===")
    print(
        f"controller_family={cfg_base.controller_family} seeds={seeds} interval={cfg_base.disturbance_interval} "
        f"steps={cfg_base.steps} target_ratchet={target:+.4f}deg/cycle "
        f"baseline_subtract_zero_xy={bool(args.baseline_subtract_zero_xy)}"
    )
    print(
        f"xy=0.000 ratchet_mean={m_zero['ratchet_mean']:+.4f} slope100_mean={m_zero['slope100_mean']:+.4f}deg/s "
        f"survival={m_zero['survival_rate']:.3f}"
    )
    print(
        f"xy={lo:.3f} ratchet_mean={m_lo['ratchet_mean']:+.4f} eff={effective(m_lo):+.4f} "
        f"ratchet_p90={m_lo['ratchet_p90']:+.4f} "
        f"slope100_mean={m_lo['slope100_mean']:+.4f}deg/s survival={m_lo['survival_rate']:.3f}"
    )
    print(
        f"xy={hi:.3f} ratchet_mean={m_hi['ratchet_mean']:+.4f} eff={effective(m_hi):+.4f} "
        f"ratchet_p90={m_hi['ratchet_p90']:+.4f} "
        f"slope100_mean={m_hi['slope100_mean']:+.4f}deg/s survival={m_hi['survival_rate']:.3f}"
    )

    lo_ok = bool(np.isfinite(m_lo["ratchet_mean"]) and (effective(m_lo) <= target))
    hi_ok = bool(np.isfinite(m_hi["ratchet_mean"]) and (effective(m_hi) <= target))
    if (not lo_ok) and (not hi_ok):
        print("No feasible point in range: ratchet is above target even at xy-min.")
        return
    if lo_ok and hi_ok:
        print("Entire range feasible: ratchet is below target even at xy-max.")
        return

    stable_xy = lo if lo_ok else hi
    unstable_xy = hi if lo_ok else lo
    if stable_xy > unstable_xy:
        stable_xy, unstable_xy = unstable_xy, stable_xy

    for i in range(1, max(int(args.max_iters), 1) + 1):
        mid = 0.5 * (stable_xy + unstable_xy)
        m_mid = eval_xy(mid)
        eff_mid = effective(m_mid)
        ok = bool(np.isfinite(m_mid["ratchet_mean"]) and (eff_mid <= target))
        print(
            f"iter={i} xy={mid:.4f} ratchet_mean={m_mid['ratchet_mean']:+.4f} eff={eff_mid:+.4f} "
            f"ratchet_p90={m_mid['ratchet_p90']:+.4f} slope100_mean={m_mid['slope100_mean']:+.4f}deg/s "
            f"survival={m_mid['survival_rate']:.3f} {'OK' if ok else 'FAIL'}"
        )
        if ok:
            stable_xy = mid
        else:
            unstable_xy = mid
        if abs(unstable_xy - stable_xy) <= float(max(args.xy_tol, 1e-6)):
            break

    print(
        f"boundary_xy_approx={stable_xy:.4f} "
        f"(largest tested magnitude with effective_ratchet <= {target:+.4f}deg/cycle)"
    )


if __name__ == "__main__":
    main()
