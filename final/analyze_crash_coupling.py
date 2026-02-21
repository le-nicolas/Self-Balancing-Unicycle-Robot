import argparse
import json
from pathlib import Path

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
        description=(
            "Replay one benchmark episode and inspect disturbance-event wheel-speed/pitch trend "
            "before crash (trajectory-level coupling check)."
        )
    )
    parser.add_argument(
        "--ablation-json",
        type=str,
        default=str(Path(__file__).with_name("results") / "throughput_ablation_seed42.json"),
        help="Path to throughput_ablation JSON containing episode_seeds and steps.",
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        default=20,
        help="1-based episode index in throughput_ablation episode_seeds list.",
    )
    parser.add_argument(
        "--zero-based-index",
        action="store_true",
        help="Interpret --episode-index as 0-based instead of 1-based.",
    )
    parser.add_argument("--seed-fallback", type=int, default=1074497555, help="Used if --ablation-json is missing.")
    parser.add_argument("--steps", type=int, default=6000, help="Episode length (overridden by JSON when available).")
    parser.add_argument("--controller-family", type=str, default="current")
    parser.add_argument("--preset", type=str, default="default")
    parser.add_argument("--stability-profile", type=str, default="default")
    parser.add_argument("--disturbance-xy", type=float, default=4.0)
    parser.add_argument("--disturbance-z", type=float, default=2.0)
    parser.add_argument(
        "--disturbance-interval",
        type=int,
        default=200,
        help="Set to 200 to match the six-event 4600-5600 window check.",
    )
    parser.add_argument("--control-hz", type=float, default=250.0)
    parser.add_argument("--control-delay-steps", type=int, default=1)
    parser.add_argument("--dob-cutoff-hz", type=float, default=5.0)
    parser.add_argument("--window-start", type=int, default=4600)
    parser.add_argument("--window-end", type=int, default=5600)
    parser.add_argument("--events", type=int, default=6, help="Number of pre-crash disturbance events to inspect.")
    parser.add_argument("--legacy-model", action="store_true", help="Disable hardware-realistic timing/noise model.")
    return parser.parse_args()


def resolve_episode_seed(args: argparse.Namespace) -> tuple[int, int]:
    path = Path(args.ablation_json)
    if not path.exists():
        return int(args.seed_fallback), int(args.steps)
    data = json.loads(path.read_text(encoding="utf-8"))
    steps = int(data.get("steps", args.steps))
    seeds = [int(s) for s in data.get("episode_seeds", [])]
    if not seeds:
        return int(args.seed_fallback), steps
    if bool(args.zero_based_index):
        idx0 = max(int(args.episode_index), 0)
    else:
        idx0 = max(int(args.episode_index) - 1, 0)
    if idx0 >= len(seeds):
        raise ValueError(
            f"episode-index={args.episode_index} out of range for {path} (n={len(seeds)})."
        )
    return int(seeds[idx0]), steps


def summarize(events: list[dict[str, float]]) -> tuple[float, float, bool]:
    if len(events) < 2:
        return 0.0, 0.0, False
    wheel = np.asarray([float(ev["wheel_speed"]) for ev in events], dtype=float)
    pitch = np.asarray([float(ev["pitch"]) for ev in events], dtype=float)
    idx = np.arange(len(events), dtype=float)
    wheel_slope = float(np.polyfit(idx, wheel, deg=1)[0])
    pitch_slope = float(np.polyfit(idx, pitch, deg=1)[0])
    confirmed = (wheel_slope < 0.0) and (pitch_slope > 0.0)
    return wheel_slope, pitch_slope, confirmed


def main() -> None:
    args = parse_args()
    episode_seed, steps = resolve_episode_seed(args)
    cfg = EpisodeConfig(
        steps=int(steps),
        disturbance_magnitude_xy=float(args.disturbance_xy),
        disturbance_magnitude_z=float(args.disturbance_z),
        disturbance_interval=max(int(args.disturbance_interval), 1),
        control_hz=float(args.control_hz),
        control_delay_steps=max(int(args.control_delay_steps), 0),
        hardware_realistic=not bool(args.legacy_model),
        preset=str(args.preset),
        stability_profile=str(args.stability_profile),
        controller_family=str(args.controller_family),
        dob_cutoff_hz=float(max(args.dob_cutoff_hz, 0.0)),
    )
    evaluator = ControllerEvaluator(Path(__file__).with_name("final.xml"))
    result = evaluator.simulate_episode(
        params=baseline_candidate(),
        episode_seed=int(episode_seed),
        config=cfg,
        collect_disturbance_events=True,
    )
    events = list(result.get("disturbance_events", []))
    crash_step = result.get("crash_step", np.nan)
    crashed = bool(result.get("survived", 1.0) < 0.5)
    if crashed and np.isfinite(crash_step):
        events = [ev for ev in events if int(ev["step"]) <= int(crash_step)]

    win_start = int(args.window_start)
    win_end = int(args.window_end)
    win_events = [ev for ev in events if win_start <= int(ev["step"]) <= win_end]
    chosen = win_events[-int(args.events):] if len(win_events) >= int(args.events) else events[-int(args.events):]

    print("=== Crash Coupling Analysis ===")
    if bool(args.zero_based_index):
        print(f"episode_index_0based={int(args.episode_index)} episode_seed={episode_seed}")
    else:
        print(f"episode_index_1based={int(args.episode_index)} episode_seed={episode_seed}")
    print(
        f"controller_family={cfg.controller_family} preset={cfg.preset} "
        f"stability_profile={cfg.stability_profile} disturbance_interval={cfg.disturbance_interval}"
    )
    print(f"crashed={crashed} crash_step={crash_step} crash_reason={result.get('crash_reason', '')}")
    print(f"window=[{win_start},{win_end}] events_selected={len(chosen)}")
    if not chosen:
        print("No disturbance events found for requested window/episode.")
        return

    print("step,wheel_speed_rad_s,pitch_rad,pitch_deg,pitch_rate_rad_s,force_x,force_y,force_z")
    for ev in chosen:
        pitch = float(ev["pitch"])
        print(
            f"{int(ev['step'])},{float(ev['wheel_speed']):+.6f},{pitch:+.6f},{np.degrees(pitch):+.3f},"
            f"{float(ev['pitch_rate']):+.6f},{float(ev['force_x']):+.6f},{float(ev['force_y']):+.6f},{float(ev['force_z']):+.6f}"
        )

    wheel_slope, pitch_slope, confirmed = summarize(chosen)
    print(
        f"trend wheel_slope={wheel_slope:+.6f} rad/s/event "
        f"pitch_slope={pitch_slope:+.6f} rad/event "
        f"coupling_confirmed={int(confirmed)}"
    )


if __name__ == "__main__":
    main()
