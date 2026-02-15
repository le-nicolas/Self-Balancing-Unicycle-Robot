import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FAMILY_ORDER = [
    "baseline_robust_hinf_like",
    "hybrid_modern",
    "current",
    "paper_split_baseline",
    "baseline_mpc",
]


def newest_benchmark_csv(results_dir: Path) -> Path:
    candidates = sorted(results_dir.glob("benchmark_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    for candidate in candidates:
        try:
            with candidate.open("r", encoding="utf-8", newline="") as f:
                header = f.readline()
            if "controller_family" in header and "mode_id" in header:
                return candidate
        except OSError:
            continue
    raise FileNotFoundError("No benchmark CSV with expected columns was found.")


def load_baseline_rows(
    csv_path: Path,
    mode_id: str,
    model_variant_id: str,
    domain_profile_id: str,
):
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    filtered = [
        r
        for r in rows
        if r.get("run_id") == "baseline"
        and r.get("mode_id") == mode_id
        and r.get("model_variant_id") == model_variant_id
        and r.get("domain_profile_id") == domain_profile_id
    ]
    if not filtered:
        raise ValueError(
            f"No baseline rows for mode={mode_id}, variant={model_variant_id}, domain={domain_profile_id} in {csv_path}"
        )
    by_family = {str(r["controller_family"]): r for r in filtered}
    return by_family


def pick_families(by_family: dict[str, dict[str, str]]) -> list[str]:
    chosen = [f for f in FAMILY_ORDER if f in by_family]
    if not chosen:
        chosen = sorted(by_family.keys())
    return chosen


def to_float(value: str, fallback: float = np.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def main():
    parser = argparse.ArgumentParser(description="Plot paper-vs-simulation comparison figure for README.")
    parser.add_argument("--csv", type=str, default=None, help="Benchmark CSV path. Defaults to latest under final/results.")
    parser.add_argument("--mode-id", type=str, default="mode_default")
    parser.add_argument("--model-variant-id", type=str, default="nominal")
    parser.add_argument("--domain-profile-id", type=str, default="default")
    parser.add_argument("--out", type=str, default="final/results/paper_vs_sim_comparison.png")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    results_dir = repo_root / "final" / "results"

    csv_path = Path(args.csv) if args.csv else newest_benchmark_csv(results_dir)
    if not csv_path.is_absolute():
        csv_path = repo_root / csv_path
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = repo_root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    by_family = load_baseline_rows(csv_path, args.mode_id, args.model_variant_id, args.domain_profile_id)
    families = pick_families(by_family)

    score = np.array([to_float(by_family[f].get("score_composite")) for f in families], dtype=float)
    survival = np.array([to_float(by_family[f].get("survival_rate")) for f in families], dtype=float)
    crash = np.array([to_float(by_family[f].get("crash_rate")) for f in families], dtype=float)
    roll = np.array([to_float(by_family[f].get("worst_roll_deg")) for f in families], dtype=float)
    pitch = np.array([to_float(by_family[f].get("worst_pitch_deg")) for f in families], dtype=float)

    episodes = int(to_float(by_family[families[0]].get("episodes"), fallback=np.nan))
    steps = int(to_float(by_family[families[0]].get("steps_per_episode"), fallback=np.nan))

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))

    x = np.arange(len(families), dtype=float)
    colors = plt.cm.viridis(np.clip(survival, 0.0, 1.0))
    bars = axes[0].bar(x, score, color=colors, edgecolor="black", linewidth=0.6)
    axes[0].axhline(75.0, linestyle="--", linewidth=1.0, color="#444444", label="score=75 threshold")
    axes[0].set_xticks(x, families, rotation=20, ha="right")
    axes[0].set_ylabel("Composite score")
    axes[0].set_title("Simulation benchmark snapshot")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(loc="upper right", frameon=False)
    for i, b in enumerate(bars):
        axes[0].text(
            b.get_x() + b.get_width() / 2.0,
            b.get_height() + 0.8,
            f"S={survival[i]:.3f}\nC={crash[i]:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    angle_labels = ["Paper ramp", "Paper ladder", "Paper parabolic", "Sim current", "Sim hybrid_modern"]
    paper_roll = [1.0, 2.0, 1.0]
    paper_pitch = [2.0, 4.0, 2.0]

    sim_current_roll = to_float(by_family.get("current", {}).get("worst_roll_deg"))
    sim_current_pitch = to_float(by_family.get("current", {}).get("worst_pitch_deg"))
    sim_hybrid_roll = to_float(by_family.get("hybrid_modern", {}).get("worst_roll_deg"))
    sim_hybrid_pitch = to_float(by_family.get("hybrid_modern", {}).get("worst_pitch_deg"))

    roll_vals = np.array(paper_roll + [sim_current_roll, sim_hybrid_roll], dtype=float)
    pitch_vals = np.array(paper_pitch + [sim_current_pitch, sim_hybrid_pitch], dtype=float)

    x2 = np.arange(len(angle_labels), dtype=float)
    w = 0.36
    axes[1].bar(x2 - w / 2.0, roll_vals, width=w, label="Roll max abs (deg)", color="#4C72B0")
    axes[1].bar(x2 + w / 2.0, pitch_vals, width=w, label="Pitch max abs (deg)", color="#DD8452")
    axes[1].set_xticks(x2, angle_labels, rotation=20, ha="right")
    axes[1].set_ylabel("Max absolute angle (deg)")
    axes[1].set_title("2013 paper envelopes vs simulation")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False)

    fig.suptitle(
        (
            "Paper-vs-Simulation Comparison (for context, not strict apples-to-apples)\n"
            f"CSV: {csv_path.name} | mode={args.mode_id} | episodes={episodes} | steps={steps}"
        ),
        fontsize=11,
    )
    fig.text(
        0.5,
        0.01,
        "Paper reference: IEEE TIE 2013, DOI 10.1109/TIE.2012.2208431. "
        "Paper values are task-specific envelopes; simulation values are benchmark worst-case under injected noise/delay/disturbance.",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    fig.tight_layout(rect=[0.0, 0.05, 1.0, 0.90])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    print(f"Wrote: {out_path}")
    print(f"Source CSV: {csv_path}")


if __name__ == "__main__":
    main()
