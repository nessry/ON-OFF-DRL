"""
adaptive_plot.py
================
Visualizes results from an adaptive_loop.py run against the baseline PPO/ACER logs.

Produces three figures saved to plots/adaptive/:

  1. phase_curves.pdf   — Per-phase reward curves overlaid (one line per phase,
                          colored by phase index, labeled with LLM strategy name).

  2. phase_summary.pdf  — Bar chart of final reward per phase + improvement % on a
                          twin axis, showing the phase-to-phase trend at a glance.

  3. cumulative_vs_baseline.pdf — Stitched adaptive reward vs. baseline PPO (and
                                  optionally ACER) on a shared cumulative-timestep
                                  axis, making convergence speed directly comparable.

Usage
-----
  python adaptive_plot.py                          # auto-detects latest run
  python adaptive_plot.py --run_dir adaptive_runs/run_20260318_004407
  python adaptive_plot.py --baseline_dir tests     # override baseline location
"""

import os
import sys
import json
import glob
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

WINDOW = 5          # rolling-mean window (matches reward_plot.py)
LINEWIDTH = 2.5
ALPHA = 0.9
BASELINE_DIR = "tests"
OUTPUT_DIR = os.path.join("plots", "adaptive")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def smooth(series: pd.Series, window: int = WINDOW) -> pd.Series:
    return series.rolling(window=window, win_type="triang",
                          min_periods=1).mean()


def latest_run_dir(base: str = "adaptive_runs") -> str | None:
    """Return the most recently created run directory, or None."""
    runs = sorted(glob.glob(os.path.join(base, "run_*")))
    return runs[-1] if runs else None


def load_phase_logs(run_dir: str) -> list[tuple[int, pd.DataFrame]]:
    """Load all phase_NNN.csv files sorted by phase index."""
    log_dir = os.path.join(run_dir, "logs")
    paths = sorted(glob.glob(os.path.join(log_dir, "phase_*.csv")))
    result = []
    for p in paths:
        idx = int(os.path.basename(p).replace("phase_", "").replace(".csv", ""))
        df = pd.read_csv(p)
        if not df.empty and "reward" in df.columns:
            result.append((idx, df))
    return result


def load_strategy_log(run_dir: str) -> dict[int, str]:
    """Return {phase_index: strategy_name} from strategy_log.jsonl."""
    path = os.path.join(run_dir, "strategy_log.jsonl")
    names: dict[int, str] = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    names[entry["phase"]] = entry.get("strategy_name", f"phase_{entry['phase']}")
                except (json.JSONDecodeError, KeyError):
                    pass
    return names


def load_baseline(baseline_dir: str, algo: str) -> pd.DataFrame | None:
    """Load first reward CSV for PPO or ACER from the baseline directory."""
    pattern = os.path.join(
        baseline_dir,
        f"{algo}_files", "resource_allocation", "reward",
        f"{algo}_resource_allocation_log_0.csv",
    )
    if os.path.exists(pattern):
        df = pd.read_csv(pattern)
        return df if not df.empty else None
    return None


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

# ---------------------------------------------------------------------------
# Plot 1: per-phase reward curves
# ---------------------------------------------------------------------------

def plot_phase_curves(phases: list[tuple[int, pd.DataFrame]],
                      strategies: dict[int, str],
                      out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = cm.get_cmap("tab10", max(len(phases), 1))

    for phase_idx, df in phases:
        color = cmap(phase_idx % 10)
        label = f"Phase {phase_idx}: {strategies.get(phase_idx, 'unknown')}"
        y = smooth(df["reward"])
        x = df["timestep"]
        ax.plot(x, y, color=color, linewidth=LINEWIDTH, alpha=ALPHA,
                label=label, marker="o", markevery=max(1, len(x)//10),
                markersize=4)

    ax.set_xlabel("Timesteps (within phase)", fontsize=12)
    ax.set_ylabel("Reward (smoothed)", fontsize=12)
    ax.set_title("Adaptive Training: Per-Phase Reward Curves", fontsize=13)
    ax.grid(color="gray", linestyle="-", linewidth=1, alpha=0.2)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 2: phase summary bar chart
# ---------------------------------------------------------------------------

def plot_phase_summary(phases: list[tuple[int, pd.DataFrame]],
                       strategies: dict[int, str],
                       out_path: str) -> None:
    if not phases:
        return

    phase_ids = [p for p, _ in phases]
    final_rewards = []
    improvements = []

    for phase_idx, df in phases:
        rewards = df["reward"].values
        n20 = max(1, len(rewards) // 5)
        final = float(np.mean(rewards[-n20:]))
        initial = float(np.mean(rewards[:n20]))
        final_rewards.append(final)
        pct = (final - initial) / (abs(initial) + 1e-9) * 100
        improvements.append(pct)

    x = np.arange(len(phase_ids))
    labels = [strategies.get(i, f"P{i}") for i in phase_ids]

    fig, ax1 = plt.subplots(figsize=(max(8, len(phases) * 1.5), 5))
    ax2 = ax1.twinx()

    bars = ax1.bar(x - 0.2, final_rewards, width=0.35, color="steelblue",
                   alpha=0.85, label="Final reward (mean last 20%)")
    ax2.bar(x + 0.2, improvements, width=0.35, color="coral",
            alpha=0.85, label="Intra-phase improvement (%)")

    ax1.set_xlabel("Phase", fontsize=12)
    ax1.set_ylabel("Final Reward", fontsize=12, color="steelblue")
    ax2.set_ylabel("Intra-phase Improvement (%)", fontsize=12, color="coral")
    ax1.set_title("Adaptive Training: Per-Phase Summary", fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"P{i}\n{l}" for i, l in zip(phase_ids, labels)],
                        fontsize=8, rotation=15, ha="right")
    ax1.tick_params(axis="y", colors="steelblue")
    ax2.tick_params(axis="y", colors="coral")
    ax1.grid(axis="y", color="gray", linestyle="--", alpha=0.3)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="lower right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 3: cumulative adaptive vs. baseline
# ---------------------------------------------------------------------------

def plot_cumulative_vs_baseline(phases: list[tuple[int, pd.DataFrame]],
                                baseline_ppo: pd.DataFrame | None,
                                baseline_acer: pd.DataFrame | None,
                                out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))

    # --- Stitch adaptive phases into one cumulative series ---
    cumulative_timesteps = []
    cumulative_rewards = []
    offset = 0
    for phase_idx, df in phases:
        ts = df["timestep"].values
        rew = df["reward"].values
        cumulative_timesteps.extend((ts + offset).tolist())
        cumulative_rewards.extend(rew.tolist())
        offset += int(ts.max())

    if cumulative_timesteps:
        adaptive_df = pd.DataFrame({
            "timestep": cumulative_timesteps,
            "reward": cumulative_rewards,
        }).sort_values("timestep").reset_index(drop=True)
        y_adaptive = smooth(adaptive_df["reward"])
        ax.plot(adaptive_df["timestep"], y_adaptive,
                color="royalblue", linewidth=LINEWIDTH, alpha=ALPHA,
                label="Adaptive PPO (LLM-guided)", marker="o",
                markevery=max(1, len(adaptive_df)//15), markersize=4)

    # --- Baseline PPO ---
    if baseline_ppo is not None:
        y_ppo = smooth(baseline_ppo["reward"])
        ax.plot(baseline_ppo["timestep"], y_ppo,
                color="red", linewidth=LINEWIDTH, alpha=0.8,
                label="Baseline PPO (fixed traces)", marker="^",
                markevery=max(1, len(baseline_ppo)//15), markersize=4,
                linestyle="--")

    # --- Baseline ACER ---
    if baseline_acer is not None:
        y_acer = smooth(baseline_acer["reward"])
        ax.plot(baseline_acer["timestep"], y_acer,
                color="green", linewidth=LINEWIDTH, alpha=0.8,
                label="Baseline ACER (fixed traces)", marker="s",
                markevery=max(1, len(baseline_acer)//15), markersize=4,
                linestyle="--")

    # Phase boundary markers
    if cumulative_timesteps and len(phases) > 1:
        boundary = 0
        for i, (phase_idx, df) in enumerate(phases[:-1]):
            boundary += int(df["timestep"].max())
            ax.axvline(x=boundary, color="gray", linestyle=":", linewidth=1.2,
                       alpha=0.7)
            ax.text(boundary + 200, ax.get_ylim()[0] * 0.98,
                    f"P{phase_idx+1}", fontsize=8, color="gray", va="bottom")

    ax.set_xlabel("Cumulative Timesteps", fontsize=12)
    ax.set_ylabel("Reward (smoothed)", fontsize=12)
    ax.set_title("Adaptive vs. Baseline: Convergence Comparison", fontsize=13)
    ax.grid(color="gray", linestyle="-", linewidth=1, alpha=0.2)
    ax.legend(fontsize=10, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot adaptive training results vs. PPO/ACER baselines"
    )
    parser.add_argument(
        "--run_dir", type=str, default=None,
        help="Path to adaptive run directory (default: latest under adaptive_runs/)",
    )
    parser.add_argument(
        "--baseline_dir", type=str, default=BASELINE_DIR,
        help=f"Directory containing PPO_files/ and ACER_files/ (default: {BASELINE_DIR})",
    )
    parser.add_argument(
        "--output_dir", type=str, default=OUTPUT_DIR,
        help=f"Where to save plots (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--no_acer", action="store_true",
        help="Skip loading ACER baseline",
    )
    args = parser.parse_args()

    # Resolve run directory
    run_dir = args.run_dir or latest_run_dir()
    if run_dir is None or not os.path.isdir(run_dir):
        print(
            "No adaptive run directory found.\n"
            "Run  python adaptive_loop.py  first to generate training data,\n"
            "then re-run this script."
        )
        sys.exit(1)

    print(f"Loading adaptive run: {run_dir}")

    phases = load_phase_logs(run_dir)
    if not phases:
        print(f"No phase log CSVs found in {run_dir}/logs/. Nothing to plot.")
        sys.exit(1)

    strategies = load_strategy_log(run_dir)
    baseline_ppo  = load_baseline(args.baseline_dir, "PPO")
    baseline_acer = None if args.no_acer else load_baseline(args.baseline_dir, "ACER")

    ensure_output_dir(args.output_dir)

    print(f"\nPhases loaded : {len(phases)}")
    print(f"Baseline PPO  : {'found' if baseline_ppo  is not None else 'not found'}")
    print(f"Baseline ACER : {'found' if baseline_acer is not None else 'not found'}")
    print()

    plot_phase_curves(
        phases, strategies,
        os.path.join(args.output_dir, "phase_curves.pdf"),
    )
    plot_phase_summary(
        phases, strategies,
        os.path.join(args.output_dir, "phase_summary.pdf"),
    )
    plot_cumulative_vs_baseline(
        phases, baseline_ppo, baseline_acer,
        os.path.join(args.output_dir, "cumulative_vs_baseline.pdf"),
    )

    print("\nAll plots saved.")


if __name__ == "__main__":
    main()
