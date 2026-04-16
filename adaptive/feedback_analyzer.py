"""
Analyzes training log CSVs to extract learning curve metrics
that are fed back to the LLM advisor for adaptive curriculum generation.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List


@dataclass
class PhaseFeedback:
    phase: int
    algorithm: str
    n_timesteps: int
    mean_reward: float
    final_reward: float      # mean of last 20% of log points
    initial_reward: float    # mean of first 20% of log points
    improvement_pct: float   # (final - initial) / |initial| * 100
    reward_std: float        # std of last 20% (stability measure)
    is_plateaued: bool
    trend: str               # "improving", "plateaued", "degrading", "unstable"
    strategy_used: str       # name of generation strategy that produced this data
    reward_history: List[float] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"Phase {self.phase} [{self.algorithm}] | strategy={self.strategy_used} | "
            f"trend={self.trend} | reward {self.initial_reward:.4f} -> {self.final_reward:.4f} "
            f"({self.improvement_pct:+.1f}%) | std={self.reward_std:.4f}"
        )


def analyze_log(log_path: str, phase: int, algorithm: str,
                strategy_used: str = "unknown") -> PhaseFeedback:
    """
    Read a training CSV log (columns: episode, timestep, reward) and
    extract learning curve metrics for LLM feedback.
    """
    df = pd.read_csv(log_path)
    if df.empty:
        raise ValueError(f"Empty log file: {log_path}")

    rewards = df["reward"].values
    n = len(rewards)
    n20 = max(1, n // 5)

    initial_reward = float(np.mean(rewards[:n20]))
    final_reward = float(np.mean(rewards[-n20:]))
    mean_reward = float(np.mean(rewards))
    reward_std = float(np.std(rewards[-n20:]))

    improvement_pct = 0.0
    if abs(initial_reward) > 1e-9:
        improvement_pct = (final_reward - initial_reward) / abs(initial_reward) * 100

    # Plateau detection: linear fit slope on last 50% of training
    n50 = max(2, n // 2)
    x = np.arange(n50, dtype=float)
    y = rewards[-n50:]
    slope = float(np.polyfit(x, y, 1)[0]) if n50 >= 2 else 0.0

    # Normalize slope relative to reward range so threshold is scale-independent
    reward_range = max(abs(rewards.max() - rewards.min()), 1e-9)
    normalized_slope = slope * n50 / reward_range
    is_plateaued = abs(normalized_slope) < 0.05

    # Classify overall trend
    relative_std = reward_std / (abs(final_reward) + 1e-9)
    if relative_std > 0.4:
        trend = "unstable"
    elif normalized_slope > 0.05:
        trend = "improving"
    elif normalized_slope < -0.05:
        trend = "degrading"
    else:
        trend = "plateaued"

    n_timesteps = int(df["timestep"].max()) if "timestep" in df.columns else n

    return PhaseFeedback(
        phase=phase,
        algorithm=algorithm,
        n_timesteps=n_timesteps,
        mean_reward=mean_reward,
        final_reward=final_reward,
        initial_reward=initial_reward,
        improvement_pct=improvement_pct,
        reward_std=reward_std,
        is_plateaued=is_plateaued,
        trend=trend,
        strategy_used=strategy_used,
        reward_history=rewards.tolist(),
    )


def format_history_for_llm(history: List[PhaseFeedback]) -> str:
    """Render training history as a structured text block for the LLM prompt."""
    lines = [
        "## Training History",
        f"Completed phases: {len(history)}",
        "",
    ]
    for fb in history:
        # Downsample reward history to at most 20 points to keep prompt concise
        step = max(1, len(fb.reward_history) // 20)
        sampled = fb.reward_history[::step][:20]
        sampled_str = "[" + ", ".join(f"{v:.4f}" for v in sampled) + "]"
        lines += [
            f"### Phase {fb.phase}",
            f"- Algorithm: {fb.algorithm}",
            f"- Generation strategy used: {fb.strategy_used}",
            f"- Timesteps this phase: {fb.n_timesteps}",
            f"- Reward (initial → final): {fb.initial_reward:.4f} → {fb.final_reward:.4f}",
            f"- Improvement: {fb.improvement_pct:+.1f}%",
            f"- Reward std (last 20%): {fb.reward_std:.4f}",
            f"- Trend: {fb.trend}",
            f"- Reward curve (sampled): {sampled_str}",
            "",
        ]
    return "\n".join(lines)
