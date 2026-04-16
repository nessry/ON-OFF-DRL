"""
LLMAdvisor: uses the OpenAI API to analyze training feedback and return
a JSON generation strategy for the next training phase.

Falls back to FallbackAdvisor (rule-based curriculum) when the openai
library is not installed or OPENAI_API_KEY is not set.
"""

import json
import os
from typing import List

from adaptive.feedback_analyzer import PhaseFeedback, format_history_for_llm
from adaptive.trace_generator import TraceParams, trace_params_from_dict

try:
    import openai
    _OPENAI_OK = True
except ImportError:
    _OPENAI_OK = False


# ---------------------------------------------------------------------------
# Prompt engineering
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an expert in reinforcement learning curriculum design and distributed systems.

## Task
An RL agent (PPO or ACER) is learning to allocate tasks to servers in an Open RAN system.
You will receive its recent training history and must suggest a trace-generation strategy
that will help the agent improve via curriculum learning.

## Environment
- 10 servers, each with 100 CPU units (1 unit = 1/100 core) and 100 memory units.
- Each task has: plan_cpu ∈ [1,100], plan_mem ∈ [1,100], duration (seconds), arrival_time.
- The agent assigns each task to one server. The objective is:
    reward = -(w1 × total_power  +  w2 × cumulative_latency)
  where power is a nonlinear function of CPU utilisation and latency is the cumulative
  sum of (start_time − arrive_time) over all tasks.
- Good policies: spread load to keep servers from idling, avoid long queues, prefer
  servers with enough free resources.

## Curriculum Principles
- If reward is plateaued → introduce harder or more diverse scenarios.
- If reward is improving → continue in a similar direction but increase difficulty slightly.
- If reward is unstable → reduce variance: use smoother distributions or a smaller synthetic_ratio.
- If reward is degrading → revert toward real data (lower synthetic_ratio).
- Mix real Alibaba traces with synthetic ones: synthetic_ratio ∈ [0, 1].

## Available Distribution Types
cpu / mem fields:
  "uniform"    – sample from [low, high]
  "normal"     – clipped normal(mean, std) in [low, high]
  "bimodal"    – mixture of two normals at low_mean / high_mean, weighted by ratio ∈ [0,1]
  "exponential"– exponential(scale), clipped at high
  "heavy_tail" – Pareto-based, scale sets the median

duration field: use "exponential" (typical), "normal", or "uniform".

arrival_pattern:
  "uniform"  – tasks spread uniformly over total_time_span
  "bursty"   – Poisson bursts; burst_prob fraction of time at burst_intensity × base_rate
  "periodic" – tasks cluster around 10 periodic peaks

## Response Format
Respond with ONLY a JSON object — no markdown, no commentary outside the JSON:
{
  "reasoning": "<2-3 sentences explaining the diagnosis and strategy>",
  "strategy_name": "<short slug, e.g. high_load_bursty>",
  "trace_params": {
    "n_tasks": 2000,
    "cpu":      { "type": "...", "low": 5, "high": 95, "mean": 50, "std": 20,
                  "low_mean": 20, "high_mean": 80, "ratio": 0.3, "scale": 50 },
    "mem":      { ...same structure as cpu... },
    "duration": { "type": "exponential", "low": 10, "high": 5000, "scale": 500 },
    "arrival_pattern": "uniform",
    "total_time_span": 86400.0,
    "burst_prob": 0.3,
    "burst_intensity": 3.0
  },
  "env_adjustments": {
    "w1_scale": 1.0,
    "w2_scale": 1.0
  },
  "synthetic_ratio": 0.5,
  "notes": "<optional implementation notes>"
}

Only include the distribution fields relevant to the chosen type (e.g., no need for
mean/std when type is "uniform"). w1_scale / w2_scale multiply the current weights
(use 1.0 to leave them unchanged).
"""


def _build_user_message(history: List[PhaseFeedback]) -> str:
    return (
        format_history_for_llm(history)
        + "\n\nBased on this training history, provide the JSON generation strategy "
          "for the next phase."
    )


def _parse_response(text: str) -> dict:
    """Extract and parse JSON from the model response."""
    text = text.strip()
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0].strip()
    return json.loads(text)


# ---------------------------------------------------------------------------
# Advisors
# ---------------------------------------------------------------------------

class LLMAdvisor:
    """Uses OpenAI API to generate adaptive trace strategies."""

    def __init__(self, model: str = "gpt-4o"):
        if not _OPENAI_OK:
            raise ImportError(
                "openai package not found. Install with: pip install openai"
            )
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set."
            )
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def get_strategy(self, history: List[PhaseFeedback]) -> dict:
        """
        Call GPT with training history and return a strategy dict containing:
          reasoning, strategy_name, trace_params, env_adjustments,
          synthetic_ratio, notes

        Retries up to 3 times on parse failure. Falls back to FallbackAdvisor
        on all failures rather than crashing the training loop.
        """
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": _build_user_message(history)},
        ]
        last_error = None
        for attempt in range(1, 4):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=2048,
                    messages=messages,
                )
                response_text = response.choices[0].message.content
                strategy = _parse_response(response_text)
                return strategy
            except (json.JSONDecodeError, KeyError, IndexError, ValueError) as e:
                last_error = e
                print(f"[advisor] LLM parse error (attempt {attempt}/3): {e}")
                # Ask the model to try again with an explicit reminder
                messages.append({"role": "assistant", "content": response_text})
                messages.append({
                    "role": "user",
                    "content": "Your response could not be parsed as JSON. "
                               "Reply with ONLY a valid JSON object, no markdown.",
                })

        print(f"[advisor] All retries failed ({last_error}). "
              "Falling back to rule-based strategy for this phase.")
        return FallbackAdvisor().get_strategy(history)


class FallbackAdvisor:
    """
    Rule-based curriculum advisor used when Claude API is unavailable.
    Cycles through a preset sequence of strategies.
    """

    _PRESETS = [
        {
            "reasoning": "Baseline: train on real Alibaba traces.",
            "strategy_name": "baseline_real",
            "trace_params": {
                "n_tasks": 2000,
                "cpu":      {"type": "uniform", "low": 5, "high": 80},
                "mem":      {"type": "uniform", "low": 5, "high": 80},
                "duration": {"type": "exponential", "scale": 500, "low": 10, "high": 10000},
                "arrival_pattern": "uniform",
                "total_time_span": 86400.0,
                "burst_prob": 0.3,
                "burst_intensity": 3.0,
            },
            "env_adjustments": {"w1_scale": 1.0, "w2_scale": 1.0},
            "synthetic_ratio": 0.0,
            "notes": "Phase 0 – pure real data.",
        },
        {
            "reasoning": (
                "Introduce high-load bursty scenarios to challenge load-balancing "
                "and prevent the agent from over-relying on patterns in the real trace."
            ),
            "strategy_name": "high_load_bursty",
            "trace_params": {
                "n_tasks": 2000,
                "cpu":      {"type": "bimodal", "low_mean": 60, "high_mean": 90,
                             "ratio": 0.4, "std": 10, "low": 1, "high": 100},
                "mem":      {"type": "uniform", "low": 30, "high": 90},
                "duration": {"type": "exponential", "scale": 300, "low": 10, "high": 5000},
                "arrival_pattern": "bursty",
                "total_time_span": 86400.0,
                "burst_prob": 0.4,
                "burst_intensity": 5.0,
            },
            "env_adjustments": {"w1_scale": 1.0, "w2_scale": 1.0},
            "synthetic_ratio": 0.5,
            "notes": "50% high-CPU bursty synthetic tasks mixed with real data.",
        },
        {
            "reasoning": (
                "Focus on latency: long-duration, uniform-CPU tasks. "
                "Upweight the latency term so the agent learns to minimise queueing."
            ),
            "strategy_name": "latency_focused",
            "trace_params": {
                "n_tasks": 2000,
                "cpu":      {"type": "uniform", "low": 10, "high": 60},
                "mem":      {"type": "normal", "mean": 40, "std": 20, "low": 5, "high": 95},
                "duration": {"type": "normal", "mean": 1500, "std": 500, "low": 100, "high": 8000},
                "arrival_pattern": "uniform",
                "total_time_span": 86400.0,
                "burst_prob": 0.2,
                "burst_intensity": 2.0,
            },
            "env_adjustments": {"w1_scale": 0.5, "w2_scale": 2.0},
            "synthetic_ratio": 0.6,
            "notes": "Increased w2 to emphasise latency; long-running tasks.",
        },
        {
            "reasoning": (
                "Heavy-tail CPU requirements: rare very-large tasks test whether "
                "the agent can handle resource-tight situations."
            ),
            "strategy_name": "heavy_tail_cpu",
            "trace_params": {
                "n_tasks": 2000,
                "cpu":      {"type": "heavy_tail", "scale": 20, "low": 1, "high": 100},
                "mem":      {"type": "uniform", "low": 5, "high": 70},
                "duration": {"type": "exponential", "scale": 600, "low": 10, "high": 10000},
                "arrival_pattern": "uniform",
                "total_time_span": 86400.0,
                "burst_prob": 0.3,
                "burst_intensity": 3.0,
            },
            "env_adjustments": {"w1_scale": 1.5, "w2_scale": 1.0},
            "synthetic_ratio": 0.4,
            "notes": "Upweight power term; heavy-tail CPU to stress power model.",
        },
        {
            "reasoning": (
                "Periodic arrival pattern: 10 load peaks test the agent's ability "
                "to pre-emptively distribute load before saturation."
            ),
            "strategy_name": "periodic_peaks",
            "trace_params": {
                "n_tasks": 2000,
                "cpu":      {"type": "normal", "mean": 45, "std": 25, "low": 1, "high": 100},
                "mem":      {"type": "normal", "mean": 40, "std": 20, "low": 1, "high": 100},
                "duration": {"type": "exponential", "scale": 400, "low": 10, "high": 6000},
                "arrival_pattern": "periodic",
                "total_time_span": 86400.0,
                "burst_prob": 0.3,
                "burst_intensity": 4.0,
            },
            "env_adjustments": {"w1_scale": 1.0, "w2_scale": 1.5},
            "synthetic_ratio": 0.55,
            "notes": "Periodic peaks with balanced weights.",
        },
    ]

    def __init__(self):
        self._phase = 0

    def get_strategy(self, history: List[PhaseFeedback]) -> dict:
        """
        Return the next preset strategy, with simple rule-based override:
          - Degrading trend → revert to more real data (halve synthetic_ratio).
          - Unstable trend  → halve synthetic_ratio and smooth distributions.
        """
        strategy = dict(self._PRESETS[self._phase % len(self._PRESETS)])
        self._phase += 1

        if history:
            last = history[-1]
            if last.trend == "degrading":
                strategy = dict(strategy)
                strategy["synthetic_ratio"] = max(0.0, strategy["synthetic_ratio"] - 0.3)
                strategy["reasoning"] += " [Overridden: degrading → reduced synthetic ratio.]"
            elif last.trend == "unstable":
                strategy = dict(strategy)
                strategy["synthetic_ratio"] = max(0.0, strategy["synthetic_ratio"] - 0.2)
                strategy["reasoning"] += " [Overridden: unstable → reduced synthetic ratio.]"

        return strategy


def build_advisor(use_llm: bool = True,
                  model: str = "gpt-4o") -> "LLMAdvisor | FallbackAdvisor":
    """
    Return an LLMAdvisor if possible, otherwise a FallbackAdvisor.
    Set use_llm=False to force rule-based fallback (useful for offline testing).
    """
    if use_llm and _OPENAI_OK and os.environ.get("OPENAI_API_KEY"):
        print("[advisor] Using LLMAdvisor (OpenAI API)")
        return LLMAdvisor(model=model)
    else:
        reasons = []
        if not use_llm:
            reasons.append("--no-llm flag set")
        if not _OPENAI_OK:
            reasons.append("openai not installed")
        if not os.environ.get("OPENAI_API_KEY"):
            reasons.append("OPENAI_API_KEY not set")
        print(f"[advisor] Using FallbackAdvisor ({', '.join(reasons)})")
        return FallbackAdvisor()


def apply_env_adjustments(env, adjustments: dict,
                           base_w1: float, base_w2: float) -> None:
    """Apply w1/w2 scale adjustments to an AdaptiveEnv instance."""
    w1 = base_w1 * float(adjustments.get("w1_scale", 1.0))
    w2 = base_w2 * float(adjustments.get("w2_scale", 1.0))
    env.set_weights(w1=w1, w2=w2)
