"""
Generates synthetic task traces parameterized by LLM-provided distribution specs.

Supported distributions per field (cpu, mem, duration):
  - uniform:     sample from [low, high]
  - normal:      clipped normal(mean, std) in [low, high]
  - bimodal:     mixture of two normals (low_mean / high_mean) weighted by ratio
  - exponential: exponential(scale), clipped at high
  - heavy_tail:  Pareto-derived, for rare very-large tasks

Arrival patterns:
  - uniform:   tasks spread uniformly over total_time_span
  - bursty:    Poisson process with burst periods (higher rate)
  - periodic:  tasks cluster around periodic peaks
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

from env import Task


@dataclass
class DistributionSpec:
    type: str = "uniform"
    # uniform
    low: float = 5.0
    high: float = 80.0
    # normal / bimodal
    mean: float = 50.0
    std: float = 20.0
    low_mean: float = 20.0
    high_mean: float = 80.0
    ratio: float = 0.3    # fraction drawn from high_mean component in bimodal
    # exponential / heavy_tail
    scale: float = 50.0


@dataclass
class TraceParams:
    n_tasks: int = 2000
    cpu: DistributionSpec = field(default_factory=lambda: DistributionSpec(
        type="uniform", low=5.0, high=80.0))
    mem: DistributionSpec = field(default_factory=lambda: DistributionSpec(
        type="uniform", low=5.0, high=80.0))
    duration: DistributionSpec = field(default_factory=lambda: DistributionSpec(
        type="exponential", scale=500.0, low=10.0, high=10000.0))
    arrival_pattern: str = "uniform"   # "uniform", "bursty", "periodic"
    total_time_span: float = 86400.0   # seconds
    burst_prob: float = 0.3            # fraction of time in burst mode
    burst_intensity: float = 3.0       # arrival rate multiplier during burst


def _sample(spec: DistributionSpec, n: int, rng: np.random.Generator) -> np.ndarray:
    t = spec.type
    if t == "uniform":
        return rng.uniform(spec.low, spec.high, n)
    elif t == "normal":
        vals = rng.normal(spec.mean, spec.std, n)
        return np.clip(vals, spec.low, spec.high)
    elif t == "bimodal":
        mask = rng.random(n) < spec.ratio
        lo = rng.normal(spec.low_mean, spec.std, n)
        hi = rng.normal(spec.high_mean, spec.std, n)
        return np.clip(np.where(mask, hi, lo), spec.low, spec.high)
    elif t == "exponential":
        vals = rng.exponential(spec.scale, n)
        return np.clip(vals, spec.low, spec.high if spec.high > spec.low else spec.scale * 10)
    elif t == "heavy_tail":
        alpha = 1.5
        vals = (rng.pareto(alpha, n) + 1) * spec.scale
        return np.clip(vals, spec.low, spec.high if spec.high > spec.low else spec.scale * 20)
    else:
        return rng.uniform(spec.low, spec.high, n)


def _arrival_times(n: int, pattern: str, total_time: float,
                   burst_prob: float, burst_intensity: float,
                   rng: np.random.Generator) -> np.ndarray:
    if pattern == "uniform":
        return np.sort(rng.uniform(0, total_time, n))

    elif pattern == "bursty":
        # Two-state Poisson: normal rate vs burst rate
        base_rate = n / total_time
        times = []
        t = 0.0
        while len(times) < n:
            in_burst = rng.random() < burst_prob
            rate = base_rate * (burst_intensity if in_burst else 1.0)
            inter = rng.exponential(1.0 / max(rate, 1e-9))
            t += inter
            times.append(t)
        times = np.array(times[:n])
        # Re-scale to fit total_time_span
        times = times / times[-1] * total_time
        return times

    elif pattern == "periodic":
        n_peaks = 10
        peak_times = np.linspace(0, total_time, n_peaks + 2)[1:-1]
        tasks_per_peak = n // n_peaks
        all_times = []
        peak_std = total_time / n_peaks * 0.08
        for pt in peak_times:
            all_times.extend(rng.normal(pt, peak_std, tasks_per_peak))
        # Fill remainder uniformly
        remainder = n - len(all_times)
        if remainder > 0:
            all_times.extend(rng.uniform(0, total_time, remainder))
        return np.sort(np.clip(np.array(all_times[:n]), 0, total_time))

    else:
        return np.sort(rng.uniform(0, total_time, n))


def generate_synthetic_tasks(params: TraceParams,
                              seed: Optional[int] = None) -> List[Task]:
    """Generate a list of Task objects according to the given distribution params."""
    rng = np.random.default_rng(seed)
    n = params.n_tasks

    cpus = np.clip(_sample(params.cpu, n, rng), 1.0, 100.0)
    mems = np.clip(_sample(params.mem, n, rng), 1.0, 100.0)
    durations = np.clip(_sample(params.duration, n, rng), 10.0, None)
    arrivals = _arrival_times(
        n, params.arrival_pattern, params.total_time_span,
        params.burst_prob, params.burst_intensity, rng,
    )

    tasks = []
    for i in range(n):
        task = Task(
            name=f"syn_{i}",
            start_time=float(arrivals[i]),
            end_time=float(arrivals[i] + durations[i]),
            plan_cpu=float(cpus[i]),
            plan_mem=float(mems[i]),
        )
        tasks.append(task)
    return tasks


def mix_tasks(real_tasks: List[Task], synthetic_tasks: List[Task],
              synthetic_ratio: float,
              seed: Optional[int] = None) -> List[Task]:
    """
    Build a mixed task list of the same size as real_tasks.
    synthetic_ratio=0.0 → pure real data; synthetic_ratio=1.0 → pure synthetic.
    """
    rng = np.random.default_rng(seed)
    n = len(real_tasks)
    n_syn = int(round(n * max(0.0, min(1.0, synthetic_ratio))))
    n_real = n - n_syn

    real_idx = rng.choice(len(real_tasks), size=min(n_real, len(real_tasks)), replace=False)
    real_sample = [real_tasks[i] for i in real_idx]

    if synthetic_tasks and n_syn > 0:
        syn_idx = rng.choice(len(synthetic_tasks), size=n_syn, replace=True)
        syn_sample = [synthetic_tasks[i] for i in syn_idx]
    else:
        # Fall back to more real tasks if no synthetic available
        extra = rng.choice(len(real_tasks), size=n_syn, replace=True)
        syn_sample = [real_tasks[i] for i in extra]

    mixed = real_sample + syn_sample
    mixed.sort(key=lambda t: t.arrive_time)
    return mixed


def dist_spec_from_dict(d: dict) -> DistributionSpec:
    return DistributionSpec(
        type=d.get("type", "uniform"),
        low=float(d.get("low", 5.0)),
        high=float(d.get("high", 80.0)),
        mean=float(d.get("mean", 50.0)),
        std=float(d.get("std", 20.0)),
        low_mean=float(d.get("low_mean", 20.0)),
        high_mean=float(d.get("high_mean", 80.0)),
        ratio=float(d.get("ratio", 0.3)),
        scale=float(d.get("scale", 50.0)),
    )


def trace_params_from_dict(d: dict) -> TraceParams:
    tp = d.get("trace_params", d)
    return TraceParams(
        n_tasks=int(tp.get("n_tasks", 2000)),
        cpu=dist_spec_from_dict(tp.get("cpu", {})),
        mem=dist_spec_from_dict(tp.get("mem", {})),
        duration=dist_spec_from_dict(tp.get("duration", {})),
        arrival_pattern=tp.get("arrival_pattern", "uniform"),
        total_time_span=float(tp.get("total_time_span", 86400.0)),
        burst_prob=float(tp.get("burst_prob", 0.3)),
        burst_intensity=float(tp.get("burst_intensity", 3.0)),
    )
