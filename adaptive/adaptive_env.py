"""
AdaptiveEnv: extends Env to support custom/synthetic task injection
and dynamic environment parameter adjustment.

Key differences from base Env:
  - set_tasks(tasks): replace the task list without re-loading CSVs
  - set_weights(w1, w2): adjust reward weights per phase
  - step() uses task.arrive_time directly (no dependency on batch_task DataFrame)
  - calc_total_latency() uses self.tasks directly
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from typing import List, Optional

from env import Env, Task, Machine
from argparser import args


class AdaptiveEnv(Env):
    """Env subclass that supports task-list injection and weight adjustment."""

    def __init__(self):
        super().__init__()
        # Cache real tasks loaded from CSV; used as the reference dataset
        self._real_tasks: List[Task] = list(self.tasks)

    # ------------------------------------------------------------------
    # Public API for adaptive loop
    # ------------------------------------------------------------------

    def set_tasks(self, tasks: List[Task]) -> None:
        """Replace the task list used in training episodes."""
        self.tasks = list(tasks)
        self.n_tasks = len(tasks)

    def get_real_tasks(self) -> List[Task]:
        """Return the original Alibaba-trace tasks."""
        return list(self._real_tasks)

    def set_weights(self, w1: Optional[float] = None,
                    w2: Optional[float] = None) -> None:
        """Adjust reward weights for the current phase."""
        if w1 is not None:
            self.w1 = w1
        if w2 is not None:
            self.w2 = w2

    def reset_weights(self) -> None:
        """Restore original weights from argparser."""
        self.w1 = args.w1
        self.w2 = args.w2

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def step(self, action: int):
        """
        Identical to Env.step() but reads arrive_time from the Task object
        instead of self.batch_task DataFrame, allowing synthetic task lists.
        """
        cur_task = self.tasks[self.cur]
        self.cur_time = cur_task.arrive_time   # <-- key change

        done = False
        self.cur += 1

        if self.cur == self.n_tasks:
            self.latency = [t.start_time - t.arrive_time for t in self.tasks]
            for i in range(1, len(self.latency)):
                self.latency[i] = self.latency[i] + self.latency[i - 1]
            done = True
            self.cur = 0

        nxt_task = self.tasks[self.cur]

        for m in self.machines:
            m.process(self.cur_time)
        self.power_usage.append(np.sum([m.power_usage for m in self.machines]))

        self.machines[action].add_task(cur_task)
        return (
            self.get_states(nxt_task),
            self.get_reward(nxt_task),
            done,
            (self.latency, self.power_usage),
        )

    def calc_total_latency(self) -> float:
        """Use self.tasks directly (works for both real and synthetic tasks)."""
        wait = [t.start_time - t.arrive_time for t in self.tasks]
        for i in range(1, len(wait)):
            wait[i] += wait[i - 1]
        return float(np.sum(wait))
