"""
Adaptive Training Loop
======================
LLM-guided curriculum learning for RL-based resource allocation in Open RAN.

The loop:
  1. Train PPO for `phase_timesteps` on the current task distribution.
  2. Analyze the reward curve (trend, improvement %, variance).
  3. Send feedback history to Claude (or rule-based fallback).
  4. Receive a generation strategy: trace distributions + env weight adjustments.
  5. Generate synthetic tasks and mix them with real Alibaba traces.
  6. Update AdaptiveEnv, continue training (weights are NOT reset between phases).
  7. Repeat for `n_phases` phases.

Usage
-----
  python adaptive_loop.py --n_phases 5 --phase_timesteps 20000 --hidden_size 64
  python adaptive_loop.py --n_phases 5 --phase_timesteps 20000 --no-llm   # rule-based
  python adaptive_loop.py --help
"""

import os
import sys
import json
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from argparser import args as env_args
from adaptive.adaptive_env import AdaptiveEnv
from adaptive.feedback_analyzer import analyze_log
from adaptive.trace_generator import (
    generate_synthetic_tasks,
    mix_tasks,
    trace_params_from_dict,
)
from adaptive.llm_advisor import build_advisor, apply_env_adjustments


# ---------------------------------------------------------------------------
# PPO implementation (hidden_size configurable, no module-level side effects)
# ---------------------------------------------------------------------------

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, action_dim), nn.Softmax(dim=-1),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def act(self, state):
        probs = self.actor(state)
        dist = Categorical(probs)
        action = dist.sample()
        return action.detach(), dist.log_prob(action).detach()

    def evaluate(self, state, action):
        probs = self.actor(state)
        dist = Categorical(probs)
        return dist.log_prob(action), self.critic(state), dist.entropy()


class PPO:
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int,
                 lr_actor: float = 3e-4, lr_critic: float = 1e-3,
                 gamma: float = 0.99, K_epochs: int = 40,
                 eps_clip: float = 0.2):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.buffer = RolloutBuffer()
        self.policy = ActorCritic(state_dim, action_dim, hidden_size).to(device)
        self.policy_old = ActorCritic(state_dim, action_dim, hidden_size).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.Adam([
            {"params": self.policy.actor.parameters(),  "lr": lr_actor},
            {"params": self.policy.critic.parameters(), "lr": lr_critic},
        ])
        self.mse = nn.MSELoss()

    def select_action(self, state: np.ndarray) -> int:
        with torch.no_grad():
            s = torch.FloatTensor(state).to(device)
            action, logprob = self.policy_old.act(s)
        self.buffer.states.append(s)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(logprob)
        return action.item()

    def update(self):
        rewards = []
        disc = 0.0
        for r, done in zip(reversed(self.buffer.rewards),
                           reversed(self.buffer.is_terminals)):
            if done:
                disc = 0.0
            disc = r + self.gamma * disc
            rewards.insert(0, disc)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        old_states   = torch.squeeze(torch.stack(self.buffer.states,   dim=0)).detach()
        old_actions  = torch.squeeze(torch.stack(self.buffer.actions,  dim=0)).detach()
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach()

        for _ in range(self.K_epochs):
            logprobs, values, entropy = self.policy.evaluate(old_states, old_actions)
            values = torch.squeeze(values)
            ratios = torch.exp(logprobs - old_logprobs)
            adv = rewards - values.detach()
            surr = torch.min(ratios * adv,
                             torch.clamp(ratios, 1 - self.eps_clip,
                                         1 + self.eps_clip) * adv)
            loss = -surr + 0.5 * self.mse(values, rewards) - 0.01 * entropy
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

    def save(self, path: str):
        torch.save(self.policy_old.state_dict(), path)

    def load(self, path: str):
        sd = torch.load(path, map_location="cpu")
        self.policy_old.load_state_dict(sd)
        self.policy.load_state_dict(sd)


# ---------------------------------------------------------------------------
# Training phase
# ---------------------------------------------------------------------------

def run_phase(agent: PPO, env: AdaptiveEnv,
              phase_timesteps: int, max_ep_len: int,
              update_timestep: int, log_freq: int,
              log_path: str) -> None:
    """
    Train `agent` on `env` for `phase_timesteps` steps.
    Appends episode/timestep/reward rows to `log_path`.
    """
    log_f = open(log_path, "w")
    log_f.write("episode,timestep,reward\n")

    time_step = 0
    i_episode = 0
    log_running_reward = 0.0
    log_running_episodes = 0

    state = env.reset()

    while time_step < phase_timesteps:
        current_ep_reward = 0.0

        for _ in range(1, max_ep_len + 1):
            action = agent.select_action(state)
            state, reward, done, _ = env.step(action)

            agent.buffer.rewards.append(reward)
            agent.buffer.is_terminals.append(done)

            time_step += 1
            current_ep_reward += reward

            if time_step % update_timestep == 0:
                agent.update()

            if time_step % log_freq == 0 and log_running_episodes > 0:
                avg = log_running_reward / log_running_episodes
                log_f.write(f"{i_episode},{time_step},{avg:.6f}\n")
                log_f.flush()
                log_running_reward = 0.0
                log_running_episodes = 0

            if time_step >= phase_timesteps:
                break
            if done:
                break

        log_running_reward += current_ep_reward
        log_running_episodes += 1
        i_episode += 1

        if time_step >= phase_timesteps:
            break

    log_f.close()


# ---------------------------------------------------------------------------
# Adaptive loop
# ---------------------------------------------------------------------------

def adaptive_loop(
    n_phases: int,
    phase_timesteps: int,
    hidden_size: int,
    use_llm: bool,
    llm_model: str,
    output_dir: str,
    seed: int,
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Directories
    os.makedirs(output_dir, exist_ok=True)
    log_dir = os.path.join(output_dir, "logs")
    model_dir = os.path.join(output_dir, "models")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Environment & agent dimensions
    state_dim  = env_args.n_servers * env_args.n_resources + env_args.n_resources + 1
    action_dim = env_args.n_servers
    base_w1 = env_args.w1
    base_w2 = env_args.w2

    # PPO hyperparameters
    max_ep_len       = 200
    update_timestep  = max_ep_len * 4
    log_freq         = max_ep_len * 2

    env = AdaptiveEnv()
    agent = PPO(state_dim, action_dim, hidden_size)
    advisor = build_advisor(use_llm=use_llm, model=llm_model)

    real_tasks = env.get_real_tasks()
    feedback_history = []
    strategy_log_path = os.path.join(output_dir, "strategy_log.jsonl")

    print(f"\n{'='*70}")
    print(f"Adaptive Training | {n_phases} phases × {phase_timesteps} steps")
    print(f"Algorithm: PPO (hidden={hidden_size})")
    print(f"Advisor:   {'LLM (Claude)' if use_llm else 'rule-based fallback'}")
    print(f"Output:    {output_dir}")
    print(f"{'='*70}\n")

    for phase in range(n_phases):
        print(f"\n--- Phase {phase} ---")

        # Step 1: Ask advisor for strategy (skip phase 0 – use pure real data)
        if phase == 0:
            strategy = {
                "reasoning": "Phase 0: warm-start on real Alibaba traces.",
                "strategy_name": "baseline_real",
                "trace_params": {
                    "n_tasks": 2000,
                    "cpu":      {"type": "uniform", "low": 5,  "high": 80},
                    "mem":      {"type": "uniform", "low": 5,  "high": 80},
                    "duration": {"type": "exponential", "scale": 500,
                                 "low": 10, "high": 10000},
                    "arrival_pattern": "uniform",
                    "total_time_span": 86400.0,
                    "burst_prob": 0.3,
                    "burst_intensity": 3.0,
                },
                "env_adjustments": {"w1_scale": 1.0, "w2_scale": 1.0},
                "synthetic_ratio": 0.0,
                "notes": "Baseline phase.",
            }
        else:
            print(f"  Consulting advisor with {len(feedback_history)} phase(s) of history...")
            strategy = advisor.get_strategy(feedback_history)

        strategy_name = strategy.get("strategy_name", f"phase_{phase}")
        synthetic_ratio = float(strategy.get("synthetic_ratio", 0.0))
        print(f"  Strategy: {strategy_name} | synthetic_ratio={synthetic_ratio:.2f}")
        print(f"  Reasoning: {strategy.get('reasoning', '')}")

        # Log strategy
        with open(strategy_log_path, "a") as sf:
            entry = {"phase": phase, **strategy}
            sf.write(json.dumps(entry) + "\n")

        # Step 2: Build task list
        if synthetic_ratio > 0.0:
            trace_params = trace_params_from_dict(strategy)
            synthetic_tasks = generate_synthetic_tasks(trace_params, seed=seed + phase)
            tasks = mix_tasks(real_tasks, synthetic_tasks,
                              synthetic_ratio=synthetic_ratio,
                              seed=seed + phase)
            print(f"  Tasks: {len(tasks)} total "
                  f"({int(len(tasks)*(1-synthetic_ratio))} real + "
                  f"{int(len(tasks)*synthetic_ratio)} synthetic)")
        else:
            tasks = list(real_tasks)
            print(f"  Tasks: {len(tasks)} (real only)")

        # Step 3: Apply env adjustments
        apply_env_adjustments(env, strategy.get("env_adjustments", {}),
                              base_w1, base_w2)
        env.set_tasks(tasks)

        # Step 4: Train
        log_path = os.path.join(log_dir, f"phase_{phase:03d}.csv")
        print(f"  Training for {phase_timesteps} timesteps -> {log_path}")
        run_phase(agent, env, phase_timesteps, max_ep_len,
                  update_timestep, log_freq, log_path)

        # Step 5: Analyze feedback
        fb = analyze_log(log_path, phase=phase, algorithm="PPO",
                         strategy_used=strategy_name)
        feedback_history.append(fb)
        print(f"  {fb.summary()}")

        # Step 6: Save checkpoint
        ckpt = os.path.join(model_dir, f"PPO{hidden_size}_phase_{phase:03d}.pth")
        agent.save(ckpt)
        print(f"  Checkpoint saved -> {ckpt}")

    # Final model
    final_path = os.path.join(model_dir,
                               f"PPO{hidden_size}_adaptive_final.pth")
    agent.save(final_path)
    print(f"\nTraining complete. Final model: {final_path}")

    # Summary
    print("\n=== Phase Summary ===")
    for fb in feedback_history:
        print(f"  {fb.summary()}")

    print(f"\nStrategy log: {strategy_log_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Adaptive LLM-guided curriculum training for RL resource allocation"
    )
    parser.add_argument("--n_phases", type=int, default=5,
                        help="Number of training phases (default: 5)")
    parser.add_argument("--phase_timesteps", type=int, default=20000,
                        help="Timesteps per phase (default: 20000)")
    parser.add_argument("--hidden_size", type=int, default=64,
                        help="PPO hidden layer size (default: 64)")
    parser.add_argument("--no-llm", dest="use_llm", action="store_false",
                        help="Disable Claude API; use rule-based fallback")
    parser.add_argument("--llm_model", type=str, default="gpt-4o",
                        help="OpenAI model ID (default: gpt-4o)")
    parser.add_argument("--output_dir", type=str,
                        default=f"adaptive_runs/run_{datetime.now():%Y%m%d_%H%M%S}",
                        help="Directory for logs, models, and strategy log")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (default: 0)")
    cli = parser.parse_args()

    adaptive_loop(
        n_phases=cli.n_phases,
        phase_timesteps=cli.phase_timesteps,
        hidden_size=cli.hidden_size,
        use_llm=cli.use_llm,
        llm_model=cli.llm_model,
        output_dir=cli.output_dir,
        seed=cli.seed,
    )
