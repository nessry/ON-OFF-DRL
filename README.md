# ON-OFF-DRL: Adaptive LLM-Guided RL for Open RAN Resource Allocation

Research project that applies an **adaptive LLM-guided curriculum training loop** to PPO-based resource allocation in Open Radio Access Network (Open RAN). An LLM observes training feedback after each phase and adjusts the task distribution and reward weights to guide the agent toward faster, more stable convergence.

Tasks are drawn from real Alibaba cluster workload traces and assigned to 10 servers, optimising a trade-off between energy consumption and task completion latency.

---

## Dependencies

```bash
pip install torch numpy pandas matplotlib openai
pip install gurobipy   # optional — only needed for the Optimal MIP baseline in tests/
```

Create a `.env` file at the project root with your OpenAI API key:

```
OPENAI_API_KEY=sk-...
```

The adaptive loop reads this automatically. Without it, pass `--no-llm` to use a rule-based curriculum instead.

---

## Repository Structure

```
ON-OFF-DRL/
│
├── adaptive_loop.py         # Main entry point — adaptive training pipeline
├── adaptive_plot.py         # Visualise adaptive run results vs. baselines
│
├── adaptive/                # Adaptive pipeline modules
│   ├── adaptive_env.py      #   AdaptiveEnv: task-list injection + dynamic reward weights
│   ├── feedback_analyzer.py #   Extracts learning-curve metrics from phase logs
│   ├── llm_advisor.py       #   OpenAI integration; rule-based fallback
│   └── trace_generator.py   #   Synthetic task generation (uniform/bimodal/heavy-tail…)
│
├── env.py                   # Base simulation environment (Alibaba traces, 10 servers)
├── argparser.py             # Environment hyperparameters (n_servers, weights, …)
├── greedy.py                # Greedy baseline (lowest-CPU server)
├── opt.py                   # Optimal MIP baseline via Gurobi
│
├── data/                    # Alibaba cluster workload CSVs
│
└── tests/                   # Pre-computed baseline results
    ├── PPO_files/           #   PPO training reward logs
    ├── ACER_files/          #   ACER training reward logs
    ├── PPO_preTrained/      #   Pre-trained PPO model weights
    ├── ACER_preTrained/     #   Pre-trained ACER model weights
    ├── logs/                #   Power and latency evaluation logs (used by adaptive_plot.py)
    └── plots/               #   Pre-generated baseline figures
```

---

## How the Adaptive Pipeline Works

```
adaptive_loop.py
  │
  ├── env.py / argparser.py          load Alibaba tasks + environment config
  ├── adaptive/adaptive_env.py       wraps Env; supports task-list injection
  │
  │   ┌── Phase loop ──────────────────────────────────────────────────────┐
  │   │                                                                    │
  │   │  1. llm_advisor        ← feedback_analyzer (reward trend metrics)  │
  │   │        │  decides: which task distribution? adjust w1/w2?          │
  │   │        ↓                                                            │
  │   │  2. trace_generator   →  synthetic tasks mixed with real traces    │
  │   │        ↓                                                            │
  │   │  3. AdaptiveEnv.set_tasks() + set_weights()                        │
  │   │        ↓                                                            │
  │   │  4. PPO trains for phase_timesteps steps  →  phase_NNN.csv         │
  │   │        ↓                                                            │
  │   │  5. feedback_analyzer.analyze_log()  →  PhaseFeedback              │
  │   │        └── fed back into llm_advisor for the next phase            │
  │   └────────────────────────────────────────────────────────────────────┘
  │
  └── outputs → adaptive_runs/run_YYYYMMDD_HHMMSS/
                    ├── logs/phase_000.csv, phase_001.csv, …
                    ├── models/PPO64_phase_000.pth, …
                    └── strategy_log.jsonl
```

---

## Running the Adaptive Pipeline

### 1. Train

```bash
# LLM-guided (requires OPENAI_API_KEY in .env)
python adaptive_loop.py --n_phases 5 --phase_timesteps 20000 --hidden_size 64

# Rule-based fallback (no API key needed)
python adaptive_loop.py --n_phases 5 --phase_timesteps 20000 --no-llm

# All options
python adaptive_loop.py --help
```

Each phase trains PPO for `phase_timesteps` steps, logs to `phase_NNN.csv`, saves a model checkpoint, then consults the LLM before the next phase.

### 2. Visualise

```bash
# Auto-detects the latest run under adaptive_runs/
python adaptive_plot.py

# Point at a specific run
python adaptive_plot.py --run_dir adaptive_runs/run_20260318_004407
```

Produces three PDFs in `plots/adaptive/`:

| File | What it shows |
|---|---|
| `phase_curves.pdf` | Per-phase reward curves labelled with LLM strategy name |
| `phase_summary.pdf` | Final reward + intra-phase improvement % per phase |
| `cumulative_vs_baseline.pdf` | Stitched adaptive reward vs. baseline PPO/ACER on a shared timestep axis |

Baseline data for the comparison plot is read from `tests/logs/` (pre-computed).

---

## References

- [higgsfield/RL-Adventure-2](https://github.com/higgsfield/RL-Adventure-2)
- [nikhilbarhate99/PPO-PyTorch](https://github.com/nikhilbarhate99/PPO-PyTorch)
- [gohsyi/cluster_optimization](https://github.com/gohsyi/cluster_optimization)
