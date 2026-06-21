# Paper A Desktop Experiment Package

This is a cleaned runnable subset for Paper A:

**Paper A = LLM Route Planning + RL Execution**

It includes:

- `src/llm_rl_nav/` Python package
- `generated_worlds/semantic_3d/` indoor maps
- 5M checkpoints for PPO / SAC / A2C / DQN on `reference_family_flat`
- fixed launcher scripts under `scripts/`
- result folder under `experiments/paper_a/results/`

## Why this package fixes your current error

Your old script ran the module with a fragile relative `PYTHONPATH` inside `conda run`:

```zsh
conda run -n ros_env env PYTHONPATH=src/llm_rl_nav python -m llm_rl_nav.training.eval_multimap_ppo
```

On your machine this caused:

```text
ModuleNotFoundError: No module named 'llm_rl_nav'
```

The new scripts set absolute paths before running Python:

```zsh
export LLM_RL_NAV_HOME="$ROOT"
export PYTHONPATH="$ROOT/src/llm_rl_nav:$ROOT/src:${PYTHONPATH:-}"
conda run --no-capture-output -n ros_env python -m llm_rl_nav.training.eval_multimap_ppo
```

So the project can be placed on Desktop and run from there.

## Mac run commands

Put this folder on Desktop, then:

```zsh
cd ~/Desktop/paper_a_experiments_desktop
chmod +x scripts/*.sh scripts/*.py
./scripts/paper_a_check_env.sh
```

Run a short smoke test first:

```zsh
./scripts/run_paper_a_smoke.sh 2 5000000
```

Then run the Paper A baselines:

```zsh
./scripts/eval_paper_a_baselines.sh 100 5000000
```

Run LM Studio LLM route planning:

```zsh
export LM_STUDIO_URL=http://127.0.0.1:1234
export LM_STUDIO_MODEL=liquid/lfm2.5-1.2b
./scripts/eval_paper_a_lmstudio_routes.sh 100 5000000
```

Alternative model example:

```zsh
export LM_STUDIO_MODEL=google/gemma-3-1b
./scripts/eval_paper_a_lmstudio_routes.sh 100 5000000
```

## If conda environment name is not ros_env

```zsh
CONDA_ENV=your_env_name ./scripts/paper_a_check_env.sh
CONDA_ENV=your_env_name ./scripts/eval_paper_a_baselines.sh 100 5000000
```

## Outputs

Raw logs:

```text
logs/eval/*.log
```

Summary CSV:

```text
experiments/paper_a/results/paper_a_log_summary.csv
```
