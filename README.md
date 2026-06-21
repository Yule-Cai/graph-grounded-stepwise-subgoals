# Graph-Grounded Stepwise Navigation Artifact

This anonymous repository contains the code artifact for a diagnostic study of
edge-scale language models as graph-grounded high-level subgoal selectors for
long-horizon robot navigation.

The artifact is intentionally code-only. It does not include article PDFs,
article source files, author information, trained controller checkpoints, full
runtime logs, local caches, or API keys.

## What This Repository Provides

The repository supports four reviewer-facing questions:

1. Does asking a local LLM for one legal next subgoal produce more executable
   routes than asking it to generate a full route?
2. How much of the observed behavior is explained by graph structure and
   candidate ordering?
3. Can order instability be used as a conservative signal for graph-search
   fallback?
4. How do the main results change under multi-layout pressure tests, hybrid
   language-to-cost interfaces, and local model latency measurements?

The code is organized so that the route layer, experiment runners, summaries,
and figure/table generation can be inspected directly from the repository.

## Repository Layout

```text
graph-grounded-stepwise-nav-artifact/
├── paper_a_experiments_desktop/
│   ├── src/                         # Python package and training/evaluation code
│   ├── scripts/                     # helper scripts used by the desktop experiments
│   ├── experiments/                 # case files and experiment definitions
│   ├── generated_worlds/            # lightweight generated Gazebo world assets
│   ├── models/README_MODELS.md      # checkpoint policy note
│   └── requirements-macos.txt       # Python-side dependency list
├── paper_assets/paper_a/
│   ├── *.zsh                        # one-command experiment runners
│   ├── *.py                         # summarization and figure-generation scripts
│   └── *.csv / *.md                 # lightweight aggregate summaries
├── results/
│   ├── *_summary*.csv               # compact aggregate result tables
│   └── *.md                         # generated experiment reports
├── ANONYMITY_CHECKLIST.md
└── README.md
```

## Environment

The original experiments were run on macOS with:

- Python 3
- ROS 2 / Gazebo tooling for the projected indoor navigation setup
- Stable-Baselines3 controllers
- LM Studio serving local LLMs through an OpenAI-compatible local endpoint

Install the Python-side dependencies from the experiment directory:

```bash
cd paper_a_experiments_desktop
python -m pip install -r requirements-macos.txt
```

Some full reruns require ROS 2/Gazebo and locally served LLMs. The repository
therefore includes compact summary outputs under `results/` so that reviewers
can inspect the completed aggregate evidence without rerunning every model.

## Main Entry Points

Run commands from the repository root unless noted otherwise.

### Final LM Studio model sweep

```bash
zsh paper_assets/paper_a/run_final_aaai_experiments.zsh 100 5000000
```

Common environment variables:

```bash
OUT_DIR="paper_assets/paper_a/rerun_logs/final_run" \
MODEL_LIST="liquid/lfm2.5-1.2b nvidia/nemotron-3-nano-4b" \
PLANNER_MODES="llm_step llm_raw" \
SCENARIOS="long_horizon semantic_constraint" \
zsh paper_assets/paper_a/run_final_aaai_experiments.zsh 100 5000000
```

### Supplemental reviewer checks

```bash
zsh paper_assets/paper_a/run_supplemental_reviewer_experiments.zsh 100 5000000
```

This runner covers retry-on-invalid and candidate-feature ablation diagnostics.

### Multi-layout pressure test

```bash
zsh paper_assets/paper_a/run_multimap_generalization_experiments.zsh 100 5000000
```

This evaluates the stepwise interface and no-LLM graph route across multiple
projected Gazebo layouts and seeds.

### Preference and hybrid-interface diagnostics

```bash
zsh paper_assets/paper_a/run_preference_250_upgrade.zsh 250 5000000
```

This covers `language_to_cost` and `route_option_rank` style interfaces.

## Regenerating Summaries and Figures

The scripts in `paper_assets/paper_a/` consume completed logs or the compact
summary CSV files and produce publication-style aggregate tables/figures.
Useful starting points include:

```bash
python paper_assets/paper_a/summarize_revision_experiments.py
python paper_assets/paper_a/generate_story_figures_20260619.py
python paper_assets/paper_a/generate_preference_extension_figure.py
```

The figure-generation scripts are included for auditability. Generated outputs
are written to `artifacts/generated_figures/` by default.

## Included Result Summaries

The `results/` directory contains compact aggregate outputs used to audit the
main empirical claims, including:

- LM Studio final model sweep summaries
- candidate-order gate threshold summaries
- deterministic scorer and graph-route diagnostics
- multi-layout generalization summaries
- local-model latency summaries
- preference and hybrid-interface summaries
- failure-taxonomy summaries

Full per-episode logs are excluded to keep the artifact lightweight and to avoid
local-path leakage. The run scripts regenerate those logs when the required
environment and local models are available.

## Anonymity and Scope

This package is scoped to the graph-grounded stepwise navigation project only.
It excludes unrelated projects and article source files. The included
`ANONYMITY_CHECKLIST.md` summarizes the anonymization checks performed before
packaging.
