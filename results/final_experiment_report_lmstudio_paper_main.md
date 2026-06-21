# Paper A Final Experiment Report

Generated: 2026-06-03T14:02:35
Run directory: `<WORKSPACE>/paper_assets/paper_a/rerun_logs/final_aaai_stepwise_20260602_050932`
Episodes per job requested: `100`
Controller timesteps: `5000000`
Primary model: `lfm2.5-8b-a1b-mlx`
Model sweep: `lfm2.5-8b-a1b-mlx nvidia/nemotron-3-nano-4b qwen/qwen3-1.7b google/gemma-3-1b liquid/lfm2.5-1.2b google/gemma-4-e4b`
LM Studio models visible at start: lfm2.5-8b-a1b-mlx, nvidia/nemotron-3-nano-4b, qwen/qwen3-1.7b, google/gemma-3-1b, liquid/lfm2.5-1.2b, google/gemma-4-e4b, text-embedding-nomic-embed-text-v1.5

## Reviewer-Facing Experiment Plan

- Main method: `llm_step`, where the local LLM chooses one legal next subgoal at a time from the structured topological map.
- Failure-mode baseline: `llm_raw`, where the LLM directly proposes a full route without repair.
- Classical baseline: `no_llm`, which removes language planning and uses the deterministic graph route.
- Controller check: `sac` rows test whether the high-level route interface depends only on PPO.

## Best Stepwise LLM Rows

| Scenario | Algo | Mode | Model | Episodes | Success % | 95% CI | Strict-valid % | Parse-ok % | Collision % | Timeout % | Semantic cost |
|---|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| long_horizon | ppo | llm_step | liquid_lfm2.5-1.2b | 100 | 73.0 | [63.6, 80.7] | 95.0 | 95.0 | 16.0 | 11.0 | 0.000 |
| semantic_constraint | ppo | llm_step | nvidia_nemotron-3-nano-4b | 100 | 72.0 | [62.5, 79.9] | 16.0 | 86.0 | 9.0 | 16.0 | 0.143 |

## Stepwise LLM Model Sweep

| Scenario | Algo | Mode | Model | Episodes | Success % | 95% CI | Strict-valid % | Parse-ok % | Collision % | Timeout % | Semantic cost |
|---|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| long_horizon | ppo | llm_step | HuggingFaceTB__SmolLM2-1.7B | 100 | 0.0 | [0.0, 3.7] | 0.0 | 0.0 | 3.0 | 17.0 | 0.000 |
| long_horizon | ppo | llm_step | google_gemma-3-1b | 100 | 13.0 | [7.8, 21.0] | 24.0 | 24.0 | 11.0 | 76.0 | 0.000 |
| long_horizon | ppo | llm_step | google_gemma-4-e4b | 100 | 0.0 | [0.0, 3.7] | 0.0 | 0.0 | 0.0 | 1.0 | 0.000 |
| long_horizon | ppo | llm_step | lfm2.5-8b-a1b-mlx | 100 | 16.0 | [10.1, 24.4] | 21.0 | 0.0 | 7.0 | 54.0 | 0.000 |
| long_horizon | ppo | llm_step | liquid_lfm2.5-1.2b | 100 | 73.0 | [63.6, 80.7] | 95.0 | 95.0 | 16.0 | 11.0 | 0.000 |
| long_horizon | ppo | llm_step | nvidia_nemotron-3-nano-4b | 100 | 71.0 | [61.5, 79.0] | 91.0 | 89.0 | 13.0 | 16.0 | 0.000 |
| long_horizon | ppo | llm_step | qwen_qwen3-1.7b | 100 | 2.0 | [0.6, 7.0] | 6.0 | 0.0 | 5.0 | 43.0 | 0.000 |
| semantic_constraint | ppo | llm_step | HuggingFaceTB__SmolLM2-1.7B | 100 | 0.0 | [0.0, 3.7] | 0.0 | 0.0 | 1.0 | 10.0 | 0.000 |
| semantic_constraint | ppo | llm_step | google_gemma-3-1b | 100 | 13.0 | [7.8, 21.0] | 2.0 | 24.0 | 12.0 | 75.0 | 0.029 |
| semantic_constraint | ppo | llm_step | google_gemma-4-e4b | 100 | 0.0 | [0.0, 3.7] | 0.0 | 0.0 | 0.0 | 0.0 | 0.000 |
| semantic_constraint | ppo | llm_step | lfm2.5-8b-a1b-mlx | 100 | 26.0 | [18.4, 35.4] | 3.0 | 2.0 | 3.0 | 53.0 | 0.090 |
| semantic_constraint | ppo | llm_step | liquid_lfm2.5-1.2b | 100 | 64.0 | [54.2, 72.7] | 31.0 | 84.0 | 21.0 | 15.0 | 0.099 |
| semantic_constraint | ppo | llm_step | nvidia_nemotron-3-nano-4b | 100 | 72.0 | [62.5, 79.9] | 16.0 | 86.0 | 9.0 | 16.0 | 0.143 |
| semantic_constraint | ppo | llm_step | qwen_qwen3-1.7b | 100 | 0.0 | [0.0, 3.7] | 0.0 | 0.0 | 1.0 | 51.0 | 0.005 |

## Raw Full-Route LLM Baseline

| Scenario | Algo | Mode | Model | Episodes | Success % | 95% CI | Strict-valid % | Parse-ok % | Collision % | Timeout % | Semantic cost |
|---|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| long_horizon | ppo | llm_raw | google_gemma-3-1b | 100 | 39.0 | [30.0, 48.8] | 0.0 | 78.0 | 39.0 | 22.0 | 0.000 |
| long_horizon | ppo | llm_raw | google_gemma-4-e4b | 100 | 0.0 | [0.0, 3.7] | 0.0 | 0.0 | 16.0 | 23.0 | 0.000 |
| long_horizon | ppo | llm_raw | lfm2.5-8b-a1b-mlx | 100 | 3.0 | [1.0, 8.5] | 0.0 | 0.0 | 44.0 | 53.0 | 0.000 |
| long_horizon | ppo | llm_raw | liquid_lfm2.5-1.2b | 100 | 2.0 | [0.6, 7.0] | 0.0 | 91.0 | 39.0 | 59.0 | 0.000 |
| long_horizon | ppo | llm_raw | nvidia_nemotron-3-nano-4b | 100 | 1.0 | [0.2, 5.4] | 1.0 | 1.0 | 48.0 | 51.0 | 0.000 |
| long_horizon | ppo | llm_raw | qwen_qwen3-1.7b | 100 | 0.0 | [0.0, 3.7] | 0.0 | 0.0 | 44.0 | 56.0 | 0.000 |
| semantic_constraint | ppo | llm_raw | google_gemma-3-1b | 100 | 38.0 | [29.1, 47.8] | 0.0 | 79.0 | 42.0 | 20.0 | 0.095 |
| semantic_constraint | ppo | llm_raw | google_gemma-4-e4b | 100 | 3.0 | [1.0, 8.5] | 0.0 | 0.0 | 51.0 | 46.0 | 0.120 |
| semantic_constraint | ppo | llm_raw | lfm2.5-8b-a1b-mlx | 100 | 6.0 | [2.8, 12.5] | 0.0 | 0.0 | 48.0 | 46.0 | 0.108 |
| semantic_constraint | ppo | llm_raw | liquid_lfm2.5-1.2b | 100 | 4.0 | [1.6, 9.8] | 0.0 | 70.0 | 44.0 | 52.0 | 0.046 |
| semantic_constraint | ppo | llm_raw | nvidia_nemotron-3-nano-4b | 100 | 3.0 | [1.0, 8.5] | 1.0 | 2.0 | 53.0 | 44.0 | 0.104 |
| semantic_constraint | ppo | llm_raw | qwen_qwen3-1.7b | 100 | 0.0 | [0.0, 3.7] | 0.0 | 0.0 | 51.0 | 49.0 | 0.082 |

## No-LLM Baseline

| Scenario | Algo | Mode | Model | Episodes | Success % | 95% CI | Strict-valid % | Parse-ok % | Collision % | Timeout % | Semantic cost |
|---|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| long_horizon | ppo | no_llm | no_llm | 100 | 81.0 | [72.2, 87.5] | 0.0 | 0.0 | 10.0 | 9.0 | 0.000 |
| long_horizon | sac | no_llm | no_llm | 100 | 49.0 | [39.4, 58.7] | 0.0 | 0.0 | 13.0 | 38.0 | 0.000 |
| semantic_constraint | ppo | no_llm | no_llm | 100 | 78.0 | [68.9, 85.0] | 0.0 | 0.0 | 9.0 | 13.0 | 0.082 |
| semantic_constraint | sac | no_llm | no_llm | 100 | 54.0 | [44.3, 63.4] | 0.0 | 0.0 | 13.0 | 33.0 | 0.067 |

## Controller Ablation

| Scenario | Algo | Mode | Model | Episodes | Success % | 95% CI | Strict-valid % | Parse-ok % | Collision % | Timeout % | Semantic cost |
|---|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| long_horizon | sac | no_llm | no_llm | 100 | 49.0 | [39.4, 58.7] | 0.0 | 0.0 | 13.0 | 38.0 | 0.000 |
| semantic_constraint | sac | no_llm | no_llm | 100 | 54.0 | [44.3, 63.4] | 0.0 | 0.0 | 13.0 | 33.0 | 0.067 |

## Writing Notes

- Use `llm_step` as the proposed method, not the old repair-heavy route generation pipeline.
- Do not claim that the LLM universally beats graph search. The stronger claim is that planning granularity determines whether small local LLMs can produce executable semantic subgoal chains.
- In the paper, emphasize `llm_raw` failures as evidence against one-shot full-route generation and `no_llm` as a strong non-language reference point.
- If semantic-constraint success is weaker than long-horizon success, describe it as a limitation and use semantic cost / violation rate carefully.

