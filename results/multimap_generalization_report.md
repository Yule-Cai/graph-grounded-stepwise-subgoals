# Paper A Final Experiment Report

Generated: 2026-06-13T03:15:39
Run directory: `<WORKSPACE>/paper_assets/paper_a/rerun_logs/multimap_fastpass_20260604`
Episodes per job requested: `100`
Controller timesteps: `5000000`
Primary model: `nvidia/nemotron-3-nano-4b`
Model sweep: `nvidia/nemotron-3-nano-4b`
LM Studio models visible at start: not recorded

## Reviewer-Facing Experiment Plan

- Main method: `llm_step`, where the local LLM chooses one legal next subgoal at a time from the structured topological map.
- Failure-mode baseline: `llm_raw`, where the LLM directly proposes a full route without repair.
- Classical baseline: `no_llm`, which removes language planning and uses the deterministic graph route.
- Controller check: `sac` rows test whether the high-level route interface depends only on PPO.

## Best Stepwise LLM Rows

| Scenario | Algo | Mode | Model | Episodes | Success % | 95% CI | Strict-valid % | Parse-ok % | Collision % | Timeout % | Semantic cost |
|---|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| long_horizon | ppo | llm_step | liquid_lfm2.5-1.2b | 100 | 72.0 | [62.5, 79.9] | 95.0 | 95.0 | 2.0 | 26.0 | 0.000 |
| semantic_constraint | ppo | llm_step | liquid_lfm2.5-1.2b | 100 | 77.0 | [67.8, 84.2] | 64.0 | 94.0 | 1.0 | 21.0 | 0.063 |

## Stepwise LLM Model Sweep

| Scenario | Algo | Mode | Model | Episodes | Success % | 95% CI | Strict-valid % | Parse-ok % | Collision % | Timeout % | Semantic cost |
|---|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| long_horizon | ppo | llm_step | liquid_lfm2.5-1.2b | 100 | 72.0 | [62.5, 79.9] | 95.0 | 95.0 | 2.0 | 26.0 | 0.000 |
| long_horizon | ppo | llm_step | nvidia_nemotron-3-nano-4b | 100 | 0.0 | [0.0, 3.7] | 0.0 | 0.0 | 0.0 | 0.0 | 0.000 |
| semantic_constraint | ppo | llm_step | liquid_lfm2.5-1.2b | 100 | 77.0 | [67.8, 84.2] | 64.0 | 94.0 | 1.0 | 21.0 | 0.063 |
| semantic_constraint | ppo | llm_step | nvidia_nemotron-3-nano-4b | 100 | 0.0 | [0.0, 3.7] | 0.0 | 0.0 | 0.0 | 0.0 | 0.000 |

## Raw Full-Route LLM Baseline

No rows available yet.

## No-LLM Baseline

| Scenario | Algo | Mode | Model | Episodes | Success % | 95% CI | Strict-valid % | Parse-ok % | Collision % | Timeout % | Semantic cost |
|---|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| long_horizon | ppo | no_llm | no_llm | 100 | 86.0 | [77.9, 91.5] | 100.0 | 0.0 | 1.0 | 13.0 | 0.000 |
| semantic_constraint | ppo | no_llm | no_llm | 100 | 88.0 | [80.2, 93.0] | 99.0 | 0.0 | 1.0 | 11.0 | 0.023 |

## Writing Notes

- Use `llm_step` as the proposed method, not the old repair-heavy route generation pipeline.
- Do not claim that the LLM universally beats graph search. The stronger claim is that planning granularity determines whether small local LLMs can produce executable semantic subgoal chains.
- In the paper, emphasize `llm_raw` failures as evidence against one-shot full-route generation and `no_llm` as a strong non-language reference point.
- If semantic-constraint success is weaker than long-horizon success, describe it as a limitation and use semantic cost / violation rate carefully.

