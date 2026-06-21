# Paper A Supplemental Reviewer Experiment Report

Generated for the current AAAI draft after the 2026-06-03 supplemental runs.

## Scope

These experiments belong to Paper A only:

**Edge-scale LLM receding-horizon subgoal planning for long-horizon semantic robot navigation.**

They should not be mixed with unrelated projects, claims, or experimental assets.

## Data Directories

Main LM Studio sweep:

```text
paper_assets/paper_a/rerun_logs/final_aaai_stepwise_20260602_050932
```

Supplemental heuristic and retry checks:

```text
paper_assets/paper_a/rerun_logs/supplemental_reviewer_20260603
```

Supplemental candidate-feature ablations:

```text
paper_assets/paper_a/rerun_logs/supplemental_reviewer_20260603_ablation
```

## Main Interpretation

The supplemental results make the paper more defensible, but they also require a careful claim. The LLM should not be described as replacing graph search. Instead, the evidence supports this claim:

> Stepwise LLM planning is a constrained, auditable route-decision interface. It makes small local LLMs usable for legal next-subgoal selection and is much more reliable than raw full-route generation, but semantic-risk handling remains a success-safety trade-off.

## Supplemental Baselines

All rows use PPO execution and 100 episodes per condition.

| Scenario | Mode | Model | Success % | Strict % | Parse % | Collision % | Timeout % | Semantic cost |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Long | graph_shortest | no LLM | 81.0 | -- | -- | 10.0 | 9.0 | 0.000 |
| Long | greedy_hop | no LLM | 78.0 | 100.0 | 100.0 | 13.0 | 9.0 | 0.000 |
| Long | greedy_progress | no LLM | 64.0 | 86.0 | 86.0 | 14.0 | 22.0 | 0.000 |
| Long | greedy_risk | no LLM | 78.0 | 100.0 | 100.0 | 13.0 | 9.0 | 0.000 |
| Long | random_legal | no LLM | 13.0 | 33.0 | 33.0 | 17.0 | 70.0 | 0.000 |
| Long | llm_step_retry | LFM2.5-1.2B | 75.0 | 98.0 | 98.0 | 16.0 | 9.0 | 0.000 |
| Long | llm_step_retry | Nemotron-3-Nano-4B | 75.0 | 98.0 | 94.0 | 14.0 | 11.0 | 0.000 |
| Semantic | graph_shortest | no LLM | 88.0 | -- | -- | 4.0 | 8.0 | 0.164 |
| Semantic | greedy_hop | no LLM | 79.0 | 15.0 | 98.0 | 13.0 | 8.0 | 0.151 |
| Semantic | greedy_progress | no LLM | 61.0 | 10.0 | 82.0 | 17.0 | 22.0 | 0.122 |
| Semantic | greedy_risk | no LLM | 44.0 | 61.0 | 68.0 | 11.0 | 45.0 | 0.014 |
| Semantic | random_legal | no LLM | 17.0 | 4.0 | 39.0 | 20.0 | 63.0 | 0.047 |
| Semantic | llm_step_retry | LFM2.5-1.2B | 65.0 | 31.0 | 85.0 | 21.0 | 14.0 | 0.098 |
| Semantic | llm_step_retry | Nemotron-3-Nano-4B | 75.0 | 15.0 | 95.0 | 11.0 | 14.0 | 0.144 |

Key reading:

- Distance-only graph shortest has the strongest semantic success, 88.0%, but the highest semantic cost, 0.164.
- Risk-greedy has the lowest semantic cost, 0.014, but only 44.0% semantic success.
- Retry improves the two strongest LLM rows modestly. Nemotron retry reaches 75.0% success in both scenarios.
- Random legal-neighbor selection is poor, so high success is not explained by merely choosing any legal adjacent node.

## Candidate-Feature Ablations

| Scenario | Model / ablation | Success % | Strict % | Parse % | Collision % | Timeout % | Semantic cost |
|---|---|---:|---:|---:|---:|---:|---:|
| Long | LFM2.5-1.2B full | 73.0 | 95.0 | 95.0 | 16.0 | 11.0 | 0.000 |
| Long | LFM2.5-1.2B no hop | 70.0 | 95.0 | 95.0 | 17.0 | 13.0 | 0.000 |
| Long | LFM2.5-1.2B no risk | 75.0 | 99.0 | 99.0 | 14.0 | 11.0 | 0.000 |
| Long | LFM2.5-1.2B shuffled | 7.0 | 30.0 | 30.0 | 13.0 | 80.0 | 0.000 |
| Long | Nemotron-3-Nano-4B full | 71.0 | 91.0 | 89.0 | 13.0 | 16.0 | 0.000 |
| Long | Nemotron-3-Nano-4B no hop | 65.0 | 86.0 | 80.0 | 14.0 | 21.0 | 0.000 |
| Long | Nemotron-3-Nano-4B no risk | 74.0 | 94.0 | 90.0 | 12.0 | 14.0 | 0.000 |
| Long | Nemotron-3-Nano-4B shuffled | 20.0 | 46.0 | 45.0 | 13.0 | 67.0 | 0.000 |
| Semantic | LFM2.5-1.2B full | 64.0 | 31.0 | 84.0 | 21.0 | 15.0 | 0.099 |
| Semantic | LFM2.5-1.2B no hop | 60.0 | 32.0 | 84.0 | 20.0 | 20.0 | 0.090 |
| Semantic | LFM2.5-1.2B no risk | 75.0 | 13.0 | 94.0 | 15.0 | 10.0 | 0.144 |
| Semantic | LFM2.5-1.2B shuffled | 14.0 | 20.0 | 43.0 | 17.0 | 69.0 | 0.038 |
| Semantic | Nemotron-3-Nano-4B full | 72.0 | 16.0 | 86.0 | 9.0 | 16.0 | 0.143 |
| Semantic | Nemotron-3-Nano-4B no hop | 67.0 | 13.0 | 74.0 | 12.0 | 18.0 | 0.125 |
| Semantic | Nemotron-3-Nano-4B no risk | 17.0 | 2.0 | 18.0 | 5.0 | 54.0 | 0.075 |
| Semantic | Nemotron-3-Nano-4B shuffled | 13.0 | 4.0 | 23.0 | 18.0 | 54.0 | 0.049 |

Key reading:

- Candidate order is critical. Shuffling reduces LFM2.5-1.2B from 73.0% to 7.0% in long-horizon episodes and reduces Nemotron from 71.0% to 20.0%.
- Hop distance is useful but not the only factor.
- Removing risk is unstable. It can improve raw success for one model while increasing semantic cost, and it can collapse another model. This supports a limitation statement rather than a safety claim.

## Paper Placement

The main paper now includes a compact supplemental-check table and a short interpretation paragraph. The supplement contains the detailed tables above.

Recommended final language:

- Use: "The LLM layer is an auditable local route-decision interface."
- Use: "The heuristic baselines expose a success-risk trade-off."
- Avoid: "The LLM beats graph planning."
- Avoid: "Semantic safety is solved."
