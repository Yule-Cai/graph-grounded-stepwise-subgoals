# Paper A Experiment Update, 2026-06-19

This note summarizes the completed graph-perturbation and order-gate threshold-sweep experiments.

## Completion

- Graph perturbation LLM-gate: complete, 24/24 conditions, 2400 episodes.
- Order-gate threshold sweep: complete, 14/14 conditions, 1400 episodes.
- Failure taxonomy: regenerated after the new runs.
- Active experiment processes: none.
- LM Studio loaded models: none reported by `lms ps`.

## Key Output Paths

- Graph perturbation summary:
  `<WORKSPACE>/paper_assets/paper_a/rerun_logs/graph_perturbation_llm_gate_20260618/graph_perturbation_summary_with_ci.csv`
- Order-gate threshold summary:
  `<WORKSPACE>/paper_assets/paper_a/rerun_logs/order_gate_threshold_sweep_20260618/order_gate_threshold_sweep_summary_with_ci.csv`
- Failure taxonomy report:
  `<WORKSPACE>/paper_assets/paper_a/failure_taxonomy_after_gate_sweep/failure_taxonomy_report.md`
- Failure taxonomy figure:
  `<WORKSPACE>/paper_assets/paper_a/failure_taxonomy_after_gate_sweep/failure_taxonomy_stacked.pdf`

## Order-Gate Threshold Sweep

This is the strongest new evidence for the paper because it directly addresses candidate-order sensitivity.

On the reference family layout:

| Scenario | Method | Success | Strict validity | Parse OK | Semantic cost |
|---|---:|---:|---:|---:|---:|
| Long horizon | No-LLM graph | 81% | 100% | 0% | 0.000 |
| Long horizon | Stepwise LLM, canonical | 72% | 96% | 96% | 0.000 |
| Long horizon | Stepwise LLM, shuffled | 8% | 36% | 36% | 0.000 |
| Long horizon | Order ensemble | 72% | 96% | 96% | 0.000 |
| Long horizon | Gate, lenient | 75% | 98% | 76% | 0.000 |
| Long horizon | Gate, balanced | 77% | 97% | 66% | 0.000 |
| Long horizon | Gate, strict | 81% | 100% | 41% | 0.000 |
| Semantic constraint | No-LLM graph | 78% | 71% | 0% | 0.082 |
| Semantic constraint | Stepwise LLM, canonical | 63% | 28% | 80% | 0.095 |
| Semantic constraint | Stepwise LLM, shuffled | 13% | 21% | 37% | 0.026 |
| Semantic constraint | Order ensemble | 57% | 31% | 78% | 0.090 |
| Semantic constraint | Gate, lenient | 69% | 61% | 75% | 0.076 |
| Semantic constraint | Gate, balanced | 73% | 67% | 61% | 0.077 |
| Semantic constraint | Gate, strict | 76% | 71% | 32% | 0.081 |

Interpretation:

- Shuffling candidate order causes severe degradation: 72% to 8% on long horizon, and 63% to 13% on semantic constraint.
- The consistency gate recovers performance by treating order instability as an uncertainty signal and falling back to the graph route.
- Strict gate is the cleanest main-table setting: it matches No-LLM graph on long horizon success and strict validity, and nearly matches it on semantic constraint while retaining the LLM-interface diagnosis.
- Balanced gate is useful for a semantic-exposure trade-off paragraph: it reduces semantic cost from 0.082 to 0.077, but with lower success than the graph route.

Recommended paper use:

- Put canonical, shuffled, no-LLM graph, and strict gate in the main ablation table.
- Put lenient/balanced/strict threshold sweep in the appendix.
- Do not claim the gate proves that LLM decisions dominate the route. Claim that the gate converts order sensitivity into a measurable uncertainty signal and restores executable behavior through graph fallback.

## Graph Perturbation LLM-Gate

Across three maps, two scenarios, and four perturbation settings:

| Scenario | Episodes | Gate success | Gate strict validity | Gate semantic cost |
|---|---:|---:|---:|---:|
| Long horizon | 1200 | 65.7% | 87.3% | 0.000 |
| Semantic constraint | 1200 | 61.5% | 66.7% | 0.051 |

Against matched no-LLM graph baselines, the result is mixed but useful:

- On long-horizon perturbations, the gate averages 65.7% success and is roughly comparable to the no-LLM graph route on the same maps.
- On semantic constraints, the gate is slightly below no-LLM graph success overall, but tends to reduce semantic exposure on the reference family map.
- Some unseen-map semantic perturbations favor the graph route in success, which should be framed as an honest limitation rather than hidden.

Recommended paper use:

- Use this as robustness evidence, not as a headline win.
- Write that graph-gated LLM planning remains stable enough under edge-drop and risk-noise perturbations to preserve the main diagnostic conclusion.
- Keep a limitation sentence: on harder semantic layouts, the graph route remains a stronger default, while the gate mainly offers an interpretable safety interface and uncertainty signal.

## Failure Taxonomy

The regenerated taxonomy covers 17,186 episodes and 80 grouped rows.

Overall episode taxonomy:

- Success: 55.0%
- Missing goal: 19.7%
- Timeout: 11.6%
- Collision: 7.5%
- Semantic plan violation: 3.2%
- Missing edge: 3.0%
- Parse/selection: near 0.0%

Writing use:

- Raw full-route LLM failures are dominated by route-interface failures, especially missing-goal routes.
- Stepwise/gated methods shift failure modes away from malformed high-level routes and toward downstream controller outcomes such as timeout/collision.
- This supports the story that the proposed interface changes the failure surface from invalid symbolic planning to executable but still controller-limited navigation.

## Recommended Story Update

The strongest final story is:

Edge-scale LLMs should not be treated as standalone global planners. In long-horizon semantic navigation, raw route generation fails mostly through route-interface errors. A graph-grounded stepwise interface makes LLM outputs executable, but canonical ordering is a major confound. An order-consistency gate turns this confound into an uncertainty signal: when LLM choices are unstable across candidate permutations, the system falls back to deterministic graph routing. This produces a conservative planner whose contribution is not beating graph search everywhere, but making the LLM interface measurable, executable, and safer to deploy.

## Manuscript Action Items

- Main table: add or update order sensitivity / gate ablation using the strict gate row.
- Appendix table: include full threshold sweep.
- Robustness subsection: summarize graph perturbation results across 2400 episodes.
- Limitations: state that deterministic graph routing remains stronger on some semantic layouts and that the LLM is most useful as a constrained, uncertainty-gated interface rather than a replacement for graph search.
- Figure: update failure taxonomy figure from `failure_taxonomy_after_gate_sweep`.
