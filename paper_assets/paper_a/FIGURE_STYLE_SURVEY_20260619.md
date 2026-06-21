# Paper A Figure Style Survey, 2026-06-19

Purpose: collect figure-design cues from nearby top-conference / major robotics papers, then translate them into Paper A's own evidence chain. This is a design survey, not a claim that all papers are directly comparable baselines.

## Papers Checked

1. Visual Language Maps for Robot Navigation, ICRA 2023  
   Link: http://arxiv.org/abs/2210.05714  
   Useful pattern: pair a semantic-map system view with compact downstream navigation evidence.

2. LM-Nav: Robotic Navigation with Large Pre-Trained Models of Language, Vision, and Action, CoRL 2023  
   Link: http://arxiv.org/abs/2207.04429  
   Useful pattern: show the planner/controller decomposition visually before quantitative results.

3. SayCan: Do As I Can, Not As I Say, CoRL 2022  
   Link: http://arxiv.org/abs/2204.01691  
   Useful pattern: make the affordance/filtering interface visible, not only the language prompt.

4. Code as Policies, ICRA 2023  
   Link: http://arxiv.org/abs/2209.07753  
   Useful pattern: separate system mechanism, examples, and quantitative evaluation instead of crowding one figure.

5. SayPlan: Grounding Large Language Models using 3D Scene Graphs for Scalable Robot Task Planning, 2023  
   Link: http://arxiv.org/abs/2307.06135  
   Useful pattern: use scene-graph grounding as the visual anchor for language planning.

6. HOV-SG: Hierarchical Open-Vocabulary 3D Scene Graphs for Language-Grounded Robot Navigation, 2024  
   Link: http://arxiv.org/abs/2403.17846  
   Useful pattern: make hierarchy and graph grounding explicit; use qualitative map panels sparingly.

7. VLFM: Vision-Language Frontier Maps for Zero-Shot Semantic Navigation, 2023  
   Link: http://arxiv.org/abs/2312.03275  
   Useful pattern: frontier/map figures work best when paired with one clear metric panel, not a large table alone.

8. ViNT: A Foundation Model for Visual Navigation, CoRL 2023  
   Link: http://arxiv.org/abs/2306.14846  
   Useful pattern: compare policy families with clean grouped summaries and reserve detailed maps for supplement.

9. NoMaD: Goal Masked Diffusion Policies for Navigation and Exploration, CoRL 2023  
   Link: http://arxiv.org/abs/2310.07896  
   Useful pattern: use concise multi-panel figures where each panel answers a different reviewer question.

10. ETPNav: Evolving Topological Planning for Vision-Language Navigation in Continuous Environments, 2023  
    Link: http://arxiv.org/abs/2304.03047  
    Useful pattern: topological route reasoning benefits from graph-style visual summaries and explicit success/failure diagnostics.

## Design Choices Applied to Paper A

- Main paper now uses one evidence-chain figure instead of disconnected single-purpose result plots.
- The main figure answers four questions in order: planning granularity, order sensitivity, semantic success-risk trade-off, and gate conservatism.
- Success and semantic exposure are not plotted on a dual y-axis; the semantic trade-off is shown as a success-vs-cost scatter.
- Appendix figures carry detailed diagnostics: threshold sweep, perturbation heatmap, failure taxonomy, latency/preference checks, and risk-weight sensitivity.
- Captions explicitly state the conservative interpretation: the LLM interface is useful when constrained and gated, but graph routing remains the fallback/default when the cost is known.
