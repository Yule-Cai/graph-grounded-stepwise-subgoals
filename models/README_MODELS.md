# Model Checkpoints

Trained controller checkpoints are intentionally not included in the anonymous
GitHub artifact. This keeps the repository lightweight and avoids uploading
binary artifacts into the main source package.

To regenerate controllers, use the training scripts under:

```text
paper_a_experiments_desktop/src/llm_rl_nav/llm_rl_nav/training/
```

The reported experiments use controllers trained from scratch for 5M steps and
then held fixed during route-interface evaluation.
