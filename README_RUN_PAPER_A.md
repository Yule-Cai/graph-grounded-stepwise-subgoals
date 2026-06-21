# Paper A experiments desktop package

This is the full replacement package for Paper A experiments.

## 1. Replace old desktop folder

```zsh
cd ~/Desktop
mv paper_a_experiments_desktop paper_a_experiments_desktop_old_$(date +%Y%m%d_%H%M%S) 2>/dev/null || true
unzip paper_a_experiments_desktop_FULL_REPLACE.zip
cd paper_a_experiments_desktop
chmod +x scripts/*.sh scripts/*.py
```

## 2. Check environment

```zsh
./scripts/paper_a_check_env.sh
```

If `requests` is missing:

```zsh
conda run -n ros_env pip install requests
```

## 3. Smoke test

```zsh
./scripts/run_paper_a_smoke.sh 2 5000000
```

## 4. Overnight normal benchmark

```zsh
export LM_STUDIO_URL=http://127.0.0.1:1234
./scripts/run_paper_a_overnight.sh 100 5000000
```

This runs RL baselines, then LM Studio route planning for:

- liquid/lfm2.5-1.2b
- qwen/qwen3-1.7b
- google/gemma-3-1b

HF local model experiments are not run.

## 5. Hard fixed-case benchmark

```zsh
export LM_STUDIO_URL=http://127.0.0.1:1234
./scripts/run_paper_a_hard_benchmark.sh 5 5000000
```

If the 5-case test succeeds, run:

```zsh
./scripts/run_paper_a_hard_benchmark.sh 100 5000000
```

Hard benchmark logs are saved under:

```text
logs/eval/hard_benchmark_*/
```
