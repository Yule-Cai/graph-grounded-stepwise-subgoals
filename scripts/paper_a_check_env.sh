#!/usr/bin/env zsh
set -eo pipefail
source "${0:A:h}/_paper_a_common.sh"
print_header "Paper A environment check"

run_py - <<'PY'
import os, sys, importlib.util
print('Python executable:', sys.executable)
print('Working directory:', os.getcwd())
print('LLM_RL_NAV_HOME:', os.environ.get('LLM_RL_NAV_HOME'))
print('First sys.path entries:')
for p in sys.path[:6]:
    print('  ', p)

import llm_rl_nav
print('llm_rl_nav import: OK ->', llm_rl_nav.__file__)

mods = ['numpy', 'gymnasium', 'stable_baselines3', 'torch', 'requests']
for m in mods:
    spec = importlib.util.find_spec(m)
    print(f'{m}:', 'OK' if spec else 'MISSING')

required_models = [
    'models/PPO/ppo_reference_family_flat_goalfirst_5000000_from_scratch.zip',
    'models/SAC/sac_reference_family_flat_goalfirst_5000000_from_scratch.zip',
    'models/A2C/a2c_reference_family_flat_goalfirst_5000000_from_scratch.zip',
    'models/DQN/dqn_reference_family_flat_goalfirst_5000000_from_scratch.zip',
]
for f in required_models:
    print(f, 'OK' if os.path.exists(f) else 'MISSING')
PY
