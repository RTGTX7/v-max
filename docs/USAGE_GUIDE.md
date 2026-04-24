# Copyright 2026 RTGTX7.

# V-Max Usage Guide

This guide is the shortest practical path for users who need to run the repository without reverse-engineering the codebase.

## 1. What This Repository Contains

This repository is built on **V-Max**, a Waymax-based autonomous-driving research framework. In the current workspace, the main user-facing workflows are:

- training learning-based policies such as PPO, SAC, and BC
- evaluating learned checkpoints or rule-based actors on logged scenarios
- experimenting with structured observations, encoder backbones, and reward configurations
- running reward-search workflows for post-BC or post-stage-1 fine-tuning

The core code remains under `vmax/`. The main entry points are under `vmax/scripts/`.

For documented user workflows, this repository now uses module-style execution:

```bash
python -m vmax.scripts.<group>.<entry_point>
```

## 2. Environment Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/valeoai/v-max
cd v-max

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

If you want a more reproducible Python package set, use the locked environment:

```bash
./setup_env.sh venv
source venv/bin/activate
```

Notes:

- system CUDA, GPU drivers, and JAX-compatible runtime libraries are not managed by `pip`
- the repository currently targets Python `>=3.10`; the README advertises Python 3.11

## 3. Main Repository Entry Points

The most important scripts are:

- `vmax/scripts/training/train.py`: training entry point
- `vmax/scripts/evaluate/evaluate.py`: evaluation entry point
- `vmax/scripts/experiments/reward_search_manager.py`: grid / refine reward search
- `vmax/scripts/experiments/rf_bo_trainer.py`: random-forest surrogate search loop
- `vmax/scripts/experiments/rebuild_trials_from_results.py`: rebuild trial tables from stored outputs

## 4. Minimal Training Workflow

Training is launched through Hydra, with defaults loaded from:

- `vmax/config/base_config.yaml`
- `vmax/config/algorithm/*.yaml`
- `vmax/config/network/*.yaml`
- `vmax/config/network/encoder/*.yaml`

A minimal PPO run looks like this:

```bash
python -m vmax.scripts.training.train \
  algorithm=ppo \
  path_dataset=/path/to/train.tfrecord \
  path_dataset_eval=/path/to/valid.tfrecord
```

Typical overrides:

```bash
python -m vmax.scripts.training.train \
  algorithm=ppo \
  path_dataset=/path/to/train.tfrecord \
  path_dataset_eval=/path/to/valid.tfrecord \
  total_timesteps=200000000 \
  num_envs=64 \
  network/encoder=lq \
  observation_type=vec
```

## 5. Minimal Evaluation Workflow

To evaluate a trained checkpoint:

```bash
python -m vmax.scripts.evaluate.evaluate \
  --sdc_actor ai \
  --path_model /path/to/model_final.pkl \
  --path_dataset /path/to/eval.tfrecord \
  --batch_size 8
```

For rule-based baselines:

```bash
python -m vmax.scripts.evaluate.evaluate \
  --sdc_actor expert \
  --path_dataset /path/to/eval.tfrecord
```

Evaluation outputs typically include:

- `evaluation_episodes.csv`
- `evaluation_results.txt`
- rendered videos if rendering is enabled

## 6. Where to Start for Common Tasks

### Train a model

Read:

1. [Configuration Guide](./1_config.md)
2. [Training Pipeline](./2_training_pipeline.md)

### Understand the observation

Read:

1. [Observation Guide](./3_observation.md)

### Understand the reward

Read:

1. [Reward Guide](./4_reward.md)

### Run evaluation and read metrics

Read:

1. [Evaluation Guide](./6_evaluation.md)
2. [Metrics Guide](./5_metrics.md)

### Run reward search

Read:

1. [Training Pipeline](./2_training_pipeline.md)
2. [Multi-Machine Search](./7_multi_machine_search.md)
3. [Command Reference](./COMMAND_REFERENCE.md)

## 7. Project-Specific Extensions in This Workspace

The current workspace includes additional research and engineering material beyond the base V-Max repository:

- multiplicative reward wrapper variants
- safe-bubble / route-safe reward shaping logic
- PPO warm-start and experiment-search utilities
- paper-writing notes and figure assets in `docs/`

For code inspection, start from:

- `vmax/simulator/wrappers/reward_tmp/`
- `vmax/simulator/metrics/safe_bubble.py`
- `vmax/agents/learning/reinforcement/ppo/`
- `vmax/scripts/experiments/`

## 8. Submission-Ready Repository Checklist

Before handing the repository to another user or attaching it to a paper artifact, verify:

- dataset paths are documented and not hard-coded to local machines
- the exact training command for the reported results is recorded
- the exact evaluation command for the reported results is recorded
- reward-search scripts reference valid root run directories and checkpoints
- draft-only files in `docs/` are clearly separated from user-facing documentation

The repository is easiest to consume if you keep the user path simple:

1. install
2. train
3. evaluate
4. inspect results

Everything else should branch off from that path.

For final packaging, also review:

- [Repository Organization Guide](./REPOSITORY_ORGANIZATION.md)
- [Submission Checklist](./SUBMISSION_CHECKLIST.md)
