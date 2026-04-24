# Copyright 2026 RTGTX7.

# Command Reference

This document lists the main commands that a user is expected to run in the current repository. It also highlights the project-specific experiment utilities added beyond the base V-Max training and evaluation flow.

Unless there is a strong reason not to, the canonical invocation style is:

```bash
python -m vmax.scripts.<group>.<entry_point>
```

## 1. Canonical Commands

These are the commands that should appear in a user guide, README, or paper artifact.

### Train a PPO model

```bash
python -m vmax.scripts.training.train \
  algorithm=ppo \
  path_dataset=/path/to/train.tfrecord \
  path_dataset_eval=/path/to/valid.tfrecord
```

### Train a PPO model with LQ encoder

```bash
python -m vmax.scripts.training.train \
  algorithm=ppo \
  path_dataset=/path/to/train.tfrecord \
  path_dataset_eval=/path/to/valid.tfrecord \
  network/encoder=lq \
  observation_type=vec
```

### Evaluate a trained policy

```bash
python -m vmax.scripts.evaluate.evaluate \
  --sdc_actor ai \
  --path_model /path/to/model_final.pkl \
  --path_dataset /path/to/eval.tfrecord \
  --batch_size 8
```

### Evaluate the expert policy

```bash
python -m vmax.scripts.evaluate.evaluate \
  --sdc_actor expert \
  --path_dataset /path/to/eval.tfrecord
```

## 2. Project-Specific Experiment Commands

These commands correspond to functionality that is present in the current workspace but is not part of the simplest user path.

### Grid reward search

```bash
python -m vmax.scripts.experiments.reward_search_manager \
  --root_run_dir /path/to/stage1_run \
  --ckpt_path /path/to/stage1_run/model/model_final.pkl \
  --param_space_json vmax/config/reward_space_bc_rl_3x3x3.json \
  --mode grid
```

### Grid-refine reward search

```bash
python -m vmax.scripts.experiments.reward_search_manager \
  --root_run_dir /path/to/stage1_run \
  --ckpt_path /path/to/stage1_run/model/model_final.pkl \
  --param_space_json vmax/config/reward_space_bc_rl_3x3x3.json \
  --mode grid_refine
```

### Random-forest BO trainer

```bash
python -m vmax.scripts.experiments.rf_bo_trainer \
  --root_run_dir /path/to/stage1_run \
  --ckpt_path /path/to/stage1_run/model/model_final.pkl \
  --rounds 20 \
  --batch_size 8 \
  --total_timesteps 50000000 \
  --xmin 0.2 --xmax 3.0 --ymin 0.2 --ymax 3.0
```

### Random-forest suggestion only

```bash
python -m vmax.scripts.experiments.rf_bo_suggester \
  --trials_csv /path/to/root_run_dir/map_db/trials_long.csv \
  --out_csv /path/to/root_run_dir/map_db/suggestions.csv
```

### Rebuild trial table from stored results

```bash
python -m vmax.scripts.experiments.rebuild_trials_from_results \
  --root_run_dir /path/to/stage1_run
```

### Recompute grade after changing score weights

```bash
python -m vmax.scripts.experiments.recompute_grade \
  --map_db /path/to/stage1_run/map_db
```

### Merge map databases from multiple machines

```bash
python -m vmax.scripts.experiments.merge_map_db \
  --target_map_db /path/to/merged_map_db \
  --source_map_db /path/to/run_a/map_db /path/to/run_b/map_db
```

### Evaluate all BC checkpoints in a run

```bash
python -m vmax.scripts.experiments.eval_all_bc_models \
  --run_dir /path/to/bc_run \
  --batch_size 32 \
  --seed 0
```

### Check LQ to MLP checkpoint compatibility

```bash
python -m vmax.scripts.experiments.check_lq_to_mlp_compat
```

### Mixed-precision smoke test

```bash
python -m vmax.scripts.experiments.mixed_precision_smoke_test
```

## 3. Which Commands to Put in a Paper

For an academic artifact, the minimum command set to record is:

1. the exact training command
2. the exact evaluation command
3. the exact reward-search command if reward weights were tuned post hoc

Auxiliary debugging utilities such as compatibility checks or smoke tests usually do not belong in the paper unless they are part of the method contribution.

## 4. Which Commands Are Canonical

For general users, treat these as canonical:

- `vmax/scripts/training/train.py`
- `vmax/scripts/evaluate/evaluate.py`
- `vmax/scripts/experiments/reward_search_manager.py`

Treat the rest as project utilities or debugging helpers unless you explicitly rely on them in a workflow description.
