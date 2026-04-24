# New Functions Inventory

This document summarizes the new utilities and function entry points added for BC/PPO resume, reward search, BO search, and multi-machine workflow.

## 1) Core Resume / Checkpoint Functions

### `vmax/agents/pipeline/checkpointing.py`

- `save_checkpoint(path, checkpoint)`
- `load_checkpoint(path)`
- `merge_params(target, source, strict=True)`
- `summarize_keys(keys, limit=10)`

Internal helpers:

- `_compatible_leaf(...)`
- `_flatten_params(...)`
- `_unflatten_params(...)`

Purpose:

- Unified checkpoint format support (full checkpoint + legacy params-only).
- Strict/partial param merge with clear mismatch reporting.

### `vmax/agents/learning/reinforcement/ppo/ppo_trainer.py`

Added/updated warm-start helpers:

- `_inverse_softplus(y)`
- `_adapt_policy_head_for_stochastic_policy(...)`

Resume behavior in `train(...)`:

- Supports `resume.mode=weights_only/full/...`
- Supports BC -> PPO warm start
- Includes `value_from_policy` path when BC checkpoint has no value head.

---

## 2) Reward Design Functions

### `vmax/simulator/wrappers/reward_tmp/reward_bubble2.py`

New/updated reward functions:

- `_compute_termination_switch_reward(...)`
- `_compute_route_safe_reward(...)`
- `_compute_forward_quality_reward(...)`
- `_compute_safe_bubble_reward(...)`
- plus full reward registry through `_get_reward_fn(...)`

Notable logic:

- `termination_switch` now handles termination-related penalties.
- `route_safe` and `safe_bubble` integration supports `ignore_overlap_in_safe_bubble`.

---

## 3) Sweep / Search Scripts

### `vmax/scripts/experiments/reward_search_manager.py`

Main entry:

- `main()`

Key function groups:

- Candidate generation:
  - `_build_candidates(...)`
  - `_grid_values(...)`
  - `_grid_from_range(...)`
- Trial orchestration:
  - `_run_trial(...)`
  - `_run_training(...)`
  - `_find_log_file(...)`
- Metric/grade:
  - `_parse_log_metrics(...)`
  - `_auto_detect_metrics(...)`
  - `_compute_grade(...)`
- Persistent DB:
  - `_append_trial_row(...)`
  - `_update_points_agg(...)`
  - `_select_best(...)`
  - `_write_view_map_notebook(...)`

### `vmax/scripts/experiments/rf_bo_suggester.py`

Main entry:

- `main()`

Key functions:

- `load_data(...)`
- `fit_rf(...)`
- `sample_candidates(...)`
- `compute_ei(...)`
- `pick_batch(...)`
- `plot_maps(...)`

### `vmax/scripts/experiments/rf_bo_trainer.py`

Main entry:

- `main()`

Key functions:

- `_run_training(...)`
- `_parse_metric_from_log(...)`
- `_compute_grade(...)`
- `_write_trials_row(...)`
- `_write_result_json(...)`

---

## 4) BC Evaluation / Analysis Scripts

### `vmax/scripts/experiments/eval_all_bc_models.py`

Main entry:

- `main()`

Key functions:

- `_iter_model_files(...)`
- `_evaluate_one_model(...)`
- `_load_existing_rows(...)`
- `_mean_metrics(...)`

Purpose:

- Evaluate all BC checkpoints and write `eval_all_bc/summary.csv`.

### `vmax/scripts/experiments/check_lq_to_mlp_compat.py`

Main entry:

- `main()`

Key functions:

- `_init_model(...)`
- `_diff_params(...)`
- `_encoder_output_dim(...)`

Purpose:

- LQ -> MLP compatibility diagnostics (keys/shapes/encoder output dim).

---

## 5) Utility Scripts for Map DB

### `vmax/scripts/experiments/rebuild_trials_from_results.py`

- Rebuild `trials_long.csv` from existing sweep `results/*.json` + logs.

### `vmax/scripts/experiments/recompute_grade.py`

- Recompute `grade` with new weights and regenerate aggregates.

### `vmax/scripts/experiments/merge_map_db.py`

Main entry:

- `main()`

Key functions:

- `_load_trials(...)`
- `_compute_grade(...)`
- `_dedupe_trials(...)`
- `_build_points_agg(...)`
- `_save_best(...)`
- `_save_index(...)`

Purpose:

- Merge multiple machines’ `map_db` into one canonical DB.

---

## 6) Environment Reproducibility

### New files

- `requirements.lock.txt` (pinned environment snapshot)
- `setup_env.sh` (one-command env recreate)

Usage:

```bash
./setup_env.sh venv
source venv/bin/activate
```

---

## 7) Notebook Assets Added

New notebooks under `docs/notebooks/observation/`:

- `eval.ipynb`
- `grid_refine_viz.ipynb`
- `log.ipynb`
- `rf_bo_viz.ipynb`
- `safecage.ipynb`

Use these for:

- training curve inspection
- topography/3D map visualization
- BO suggest/exploit/explore visualization

---

## 8) Frequently Used Commands

### BC checkpoint batch evaluation

```bash
python -m vmax.scripts.experiments.eval_all_bc_models \
  --run_dir runs/BC_VEC_LQ_06-02_17:13:28 \
  --batch_size 32 \
  --seed 0
```

### Grid reward search (BC -> PPO warm start)

```bash
python -m vmax.scripts.experiments.reward_search_manager \
  --root_run_dir /path/to/bc_run \
  --ckpt_path /path/to/bc_run/model/model_159764480.pkl \
  --param_space_json /path/to/vmax/config/reward_space_bc_rl_3x3x3.json \
  --mode grid \
  --algorithm ppo \
  --seed 0 \
  --convergence_steps 30000000 \
  --extra_overrides network/encoder=lq algorithm.network.policy.activation=relu resume.policy_init_std=0.03
```

### Merge two machines map DB

```bash
python -m vmax.scripts.experiments.merge_map_db \
  --target_map_db /path/to/canonical/map_db \
  --source_map_db /path/to/machineA/map_db /path/to/machineB/map_db \
  --include_target_existing \
  --w_vmax 0.5 --w_nuplan 0.5 --w_collision 0.0 --w_overlap 0.0 --w_oncoming 0.0
```
