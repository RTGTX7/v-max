# Training Pipeline (V‑Max)

This document explains how training is launched, how data flows through the system, how checkpoints/resume work, and how to run reward sweeps. It is written to match the current repo behavior.

---

## 1) How Training Is Launched

**Entry point**
- `vmax/scripts/training/train.py`
- Uses Hydra + OmegaConf with `vmax/config/base_config.yaml`

**Flow**
1. Load config
2. Build env + data generator
3. Build networks + trainer
4. Run training loop
5. Save checkpoints + logs

---

## 2) Config Layout (Hydra)

**Base config**
- `vmax/config/base_config.yaml`

**Key sections**
- `algorithm`: PPO / SAC / BC / etc.
- `network`: encoder + policy/value MLPs
- `reward_type` + `reward_config`
- `total_timesteps`, `log_freq`, `save_freq`, `eval_freq`

**Output directory**
- Output directory is created by `train_utils.resolve_output_dir()`
- Default: `runs/<ALG>_<OBS>_<ENCODER>_<timestamp>/`

---

## 3) Training Data Flow (SAC Example)

```
[Simulator Scenarios] -> [Environment]
      |                     |
      v                     v
 [Policy] <- observations --[State]
      |                     |
      v                     v
  [Actions] -> [Env Step] -> [Replay Buffer]
      |                     |
      v                     v
   [Sampled Batch] -> [SGD Updates] -> [Networks]
```

---

## 4) Logging & Metrics

**Training logs**
- File: `<run_dir>/train.log` (Hydra job log)
- Format: `- metric_name: value`

**TensorBoard**
- Logged by `train_utils.log_metrics()`
- Keys:
  - `metrics/*` for general metrics
  - `rollout/*` for `ep_*`
  - `evaluation/*` for eval

**Evaluation outputs**
- `vmax/scripts/evaluate/evaluate.py`
- Outputs:
  - `evaluation_episodes.csv`
  - `evaluation_results.txt`
  - `mp4/` (if rendered)

---

## 5) Checkpoints & Resume

**Checkpoint types**
- Weights only: `model_*.pkl`, `model_final.pkl`
- Full state: `checkpoint_*.pkl`, `checkpoint_final.pkl`

**Resume modes**
- `weights_only`: warm‑start, resets optimizer + counters
- `full`: full recovery (weights + optimizer + counters + RNG + buffer for SAC)

**When to use**
- Use `weights_only` when:
  - reward weights changed
  - architecture changed
  - algorithm changed (PPO → SAC)
- Use `full` only if:
  - same algorithm + same architecture

**Examples**
```
# Start PPO
python vmax/scripts/training/train.py algorithm=ppo total_timesteps=1000000

# Resume full
python vmax/scripts/training/train.py algorithm=ppo \
  resume.enabled=true resume.ckpt_path=/path/to/checkpoint_1000000.pkl \
  resume.mode=full

# Resume weights only
python vmax/scripts/training/train.py algorithm=ppo \
  resume.enabled=true resume.ckpt_path=/path/to/model_final.pkl \
  resume.mode=weights_only resume.strict=false
```

---

## 6) Reward Sweep Manager (Stage‑2)

Script:
- `vmax/scripts/experiments/reward_search_manager.py`

Purpose:
- Runs reward sweeps under a fixed root run folder
- Maintains persistent map DB under `<root_run_dir>/map_db/`
- Skips duplicate trials automatically

**Output layout**
```
<root_run_dir>/
  stage2_sweeps/
    sweep_<timestamp>/
      manifest.yaml
      branches/
      results/
  map_db/
    trials_long.csv
    points_agg.csv
    best.yaml
    index.json
    view_map.ipynb
```

**Minimal JSON (range based)**
```
{
  "reward_keys": ["off_route", "progression"],
  "space": {
    "reward.off_route": {"type": "float", "low": 0.5, "high": 3.0, "scale": "linear"},
    "reward.progression": {"type": "float", "low": 0.3, "high": 2.0, "scale": "log"}
  },
  "search": {
    "mode": "grid_refine",
    "grid_points": 5,
    "rounds": 3,
    "shrink": 0.5,
    "budgets": [20000000, 50000000, 100000000]
  }
}
```

**Run (coarse‑to‑fine grid)**
```
python vmax/scripts/experiments/reward_search_manager.py \
  --root_run_dir /path/to/stage1_run \
  --ckpt_path /path/to/stage1_run/model/model_final.pkl \
  --param_space_json vmax/config/reward_space.json \
  --mode grid_refine
  --extra_overrides network/encoder=encoder
```

Notes:
- `root_run_dir` and `ckpt_path` typically point to the same run folder.
  Example: `root_run_dir=/path/to/stage1_run`, `ckpt_path=/path/to/stage1_run/model/model_final.pkl`.
- `--param_space_json` should point to the JSON file you created (e.g. `vmax/config/reward_space.json`).

**Run (simple grid)**
```
python vmax/scripts/experiments/reward_search_manager.py \
  --root_run_dir /path/to/stage1_run \
  --ckpt_path /path/to/stage1_run/model/model_final.pkl \
  --param_space_json vmax/config/reward_space.json \
  --mode grid \
  --convergence_steps 100000000
```

---

## 7) Tips

- Keep fixed reward weights in `base_config.yaml`.
- Only sweep the parameters listed in the JSON space.
- Use `weights_only + strict=false` if you change encoder or algorithm.
- Use `grid_refine` when you want auto coarse‑to‑fine search.

---

## 8) RF Bayesian Optimizer (SMAC‑lite)

Scripts:
- `vmax/scripts/experiments/rf_bo_suggester.py`
- `vmax/scripts/experiments/rf_bo_trainer.py`

Purpose:
- Use a Random‑Forest surrogate + EI to propose new reward points.
- Reduce grid search waste in 2D (`reward_config.off_route`, `reward_config.progression`).

**Generate suggestions only**
```
python vmax/scripts/experiments/rf_bo_suggester.py \
  --trials_csv /path/to/root_run_dir/map_db/trials_long.csv \
  --out_csv /path/to/root_run_dir/map_db/suggestions.csv \
  --map_png /path/to/root_run_dir/map_db/map_rfbo.png \
  --batch_size 12 \
  --candidates 5000 \
  --xmin 0.2 --xmax 3.0 --ymin 0.2 --ymax 3.0 \
  --log_y \
  --xi 0.01 \
  --explore_frac 0.3 \
  --min_dist 0.05 \
  --seed 0
```

Output:
- `suggestions.csv` (next points: exploit/explore, EI, uncertainty)
- `map_rfbo.png` (mean / uncertainty / EI maps + points)

**Auto‑train loop (RF‑BO + train.py)**
```
python vmax/scripts/experiments/rf_bo_trainer.py \
  --root_run_dir /path/to/stage1_run \
  --ckpt_path /path/to/stage1_run/model/model_final.pkl \
  --rounds 40 \
  --batch_size 12 \
  --total_timesteps 50000000 \
  --xmin 0.2 --xmax 3.0 --ymin 0.2 --ymax 3.0 \
  --log_y \
  --xi 0.01 \
  --explore_frac 0.3 \
  --min_dist 0.05 \
  --seed 0 \
  --extra_overrides path_dataset=/path/to/training.tfrecord path_dataset_eval=/path/to/eval.tfrecord
```

Notes:
- Auto‑trainer reads `train.log` for `vmax_aggregate_score` and `nuplan_aggregate_score`.
- Grade default: `0.5*vmax + 0.5*nuplan` (override with `--grade_*` flags).
- All runs are stored under:
  `<root_run_dir>/stage2_sweeps/rfbo_round_###/cand_###/`

Notebook:
- `docs/notebooks/observation/rf_bo_viz.ipynb`
  (loads trials, generates suggestions + map)

---

If anything here doesn’t match your workflow, tell me what you want changed and I’ll rewrite again.
