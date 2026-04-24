# Multi-Machine Parameter Search

This guide prepares the repo for running reward search on two computers and merging results safely.

## 1) Keep Repository Clean Before Upload

The root `.gitignore` now ignores local artifacts:

- `runs/`
- `data/`
- `tmp/`
- `benchmark/`
- `venv/`
- `events.out.tfevents*`
- model/checkpoint binary files (`*.pkl`, `*.ckpt`, `*.tfrecord`)

This avoids uploading large local experiment outputs.

## 2) Upload to Your GitHub

If `origin` is not your repo, replace it first:

```bash
git remote remove origin
git remote add origin <your_github_repo_url>
```

Commit and push:

```bash
git add .
git commit -m "Prepare BC->PPO search pipeline and multi-machine map merge"
git push -u origin main
```

## 3) Run Search on Two Computers

Use the same code version on both machines.

- Machine A runs one subset (for example `seed=0`).
- Machine B runs another subset (for example `seed=1`).

Example (Machine A):

```bash
python -m vmax.scripts.experiments.reward_search_manager \
  --root_run_dir /path/to/runs/BC_VEC_LQ_xxx \
  --ckpt_path /path/to/runs/BC_VEC_LQ_xxx/model/model_159764480.pkl \
  --param_space_json /path/to/vmax/config/reward_space_bc_rl_3x3x3.json \
  --mode grid \
  --algorithm ppo \
  --seed 0 \
  --convergence_steps 30000000 \
  --extra_overrides network/encoder=lq algorithm.network.policy.activation=relu resume.policy_init_std=0.03
```

Machine B: same command with `--seed 1`.

## 4) Merge `map_db` From Both Machines

Pick one machine as canonical (for example Machine A), copy Machine B map DB:

```bash
rsync -avz <machine_b>:/path/to/runs/BC_VEC_LQ_xxx/map_db/ /path/to/tmp/map_db_b/
```

Then merge:

```bash
python -m vmax.scripts.experiments.merge_map_db \
  --target_map_db /path/to/runs/BC_VEC_LQ_xxx/map_db \
  --source_map_db /path/to/runs/BC_VEC_LQ_xxx/map_db /path/to/tmp/map_db_b \
  --include_target_existing \
  --w_vmax 0.5 --w_nuplan 0.5 --w_collision 0.0 --w_overlap 0.0 --w_oncoming 0.0
```

Outputs:

- `map_db/trials_long.csv` (deduplicated by `trial_id`)
- `map_db/points_agg.csv`
- `map_db/best.yaml`
- `map_db/index.json`

## 5) Continue Search Iteratively

After merge, launch the next sweep on either machine using the updated canonical `map_db`.
Repeat run -> sync -> merge.
