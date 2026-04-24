# Copyright 2026 RTGTX7.

# Repository Organization Guide

This document explains how the repository should be read by an external user and which files should be treated as canonical, auxiliary, or research-only.

## 1. Canonical User Path

An external user should be able to follow this path without reading notebooks or draft notes:

1. install the environment
2. run `python -m vmax.scripts.training.train`
3. run `python -m vmax.scripts.evaluate.evaluate`
4. inspect checkpoints and evaluation outputs

Everything that supports this path is considered primary documentation or primary code.

## 2. Canonical Code Paths

### Training

- `vmax/scripts/training/train.py`
- `vmax/scripts/training/train_utils.py`

### Evaluation

- `vmax/scripts/evaluate/evaluate.py`
- `vmax/scripts/evaluate/utils.py`

The documented command convention for canonical entry points is module-style execution:

- `python -m vmax.scripts.training.train`
- `python -m vmax.scripts.evaluate.evaluate`
- `python -m vmax.scripts.experiments.<tool>`

### Main simulator path

- `vmax/simulator/`

### Main configs

- `vmax/config/base_config.yaml`
- `vmax/config/algorithm/*.yaml`
- `vmax/config/network/*.yaml`
- `vmax/config/network/encoder/*.yaml`

## 3. Project-Specific Extensions

These are important for this workspace and should be kept documented:

### Reward extensions

- `vmax/simulator/wrappers/reward.py`
- `vmax/simulator/wrappers/reward_tmp/reward_bubble2.py`
- `vmax/simulator/metrics/safe_bubble.py`

### Experiment search utilities

- `vmax/scripts/experiments/reward_search_manager.py`
- `vmax/scripts/experiments/rf_bo_suggester.py`
- `vmax/scripts/experiments/rf_bo_trainer.py`
- `vmax/scripts/experiments/rebuild_trials_from_results.py`
- `vmax/scripts/experiments/recompute_grade.py`
- `vmax/scripts/experiments/merge_map_db.py`

## 4. Auxiliary or Utility Scripts

These are useful, but they should not be the first files a new user sees:

- `vmax/scripts/experiments/eval_all_bc_models.py`
- `vmax/scripts/experiments/check_lq_to_mlp_compat.py`
- `vmax/scripts/experiments/mixed_precision_smoke_test.py`

They are best treated as debugging or analysis utilities.

## 5. Files That Should Not Be Treated as Primary Documentation

The following are valuable internal materials but should not be the first entry point for outside users:

- `docs/notebooks/`
- `docs/experimental_design_draft.md`
- `docs/ppo_reward_math_notes.md`
- `docs/network_structure_tikz.tex`
- `docs/network_structure_sample.dot`

These files support research, writing, or visualization. They should be clearly separated from the user-facing docs path.

## 6. Organization Principles for Submission

For a submission-ready repository:

- keep the top-level README short and accurate
- point users to one documentation index
- avoid broken links to non-existent docs
- define one canonical training script
- define one canonical evaluation script
- document all project-specific utilities that affect reported results

## 7. Practical Cleanup Recommendations

The current repository is usable, but an external reader may still be confused by a few structural issues:

- duplicate training entry points such as `train.py` and `train2.py`
- multiple reward wrapper variants under `reward_tmp/`
- draft research files mixed into the same `docs/` directory as user-facing guides

The documentation added in this workspace reduces that confusion without forcing a risky code refactor. If a stronger cleanup pass is needed later, the next step should be:

1. mark one training script as canonical
2. mark one reward wrapper as canonical
3. move draft-only docs into a clearly named research or paper subdirectory

## 8. What to Cite in a Paper Artifact

If this repository supports a paper submission, the artifact should explicitly name:

- the training script
- the evaluation script
- the exact config files used
- the reward-search script if hyperparameter search affects the final result
- the project-specific reward modules if they are part of the method

That is enough to make the implementation pathway understandable to a reviewer or collaborator.
