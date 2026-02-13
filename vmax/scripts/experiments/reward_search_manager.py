#!/usr/bin/env python3
# Copyright 2025 Valeo.

"""Reward sweep manager with persistent map DB and dedup.

Usage examples:
  python vmax/scripts/experiments/reward_search_manager.py \
    --root_run_dir /path/to/stage1_run \
    --ckpt_path /path/to/model_final.pkl \
    --param_space_json /path/to/space.json \
    --mode grid

Notes:
- Runs training via Hydra overrides and stores outputs under <root_run_dir>/stage2_sweeps/.
- Appends trial metrics to <root_run_dir>/map_db/trials_long.csv and updates aggregates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

try:
    import yaml
except Exception:  # pragma: no cover - fallback if PyYAML isn't installed
    yaml = None

try:
    import pandas as pd
except Exception as exc:  # pragma: no cover - pandas is expected in repo
    raise RuntimeError("pandas is required for this script.") from exc


REQUIRED_METRICS = [
    "vmax_aggregate_score",
    "nuplan_aggregate_score",
    "at_fault_collision",
    "overlap",
    "distance_into_oncoming_traffic",
]

METRIC_ALIASES = {
    "vmax_aggregate_score": [
        "vmax_aggregate_score",
        "vmax_score",
        "vmax_aggregate",
    ],
    "nuplan_aggregate_score": [
        "nuplan_aggregate_score",
        "nuplan_score",
        "nuplan_aggregate",
    ],
    "at_fault_collision": [
        "at_fault_collision",
        "collision",
        "at_fault_collision_rate",
    ],
    "overlap": [
        "overlap",
        "overlap_rate",
    ],
    "distance_into_oncoming_traffic": [
        "distance_into_oncoming_traffic",
        "distance_oncoming_traffic",
        "oncoming_traffic_distance",
    ],
}


@dataclass
class SearchConfig:
    mode: str
    candidates: int | None = None
    min_budget: int | None = None
    max_budget: int | None = None
    eta: int = 3
    grid_points: int = 5
    grid_points_by_round: list[int] | None = None
    rounds: int = 3
    shrink: float = 0.5
    budgets: list[int] | None = None
    resume_from_map: bool = False
    resume_shrink_rounds: int = 0
    refine_top_k: int = 1
    refine_merge_radius: float = 0.2
    refine_pool: int = 20
    refine_diversity: bool = True
    refine_fill_after_merge: bool = True


@dataclass
class GradeWeights:
    w_vmax: float
    w_nuplan: float
    w_collision: float
    w_overlap: float
    w_oncoming: float


def _now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_yaml(path: Path, data: dict) -> None:
    if yaml is not None:
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False)
    else:
        with path.open("w", encoding="utf-8") as f:
            f.write(json.dumps(data, indent=2))


def _ckpt_signature(ckpt_path: Path) -> str:
    stat = ckpt_path.stat()
    payload = f"{ckpt_path}|{stat.st_size}|{stat.st_mtime}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _trial_id(ckpt_sig: str, overrides: List[str], seed: int, algorithm: str, total_steps: int) -> str:
    blob = json.dumps(
        {
            "ckpt": ckpt_sig,
            "overrides": sorted(overrides),
            "seed": seed,
            "algorithm": algorithm,
            "total_steps": total_steps,
        },
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _param_vector(overrides: List[str]) -> str:
    return "|".join(sorted(overrides))


def _flatten_space(space: dict) -> dict:
    return space.get("space", space)


def _parse_search(cfg: dict) -> SearchConfig:
    search = cfg.get("search", {})
    return SearchConfig(
        mode=search.get("mode", "grid"),
        candidates=search.get("candidates"),
        min_budget=search.get("min_budget"),
        max_budget=search.get("max_budget"),
        eta=int(search.get("eta", 3)),
        grid_points=int(search.get("grid_points", 5)),
        grid_points_by_round=search.get("grid_points_by_round"),
        rounds=int(search.get("rounds", 3)),
        shrink=float(search.get("shrink", 0.5)),
        budgets=search.get("budgets"),
        resume_from_map=bool(search.get("resume_from_map", False)),
        resume_shrink_rounds=int(search.get("resume_shrink_rounds", 0)),
        refine_top_k=int(search.get("refine_top_k", 1)),
        refine_merge_radius=float(search.get("refine_merge_radius", 0.2)),
        refine_pool=int(search.get("refine_pool", 20)),
        refine_diversity=bool(search.get("refine_diversity", True)),
        refine_fill_after_merge=bool(search.get("refine_fill_after_merge", True)),
    )


def _value_from_space(defn: dict, rng: random.Random) -> Any:
    typ = defn.get("type", "choice")
    if typ == "choice":
        values = defn.get("values", [])
        return rng.choice(values)
    if typ == "int":
        low, high = int(defn["low"]), int(defn["high"])
        return rng.randint(low, high)
    if typ == "float":
        low, high = float(defn["low"]), float(defn["high"])
        scale = defn.get("scale", "linear")
        if scale == "log":
            low, high = math.log10(low), math.log10(high)
            return 10 ** rng.uniform(low, high)
        return rng.uniform(low, high)
    raise ValueError(f"Unknown param type: {typ}")


def _grid_values(defn: dict) -> List[Any]:
    typ = defn.get("type", "choice")
    if typ == "choice":
        return list(defn.get("values", []))
    if typ in ("float", "int"):
        if "values" in defn:
            return list(defn["values"])
    raise ValueError(f"Grid mode requires explicit values for {defn}")


def _grid_from_range(defn: dict, low: float, high: float, n: int) -> List[float]:
    if n < 2:
        return [low]
    scale = defn.get("scale", "linear")
    if scale == "log":
        low_l = math.log10(low)
        high_l = math.log10(high)
        step = (high_l - low_l) / (n - 1)
        return [10 ** (low_l + i * step) for i in range(n)]
    step = (high - low) / (n - 1)
    return [low + i * step for i in range(n)]


def _build_candidates(space: dict, mode: str, rng: random.Random, count: int | None = None) -> List[dict]:
    keys = list(space.keys())
    if mode == "grid":
        values_list = [(_grid_values(space[k])) for k in keys]
        combos: List[dict] = []
        def _walk(idx: int, current: dict):
            if idx == len(keys):
                combos.append(current.copy())
                return
            for val in values_list[idx]:
                current[keys[idx]] = val
                _walk(idx + 1, current)
        _walk(0, {})
        return combos

    if count is None:
        raise ValueError("Random/halving modes require candidates count.")

    combos = []
    for _ in range(count):
        params = {k: _value_from_space(space[k], rng) for k in keys}
        combos.append(params)
    return combos


def _param_to_override(key: str, value: Any) -> str:
    if key.startswith("reward."):
        key = "reward_config." + key.split(".", 1)[1]
    elif key.startswith("reward_config."):
        pass
    return f"{key}={value}"


def _parse_log_metrics(log_path: Path) -> dict:
    metrics: Dict[str, float] = {}
    pattern = re.compile(r"- ([^:]+): (.+)$")
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = pattern.search(line.strip())
            if not match:
                continue
            key = match.group(1).strip()
            val = match.group(2).strip()
            try:
                metrics[key] = float(val)
            except ValueError:
                continue
    return metrics


def _auto_detect_metrics(raw: dict) -> Tuple[dict, dict]:
    mapping: Dict[str, str] = {}
    result: Dict[str, float | None] = {}
    keys_lower = {k.lower(): k for k in raw.keys()}

    for target, aliases in METRIC_ALIASES.items():
        found = None
        for alias in aliases:
            if alias in raw:
                found = alias
                break
            if alias.lower() in keys_lower:
                found = keys_lower[alias.lower()]
                break
        if found is None:
            # fallback: substring match
            for k in raw.keys():
                if alias in k.lower():
                    found = k
                    break
        if found is not None:
            mapping[target] = found
            result[target] = raw[found]
        else:
            result[target] = None
    return result, mapping


def _ensure_db(root_run_dir: Path) -> Path:
    db_dir = root_run_dir / "map_db"
    db_dir.mkdir(parents=True, exist_ok=True)
    (db_dir / "trials_long.csv").touch(exist_ok=True)
    return db_dir


def _append_trial_row(path: Path, row: dict) -> None:
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _load_trials(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def _update_points_agg(trials_df: pd.DataFrame, out_path: Path, metric_cols: List[str]) -> pd.DataFrame:
    if trials_df.empty:
        return pd.DataFrame()

    group_cols = ["param_vector"]
    agg = trials_df.groupby(group_cols)[metric_cols + ["grade"]].agg(["mean", "std"]).reset_index()
    agg.columns = [
        "_".join([c for c in col if c]) if isinstance(col, tuple) else str(col)
        for col in agg.columns
    ]
    agg.to_csv(out_path, index=False)
    return agg


def _select_best(agg_df: pd.DataFrame) -> dict | None:
    if agg_df.empty or "grade_mean" not in agg_df:
        return None
    best_row = agg_df.sort_values("grade_mean", ascending=False).iloc[0].to_dict()
    return best_row


def _suggest_next(space: dict, best_params: dict, rng: random.Random, k: int, radius: float) -> List[dict]:
    suggestions = []
    for _ in range(k):
        candidate = {}
        for key, spec in space.items():
            typ = spec.get("type", "choice")
            if typ == "choice":
                values = list(spec.get("values", []))
                candidate[key] = rng.choice(values) if values else best_params.get(key)
            elif typ in ("float", "int"):
                low, high = float(spec["low"]), float(spec["high"])
                scale = spec.get("scale", "linear")
                base = float(best_params.get(key, low))
                if scale == "log":
                    low_l, high_l = math.log10(low), math.log10(high)
                    base_l = math.log10(max(base, low))
                    span = (high_l - low_l) * radius
                    new_l = min(high_l, max(low_l, base_l + rng.uniform(-span, span)))
                    value = 10 ** new_l
                else:
                    span = (high - low) * radius
                    value = min(high, max(low, base + rng.uniform(-span, span)))
                if typ == "int":
                    value = int(round(value))
                candidate[key] = value
            else:
                candidate[key] = best_params.get(key)
        suggestions.append(candidate)
    return suggestions


def _merge_centers(
    centers: list[dict],
    ranges: dict,
    merge_radius: float,
) -> list[dict]:
    if not centers:
        return []
    merged: list[dict] = []
    for c in centers:
        keep = True
        for m in merged:
            dist = 0.0
            for key, (low, high) in ranges.items():
                span = max(high - low, 1e-9)
                dist += ((c.get(key, 0.0) - m.get(key, 0.0)) / span) ** 2
            dist = math.sqrt(dist)
            if dist < merge_radius:
                keep = False
                break
        if keep:
            merged.append(c)
    return merged


def _select_diverse_centers(
    candidates: list[dict],
    ranges: dict,
    k: int,
) -> list[dict]:
    if not candidates:
        return []
    selected: list[dict] = [candidates[0]]
    while len(selected) < min(k, len(candidates)):
        best = None
        best_dist = -1.0
        for c in candidates:
            if c in selected:
                continue
            # distance to nearest selected
            dmin = 1e9
            for s in selected:
                dist = 0.0
                for key, (low, high) in ranges.items():
                    span = max(high - low, 1e-9)
                    dist += ((c.get(key, 0.0) - s.get(key, 0.0)) / span) ** 2
                dist = math.sqrt(dist)
                dmin = min(dmin, dist)
            if dmin > best_dist:
                best_dist = dmin
                best = c
        if best is None:
            break
        selected.append(best)
    return selected


def _fill_centers_after_merge(
    merged: list[dict],
    candidates: list[dict],
    ranges: dict,
    k: int,
) -> list[dict]:
    if len(merged) >= k:
        return merged
    # Fill with farthest candidates from existing centers to keep diversity.
    selected = list(merged)
    remaining = [c for c in candidates if c not in selected]
    while len(selected) < min(k, len(candidates)) and remaining:
        best = None
        best_dist = -1.0
        for c in remaining:
            if not selected:
                best = c
                break
            dmin = 1e9
            for s in selected:
                dist = 0.0
                for key, (low, high) in ranges.items():
                    span = max(high - low, 1e-9)
                    dist += ((c.get(key, 0.0) - s.get(key, 0.0)) / span) ** 2
                dist = math.sqrt(dist)
                dmin = min(dmin, dist)
            if dmin > best_dist:
                best_dist = dmin
                best = c
        if best is None:
            break
        selected.append(best)
        remaining = [c for c in remaining if c is not best]
    return selected


def _write_view_map_notebook(path: Path) -> None:
    code = """\
import pandas as pd
import matplotlib.pyplot as plt

trials = pd.read_csv('trials_long.csv')
points = pd.read_csv('points_agg.csv')

cols = ['vmax_aggregate_score', 'nuplan_aggregate_score', 'at_fault_collision', 'overlap', 'distance_into_oncoming_traffic', 'grade']
print(trials[cols].describe(percentiles=[0.25,0.5,0.75,0.9,0.99]))

leaderboard = trials.sort_values('grade', ascending=False).head(20)
print(leaderboard[['param_vector','grade'] + cols])

plt.figure(figsize=(6,4))
plt.scatter(trials['vmax_aggregate_score'], trials['at_fault_collision'], alpha=0.6)
plt.xlabel('vmax_aggregate_score')
plt.ylabel('at_fault_collision')
plt.title('VMax vs Collision')
plt.grid(True)
plt.show()

plt.figure(figsize=(6,4))
plt.hist(trials['grade'].dropna(), bins=30)
plt.xlabel('grade')
plt.ylabel('count')
plt.title('Grade Histogram')
plt.grid(True)
plt.show()
"""

    nb = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": code.splitlines(keepends=True),
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)


def _run_training(
    repo_root: Path,
    cwd: Path,
    overrides: List[str],
) -> None:
    cmd = [sys.executable, "-m", "vmax.scripts.training.train"] + overrides
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def _find_log_file(run_dir: Path) -> Path:
    logs = list(run_dir.glob("*.log"))
    if logs:
        return max(logs, key=lambda p: p.stat().st_mtime)
    raise FileNotFoundError(f"No log file found in {run_dir}")


def _trial_completed(trials_df: pd.DataFrame, trial_id: str) -> bool:
    if trials_df.empty:
        return False
    match = trials_df[trials_df["trial_id"] == trial_id]
    if match.empty:
        return False
    return match.iloc[-1].get("status") == "completed"


def _compute_grade(metrics: dict, weights: GradeWeights) -> float | None:
    try:
        return (
            weights.w_vmax * float(metrics.get("vmax_aggregate_score"))
            + weights.w_nuplan * float(metrics.get("nuplan_aggregate_score"))
            - weights.w_collision * float(metrics.get("at_fault_collision"))
            - weights.w_overlap * float(metrics.get("overlap"))
            - weights.w_oncoming * float(metrics.get("distance_into_oncoming_traffic"))
        )
    except (TypeError, ValueError):
        return None


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _run_trial(
    repo_root: Path,
    sweep_dir: Path,
    root_run_dir: Path,
    ckpt_path: Path,
    algorithm: str,
    seed: int,
    overrides: List[str],
    total_steps: int,
    resume_from: Path | None,
    strict: bool,
    force_rerun: bool,
    trials_db: Path,
    grade_weights: GradeWeights,
    split_main_steps: int | None,
    convergence_steps: int | None,
) -> dict:
    ckpt_sig = _ckpt_signature(ckpt_path)
    trial_id = _trial_id(ckpt_sig, overrides, seed, algorithm, total_steps)

    trials_df = _load_trials(trials_db)
    if not force_rerun and _trial_completed(trials_df, trial_id):
        return {"trial_id": trial_id, "status": "skipped"}

    branch_dir = sweep_dir / "branches" / trial_id
    results_dir = sweep_dir / "results"
    _ensure_dir(branch_dir)
    _ensure_dir(results_dir)

    base_overrides = [
        f"algorithm={algorithm}",
        f"seed={seed}",
        f"name_exp=branches",
        f"name_run={trial_id}",
    ]
    base_overrides += overrides

    def _run_budget(target_steps: int, resume_ckpt: Path | None) -> None:
        run_overrides = list(base_overrides) + [f"total_timesteps={target_steps}"]
        if resume_ckpt is None:
            run_overrides += [
                "resume.enabled=true",
                f"resume.ckpt_path={ckpt_path}",
                "resume.mode=weights_only",
                f"resume.strict={str(strict).lower()}",
            ]
        else:
            run_overrides += [
                "resume.enabled=true",
                f"resume.ckpt_path={resume_ckpt}",
                "resume.mode=full",
                "resume.strict=true",
            ]
        _run_training(repo_root, sweep_dir, run_overrides)

    if split_main_steps and convergence_steps and total_steps == convergence_steps and split_main_steps < convergence_steps:
        _run_budget(split_main_steps, resume_ckpt=None)
        resume_ckpt = branch_dir / "model" / "checkpoint_final.pkl"
        if not resume_ckpt.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_ckpt}")
        _run_budget(total_steps, resume_ckpt=resume_ckpt)
    else:
        _run_budget(total_steps, resume_ckpt=resume_from)

    log_path = _find_log_file(branch_dir)
    raw_metrics = _parse_log_metrics(log_path)
    metrics, mapping = _auto_detect_metrics(raw_metrics)

    grade = _compute_grade(metrics, grade_weights)

    row = {
        "trial_id": trial_id,
        "status": "completed",
        "algorithm": algorithm,
        "seed": seed,
        "total_timesteps": total_steps,
        "ckpt_signature": ckpt_sig,
        "overrides": json.dumps(overrides, sort_keys=True),
        "param_vector": _param_vector(overrides),
        "log_path": str(log_path),
        "branch_dir": str(branch_dir),
        "timestamp": datetime.now().isoformat(),
        **metrics,
        "grade": grade,
    }
    _append_trial_row(trials_db, row)

    result_path = results_dir / f"{trial_id}.json"
    with result_path.open("w", encoding="utf-8") as f:
        json.dump({"trial": row, "metric_mapping": mapping}, f, indent=2)

    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Reward sweep manager")
    parser.add_argument("--root_run_dir", required=True)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--param_space_json", required=True)
    parser.add_argument("--mode", choices=["grid", "random", "halving", "grid_refine"], default=None)
    parser.add_argument("--algorithm", default="ppo")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force_rerun", action="store_true")
    parser.add_argument("--suggest_k", type=int, default=10)
    parser.add_argument("--suggest_radius", type=float, default=0.1)
    parser.add_argument("--w_vmax", type=float, default=1.0)
    parser.add_argument("--w_nuplan", type=float, default=1.0)
    parser.add_argument("--w_collision", type=float, default=1.0)
    parser.add_argument("--w_overlap", type=float, default=1.0)
    parser.add_argument("--w_oncoming", type=float, default=1.0)
    parser.add_argument("--convergence_steps", type=int, default=None)
    parser.add_argument("--main_steps", type=int, default=None)
    parser.add_argument("--extra_overrides", nargs="*", default=None)

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    root_run_dir = Path(args.root_run_dir).resolve()
    ckpt_path = Path(args.ckpt_path).resolve()

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    space_cfg = _load_json(Path(args.param_space_json))
    space = _flatten_space(space_cfg)
    search_cfg = _parse_search(space_cfg)
    mode = args.mode or search_cfg.mode

    rng = random.Random(0)

    sweep_dir = root_run_dir / "stage2_sweeps" / f"sweep_{_now_ts()}"
    _ensure_dir(sweep_dir / "branches")
    _ensure_dir(sweep_dir / "results")

    manifest = {
        "root_run_dir": str(root_run_dir),
        "ckpt_path": str(ckpt_path),
        "ckpt_signature": _ckpt_signature(ckpt_path),
        "algorithm": args.algorithm,
        "seed": args.seed,
        "mode": mode,
        "search": space_cfg.get("search", {}),
        "created_at": datetime.now().isoformat(),
    }
    _write_yaml(sweep_dir / "manifest.yaml", manifest)

    db_dir = _ensure_db(root_run_dir)
    trials_db = db_dir / "trials_long.csv"
    points_agg_path = db_dir / "points_agg.csv"
    best_path = db_dir / "best.yaml"
    index_path = db_dir / "index.json"
    next_candidates_path = db_dir / "next_candidates.json"
    view_map_path = db_dir / "view_map.ipynb"

    grade_weights = GradeWeights(
        w_vmax=args.w_vmax,
        w_nuplan=args.w_nuplan,
        w_collision=args.w_collision,
        w_overlap=args.w_overlap,
        w_oncoming=args.w_oncoming,
    )

    total_steps = args.convergence_steps or search_cfg.max_budget
    split_main_steps = args.main_steps

    extra_overrides = args.extra_overrides or []

    if mode == "grid":
        candidates = _build_candidates(space, mode, rng)
    elif mode == "random":
        candidates = _build_candidates(space, mode, rng, count=search_cfg.candidates)
    elif mode == "halving":
        candidates = _build_candidates(space, "random", rng, count=search_cfg.candidates)
    elif mode == "grid_refine":
        candidates = []
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    def _overrides_for(params: dict) -> List[str]:
        overrides = [_param_to_override(k, v) for k, v in sorted(params.items())]
        overrides.extend(extra_overrides)
        return overrides

    if mode == "grid_refine":
        # Optional: resume around best point from existing map DB
        best_params_seed: dict | None = None
        best_centers_seed: list[dict] | None = None
        if search_cfg.resume_from_map and trials_db.exists() and trials_db.stat().st_size > 0:
            trials_df = _load_trials(trials_db)
            if not trials_df.empty and "grade" in trials_df.columns:
                top_rows = trials_df.sort_values("grade", ascending=False).head(search_cfg.refine_top_k)
                centers = []
                for _, row in top_rows.iterrows():
                    try:
                        override_list = json.loads(row["overrides"])
                        override_map = {ov.split("=", 1)[0]: ov.split("=", 1)[1] for ov in override_list}
                        center = {}
                        for key in space.keys():
                            override_key = _param_to_override(key, "VALUE").split("=", 1)[0]
                            if override_key in override_map:
                                try:
                                    center[key] = float(override_map[override_key])
                                except ValueError:
                                    continue
                        if center:
                            centers.append(center)
                    except Exception:
                        continue
                if centers:
                    best_centers_seed = centers
                    best_params_seed = centers[0]

        if search_cfg.budgets:
            budgets = [int(b) for b in search_cfg.budgets]
        else:
            if not search_cfg.min_budget or not search_cfg.max_budget:
                raise ValueError("grid_refine requires min_budget and max_budget or explicit budgets.")
            budgets = []
            if search_cfg.rounds <= 1:
                budgets = [search_cfg.max_budget]
            else:
                step = (search_cfg.max_budget - search_cfg.min_budget) / (search_cfg.rounds - 1)
                budgets = [int(search_cfg.min_budget + i * step) for i in range(search_cfg.rounds)]

        ranges = {k: (float(space[k]["low"]), float(space[k]["high"])) for k in space.keys() if "low" in space[k]}
        best_params = best_params_seed
        centers = best_centers_seed if best_centers_seed else ([] if best_params is None else [best_params])

        if centers and search_cfg.resume_shrink_rounds > 0:
            for _ in range(search_cfg.resume_shrink_rounds):
                new_ranges = {}
                for key in ranges.keys():
                    low, high = ranges[key]
                    span = (high - low) * search_cfg.shrink
                    center_vals = [c.get(key) for c in centers if key in c]
                    if not center_vals:
                        new_ranges[key] = (low, high)
                        continue
                    # use first center for shrink when multiple; additional centers get their own local grids later
                    center = center_vals[0]
                    new_low = max(low, float(center) - span / 2)
                    new_high = min(high, float(center) + span / 2)
                    new_ranges[key] = (new_low, new_high)
                ranges = new_ranges
        for round_idx, budget in enumerate(budgets):
            if search_cfg.grid_points_by_round:
                if round_idx < len(search_cfg.grid_points_by_round):
                    grid_points = int(search_cfg.grid_points_by_round[round_idx])
                else:
                    grid_points = int(search_cfg.grid_points_by_round[-1])
            else:
                grid_points = search_cfg.grid_points
            grid_candidates = []
            if not centers:
                centers = [best_params] if best_params else []
            if not centers:
                centers = [None]

            for center in centers:
                for key, spec in space.items():
                    if "low" in spec and "high" in spec:
                        low, high = ranges[key]
                        if center and key in center:
                            span = (high - low) * search_cfg.shrink
                            low = max(low, float(center[key]) - span / 2)
                            high = min(high, float(center[key]) + span / 2)
                        values = _grid_from_range(spec, low, high, grid_points)
                        space[key]["_grid_values"] = values
                    else:
                        space[key]["_grid_values"] = _grid_values(spec)

                keys = list(space.keys())
                values_list = [space[k]["_grid_values"] for k in keys]

                def _walk(idx: int, current: dict):
                    if idx == len(keys):
                        grid_candidates.append(current.copy())
                        return
                    for val in values_list[idx]:
                        current[keys[idx]] = val
                        _walk(idx + 1, current)

                _walk(0, {})

            # Anchor previous best
            if best_params is not None:
                grid_candidates.append(best_params)

            best_row = None
            round_rows = []
            for params in grid_candidates:
                overrides = _overrides_for(params)
                row = _run_trial(
                    repo_root,
                    sweep_dir,
                    root_run_dir,
                    ckpt_path,
                    args.algorithm,
                    args.seed,
                    overrides,
                    total_steps=budget,
                    resume_from=None,
                    strict=False,
                    force_rerun=args.force_rerun,
                    trials_db=trials_db,
                    grade_weights=grade_weights,
                    split_main_steps=None,
                    convergence_steps=None,
                )
                if row.get("status") == "completed":
                    round_rows.append(row)
                    if best_row is None or (row.get("grade") is not None and row.get("grade") > best_row.get("grade")):
                        best_row = row

            if best_row:
                override_list = json.loads(best_row["overrides"])
                override_map = {ov.split("=", 1)[0]: ov.split("=", 1)[1] for ov in override_list}
                best_params = {}
                for key in ranges.keys():
                    override_key = _param_to_override(key, "VALUE").split("=", 1)[0]
                    if override_key in override_map:
                        try:
                            best_params[key] = float(override_map[override_key])
                        except ValueError:
                            continue
                # compute top-k centers for next round
                round_df = None
                if round_rows:
                    round_df = pd.DataFrame(round_rows)
                if round_df is None or round_df.empty:
                    round_df = _load_trials(trials_db)
                pool_rows = round_df.sort_values("grade", ascending=False).head(search_cfg.refine_pool)
                centers = []
                for _, row in pool_rows.iterrows():
                    try:
                        override_list = json.loads(row["overrides"])
                        override_map = {ov.split("=", 1)[0]: ov.split("=", 1)[1] for ov in override_list}
                        center = {}
                        for key in ranges.keys():
                            override_key = _param_to_override(key, "VALUE").split("=", 1)[0]
                            if override_key in override_map:
                                try:
                                    center[key] = float(override_map[override_key])
                                except ValueError:
                                    continue
                        if center:
                            centers.append(center)
                    except Exception:
                        continue
                pool_centers = list(centers)
                if search_cfg.refine_diversity:
                    centers = _select_diverse_centers(centers, ranges, search_cfg.refine_top_k)
                else:
                    centers = centers[: search_cfg.refine_top_k]
                centers = _merge_centers(centers, ranges, search_cfg.refine_merge_radius)
                if search_cfg.refine_fill_after_merge and len(centers) < search_cfg.refine_top_k:
                    centers = _fill_centers_after_merge(centers, pool_centers, ranges, search_cfg.refine_top_k)

    elif mode == "halving":
        if not search_cfg.min_budget or not search_cfg.max_budget:
            raise ValueError("Halving mode requires min_budget and max_budget in search config.")
        budgets = []
        budget = search_cfg.min_budget
        while budget <= search_cfg.max_budget:
            budgets.append(budget)
            budget = int(budget * search_cfg.eta)

        current = candidates
        resume_map: Dict[str, Path | None] = {}
        for budget in budgets:
            results = []
            for params in current:
                overrides = _overrides_for(params)
                param_key = _param_vector(overrides)
                resume_from = resume_map.get(param_key)
                row = _run_trial(
                    repo_root,
                    sweep_dir,
                    root_run_dir,
                    ckpt_path,
                    args.algorithm,
                    args.seed,
                    overrides,
                    total_steps=budget,
                    resume_from=resume_from,
                    strict=False,
                    force_rerun=args.force_rerun,
                    trials_db=trials_db,
                    grade_weights=grade_weights,
                    split_main_steps=None,
                    convergence_steps=None,
                )
                if row.get("status") == "completed":
                    results.append((row.get("grade"), params))
                    branch_dir = Path(row.get("branch_dir", "\\"))
                    checkpoint_path = branch_dir / "model" / "checkpoint_final.pkl"
                    if checkpoint_path.exists():
                        resume_map[param_key] = checkpoint_path

            results = [r for r in results if r[0] is not None]
            results.sort(key=lambda x: x[0], reverse=True)
            if not results:
                break
            keep = max(1, math.ceil(len(results) / search_cfg.eta))
            current = [p for _, p in results[:keep]]
    else:
        for params in candidates:
            overrides = _overrides_for(params)
            if total_steps is None:
                raise ValueError("convergence_steps is required for grid/random mode.")
            _run_trial(
                repo_root,
                sweep_dir,
                root_run_dir,
                ckpt_path,
                args.algorithm,
                args.seed,
                overrides,
                total_steps=total_steps,
                resume_from=None,
                strict=False,
                force_rerun=args.force_rerun,
                trials_db=trials_db,
                grade_weights=grade_weights,
                split_main_steps=split_main_steps,
                convergence_steps=total_steps,
            )

    trials_df = _load_trials(trials_db)
    metric_cols = REQUIRED_METRICS
    if not trials_df.empty:
        stats = trials_df[metric_cols + ["grade"]].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.99])
        stats_path = db_dir / "stats.csv"
        stats.to_csv(stats_path)

        agg_df = _update_points_agg(trials_df, points_agg_path, metric_cols)
        best = _select_best(agg_df)
        if best:
            param_vector = best.get("param_vector")
            best_row = trials_df[trials_df["param_vector"] == param_vector].iloc[0].to_dict()
            best_payload = {
                "param_vector": param_vector,
                "grade_mean": best.get("grade_mean"),
                "overrides": json.loads(best_row.get("overrides", "[]")),
                "metrics_mean": {k: best.get(f"{k}_mean") for k in metric_cols},
                "metrics_std": {k: best.get(f"{k}_std") for k in metric_cols},
                "ready_overrides": json.loads(best_row.get("overrides", "[]")),
            }
            _write_yaml(best_path, best_payload)

            best_params: Dict[str, Any] = {}
            override_map = {ov.split("=", 1)[0]: ov.split("=", 1)[1] for ov in best_payload["ready_overrides"]}
            for key in space.keys():
                override_key = _param_to_override(key, "VALUE").split("=", 1)[0]
                if override_key in override_map:
                    try:
                        best_params[key] = float(override_map[override_key])
                    except ValueError:
                        best_params[key] = override_map[override_key]
            suggestions = _suggest_next(space, best_params, rng, args.suggest_k, args.suggest_radius)
            with next_candidates_path.open("w", encoding="utf-8") as f:
                json.dump(suggestions, f, indent=2)

        index = {
            "metric_key": "grade",
            "reward_keys": space_cfg.get("reward_keys", []),
            "normalization_rules": {
                "grade": "w_vmax*vmax + w_nuplan*nuplan - w_collision*collision - w_overlap*overlap - w_oncoming*oncoming",
            },
            "schema_version": 1,
            "created_at": datetime.now().isoformat(),
        }
        with index_path.open("w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)

    _write_view_map_notebook(view_map_path)


if __name__ == "__main__":
    main()
