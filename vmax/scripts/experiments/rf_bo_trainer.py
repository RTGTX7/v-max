#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from rf_bo_suggester import (
    load_data,
    fit_rf,
    sample_candidates,
    _min_dist_filter,
    _to_space,
    compute_ei,
    pick_batch,
)


@dataclass
class GradeWeights:
    vmax: float = 0.5
    nuplan: float = 0.5
    collision: float = 0.0
    overlap: float = 0.0
    oncoming: float = 0.0


def _parse_metric_from_log(path: Path) -> Dict[str, float]:
    metrics = {}
    if not path.exists():
        return metrics
    pattern = re.compile(r"-\s*([A-Za-z0-9_]+)\s*:\s*([-+eE0-9.]+)")
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                key = m.group(1)
                try:
                    val = float(m.group(2))
                except Exception:
                    continue
                metrics[key] = val
    return metrics


def _compute_grade(metrics: Dict[str, float], w: GradeWeights) -> float | None:
    if not metrics:
        return None
    vmax = metrics.get("vmax_aggregate_score")
    nu = metrics.get("nuplan_aggregate_score")
    coll = metrics.get("at_fault_collision")
    overlap = metrics.get("overlap")
    oncoming = metrics.get("distance_into_oncoming_traffic")
    if vmax is None or nu is None:
        return None
    grade = w.vmax * vmax + w.nuplan * nu
    if coll is not None:
        grade -= w.collision * coll
    if overlap is not None:
        grade -= w.overlap * overlap
    if oncoming is not None:
        grade -= w.oncoming * oncoming
    return grade


def _write_trials_row(out_csv: Path, row: Dict[str, object]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    exists = out_csv.exists()
    with out_csv.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _write_result_json(out_path: Path, row: Dict[str, object]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "trial": row,
        "metric_mapping": {
            "vmax_aggregate_score": "vmax_aggregate_score",
            "nuplan_aggregate_score": "nuplan_aggregate_score",
            "at_fault_collision": "at_fault_collision",
            "overlap": "overlap",
            "distance_into_oncoming_traffic": "distance_into_oncoming_traffic",
        },
    }
    out_path.write_text(json.dumps(payload, indent=2))


def _run_training(
    repo_root: Path,
    run_dir: Path,
    name_exp: str,
    overrides: Iterable[str],
    total_steps: int,
    algorithm: str,
    seed: int,
    resume_ckpt: Path,
    extra_overrides: Iterable[str],
) -> Path:
    cmd = [
        sys.executable,
        "-m",
        "vmax.scripts.training.train",
        f"algorithm={algorithm}",
        f"seed={seed}",
        f"name_exp={name_exp}",
        f"name_run={run_dir.name}",
        f"total_timesteps={total_steps}",
        "resume.enabled=true",
        f"resume.ckpt_path={resume_ckpt}",
        "resume.mode=weights_only",
        "resume.strict=false",
    ]
    cmd.extend(list(overrides))
    cmd.extend(list(extra_overrides))
    env = os.environ.copy()
    subprocess.run(cmd, cwd=str(repo_root), env=env, check=True)
    # train log path
    return run_dir / "train.log"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_run_dir", required=True)
    ap.add_argument("--ckpt_path", required=True)
    ap.add_argument("--trials_csv", default=None)
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=12)
    ap.add_argument("--candidates", type=int, default=5000)
    ap.add_argument("--total_timesteps", type=int, default=50_000_000)
    ap.add_argument("--xmin", type=float, default=0.2)
    ap.add_argument("--xmax", type=float, default=3.0)
    ap.add_argument("--ymin", type=float, default=0.2)
    ap.add_argument("--ymax", type=float, default=3.0)
    ap.add_argument("--log_x", action="store_true")
    ap.add_argument("--log_y", action="store_true", default=True)
    ap.add_argument("--xi", type=float, default=0.01)
    ap.add_argument("--explore_frac", type=float, default=0.3)
    ap.add_argument("--min_dist", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--extra_overrides", nargs="*", default=[])
    ap.add_argument("--grade_vmax", type=float, default=0.5)
    ap.add_argument("--grade_nuplan", type=float, default=0.5)
    ap.add_argument("--grade_collision", type=float, default=0.0)
    ap.add_argument("--grade_overlap", type=float, default=0.0)
    ap.add_argument("--grade_oncoming", type=float, default=0.0)
    ap.add_argument("--algorithm", default="ppo")
    ap.add_argument("--x_key", default="reward_config.off_route")
    ap.add_argument("--y_key", default="reward_config.progression")
    ap.add_argument("--write_results_json", action="store_true", default=True)
    args = ap.parse_args()

    # Resolve repo root by walking upward until /vmax exists
    repo_root = None
    for parent in [Path.cwd()] + list(Path.cwd().parents):
        if (parent / "vmax").exists():
            repo_root = parent
            break
    if repo_root is None:
        raise SystemExit("Could not resolve repo root (missing /vmax).")
    root_run_dir = Path(args.root_run_dir)
    ckpt_path = Path(args.ckpt_path)

    trials_csv = Path(args.trials_csv) if args.trials_csv else root_run_dir / "map_db" / "trials_long.csv"

    grade_w = GradeWeights(
        vmax=args.grade_vmax,
        nuplan=args.grade_nuplan,
        collision=args.grade_collision,
        overlap=args.grade_overlap,
        oncoming=args.grade_oncoming,
    )

    for round_idx in range(1, args.rounds + 1):
        # build model from current trials
        df_raw, df_valid = load_data(trials_csv, args.x_key, args.y_key, fail_as=None)
        if df_valid.empty:
            raise SystemExit("No valid rows to fit RF. Add some trials first.")

        history = df_valid[["x", "y"]].to_numpy(dtype=float)
        mx, my = _to_space(history[:, 0], history[:, 1], args.log_x, args.log_y)
        XY = np.c_[mx, my]
        rf = fit_rf(XY, df_valid["grade"].to_numpy(dtype=float), seed=args.seed)

        candidates = sample_candidates(
            args.candidates,
            args.xmin,
            args.xmax,
            args.ymin,
            args.ymax,
            args.log_x,
            args.log_y,
            seed=args.seed + round_idx,
        )
        candidates = _min_dist_filter(
            candidates,
            history,
            args.xmin,
            args.xmax,
            args.ymin,
            args.ymax,
            args.log_x,
            args.log_y,
            args.min_dist,
        )
        if candidates.size == 0:
            raise SystemExit("No candidates after min_dist filter.")
        cmx, cmy = _to_space(candidates[:, 0], candidates[:, 1], args.log_x, args.log_y)
        C = np.c_[cmx, cmy]
        best = float(np.nanmax(df_valid["grade"].to_numpy(dtype=float)))
        ei, std = compute_ei(rf, C, best=best, xi=args.xi)
        df_sugg = pick_batch(candidates, ei, std, args.batch_size, args.explore_frac)

        # Run each suggestion
        sweep_dir = root_run_dir / "stage2_sweeps" / f"rfbo_round_{round_idx:03d}"
        sweep_dir.mkdir(parents=True, exist_ok=True)
        results_dir = sweep_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        for i, row in df_sugg.iterrows():
            run_dir = sweep_dir / f"cand_{i:03d}"
            run_dir.mkdir(parents=True, exist_ok=True)

            overrides = [
                f"reward_config.off_route={row['off_route']}",
                f"reward_config.progression={row['progression']}",
            ]

            try:
                log_path = _run_training(
                    repo_root,
                    run_dir,
                    str(sweep_dir),
                    overrides,
                    args.total_timesteps,
                    args.algorithm,
                    args.seed,
                    ckpt_path,
                    args.extra_overrides,
                )
                metrics = _parse_metric_from_log(log_path)
                grade = _compute_grade(metrics, grade_w)
                status = "completed" if grade is not None else "failed"
            except subprocess.CalledProcessError:
                metrics = {}
                grade = None
                status = "failed"

            trials_row = {
                "round": round_idx,
                "status": status,
                "algorithm": args.algorithm,
                "seed": args.seed,
                "total_timesteps": args.total_timesteps,
                "overrides": json.dumps(overrides),
                "overrides_str": "|".join(overrides),
                "train_log": str((run_dir / "train.log").resolve()),
                "branch_dir": str(run_dir.resolve()),
                "vmax_aggregate_score": metrics.get("vmax_aggregate_score"),
                "nuplan_aggregate_score": metrics.get("nuplan_aggregate_score"),
                "at_fault_collision": metrics.get("at_fault_collision"),
                "overlap": metrics.get("overlap"),
                "distance_into_oncoming_traffic": metrics.get("distance_into_oncoming_traffic"),
                "grade": grade,
            }
            _write_trials_row(trials_csv, trials_row)
            if args.write_results_json:
                _write_result_json(results_dir / f"{run_dir.name}.json", trials_row)


if __name__ == "__main__":
    main()
