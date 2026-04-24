#!/usr/bin/env python
"""Rebuild trial tables from stored sweep outputs.

Canonical invocation:
- ``python -m vmax.scripts.experiments.rebuild_trials_from_results``
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def _parse_metrics_from_log(path: Path) -> dict:
    metrics = {}
    if not path.exists():
        return metrics
    pattern = re.compile(r"-\s*([A-Za-z0-9_]+)\s*:\s*([-+eE0-9.]+)")
    for line in path.read_text(errors="ignore").splitlines():
        m = pattern.search(line)
        if m:
            key = m.group(1)
            try:
                val = float(m.group(2))
            except Exception:
                continue
            metrics[key] = val
    return metrics


def _compute_grade(metrics: dict, w_vmax: float, w_nuplan: float) -> float | None:
    vmax = metrics.get("vmax_aggregate_score")
    nu = metrics.get("nuplan_aggregate_score")
    if vmax is None or nu is None:
        return None
    return w_vmax * vmax + w_nuplan * nu


def _extract_overrides_from_hydra(hydra_cfg: Path) -> list[str]:
    if not hydra_cfg.exists():
        return []
    text = hydra_cfg.read_text(errors="ignore")
    overrides = []
    for key in ["off_route", "progression"]:
        m = re.search(rf"{key}:\s*([0-9eE.+-]+)", text)
        if m:
            overrides.append(f"reward_config.{key}={m.group(1)}")
    m = re.search(r"encoder:\s*([A-Za-z0-9_]+)", text)
    if m:
        overrides.append(f"network/encoder={m.group(1)}")
    return overrides


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_run_dir", required=True)
    ap.add_argument("--out_csv", default=None)
    ap.add_argument("--w_vmax", type=float, default=0.5)
    ap.add_argument("--w_nuplan", type=float, default=0.5)
    args = ap.parse_args()

    root = Path(args.root_run_dir)
    out_csv = Path(args.out_csv) if args.out_csv else root / "map_db" / "trials_long.csv"

    rows = []

    # 1) Use existing sweep results JSONs (grid search)
    for res in root.glob("stage2_sweeps/sweep_*/results/*.json"):
        try:
            data = json.loads(res.read_text())
            trial = data.get("trial", {})
            if trial:
                rows.append(trial)
        except Exception:
            continue

    # 2) Add rfbo_round_* candidates using train.log + hydra config
    for log_path in root.glob("stage2_sweeps/rfbo_round_*/cand_*/train.log"):
        metrics = _parse_metrics_from_log(log_path)
        grade = _compute_grade(metrics, args.w_vmax, args.w_nuplan)
        overrides = _extract_overrides_from_hydra(log_path.parent / ".hydra" / "config.yaml")
        row = {
            "trial_id": log_path.parent.name,
            "status": "completed" if grade is not None else "failed",
            "algorithm": "ppo",
            "seed": 0,
            "total_timesteps": 50000000,
            "ckpt_signature": "",
            "overrides": json.dumps(overrides),
            "param_vector": "|".join(sorted(overrides)),
            "log_path": str(log_path),
            "branch_dir": str(log_path.parent),
            "timestamp": "",
            "vmax_aggregate_score": metrics.get("vmax_aggregate_score"),
            "nuplan_aggregate_score": metrics.get("nuplan_aggregate_score"),
            "at_fault_collision": metrics.get("at_fault_collision"),
            "overlap": metrics.get("overlap"),
            "distance_into_oncoming_traffic": metrics.get("distance_into_oncoming_traffic"),
            "grade": grade,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print("Wrote:", out_csv)


if __name__ == "__main__":
    main()
