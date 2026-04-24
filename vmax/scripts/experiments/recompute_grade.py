#!/usr/bin/env python3
"""Recompute grade in trials_long.csv and regenerate aggregates.

Canonical invocation:
- ``python -m vmax.scripts.experiments.recompute_grade``
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map_db", required=True)
    parser.add_argument("--out_suffix", default="regraded")
    parser.add_argument("--w_vmax", type=float, default=0.5)
    parser.add_argument("--w_nuplan", type=float, default=0.5)
    parser.add_argument("--w_collision", type=float, default=0.0)
    parser.add_argument("--w_overlap", type=float, default=0.0)
    parser.add_argument("--w_oncoming", type=float, default=0.0)
    args = parser.parse_args()

    map_db = Path(args.map_db)
    trials_path = map_db / "trials_long.csv"
    if not trials_path.exists():
        raise FileNotFoundError(trials_path)

    df = pd.read_csv(trials_path)
    for col in [
        "vmax_aggregate_score",
        "nuplan_aggregate_score",
        "at_fault_collision",
        "overlap",
        "distance_into_oncoming_traffic",
    ]:
        if col not in df.columns:
            df[col] = float("nan")

    df["grade"] = (
        args.w_vmax * df["vmax_aggregate_score"]
        + args.w_nuplan * df["nuplan_aggregate_score"]
        - args.w_collision * df["at_fault_collision"]
        - args.w_overlap * df["overlap"]
        - args.w_oncoming * df["distance_into_oncoming_traffic"]
    )

    out_trials = map_db / f"trials_long_{args.out_suffix}.csv"
    df.to_csv(out_trials, index=False)

    metric_cols = [
        "vmax_aggregate_score",
        "nuplan_aggregate_score",
        "at_fault_collision",
        "overlap",
        "distance_into_oncoming_traffic",
    ]
    agg = df.groupby(["param_vector"])[metric_cols + ["grade"]].agg(["mean", "std"]).reset_index()
    agg.columns = ["_".join([c for c in col if c]) if isinstance(col, tuple) else str(col) for col in agg.columns]
    out_agg = map_db / f"points_agg_{args.out_suffix}.csv"
    agg.to_csv(out_agg, index=False)

    best = None
    if not agg.empty and "grade_mean" in agg.columns:
        best_row = agg.sort_values("grade_mean", ascending=False).iloc[0].to_dict()
        best = best_row

    if best:
        best_payload = {
            "param_vector": best.get("param_vector"),
            "grade_mean": best.get("grade_mean"),
            "metrics_mean": {k: best.get(f"{k}_mean") for k in metric_cols},
            "metrics_std": {k: best.get(f"{k}_std") for k in metric_cols},
        }
        out_best = map_db / f"best_{args.out_suffix}.yaml"
        out_best.write_text(yaml.safe_dump(best_payload, sort_keys=False))

    print("Wrote:", out_trials)
    print("Wrote:", out_agg)
    if best:
        print("Wrote:", map_db / f"best_{args.out_suffix}.yaml")


if __name__ == "__main__":
    main()
