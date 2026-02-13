#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml


METRIC_COLUMNS = [
    "vmax_aggregate_score",
    "nuplan_aggregate_score",
    "at_fault_collision",
    "overlap",
    "distance_into_oncoming_traffic",
]


def _load_trials(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def _as_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _compute_grade(
    frame: pd.DataFrame,
    w_vmax: float,
    w_nuplan: float,
    w_collision: float,
    w_overlap: float,
    w_oncoming: float,
) -> pd.Series:
    for key in METRIC_COLUMNS:
        if key not in frame.columns:
            frame[key] = float("nan")
        frame[key] = _as_float(frame[key])
    return (
        w_vmax * frame["vmax_aggregate_score"]
        + w_nuplan * frame["nuplan_aggregate_score"]
        - w_collision * frame["at_fault_collision"]
        - w_overlap * frame["overlap"]
        - w_oncoming * frame["distance_into_oncoming_traffic"]
    )


def _dedupe_trials(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    if "timestamp" not in frame.columns:
        frame["timestamp"] = ""
    if "trial_id" not in frame.columns:
        frame["trial_id"] = frame.index.astype(str)
    frame = frame.copy()
    frame["__timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame = frame.sort_values(by=["trial_id", "__timestamp"]).drop_duplicates(subset=["trial_id"], keep="last")
    return frame.drop(columns=["__timestamp"])


def _build_points_agg(trials: pd.DataFrame) -> pd.DataFrame:
    if trials.empty:
        return pd.DataFrame()
    if "param_vector" not in trials.columns:
        trials["param_vector"] = ""
    cols = [c for c in METRIC_COLUMNS + ["grade"] if c in trials.columns]
    if not cols:
        return pd.DataFrame()
    agg = trials.groupby(["param_vector"])[cols].agg(["mean", "std"]).reset_index()
    agg.columns = ["_".join([x for x in col if x]) if isinstance(col, tuple) else str(col) for col in agg.columns]
    return agg


def _save_best(points_agg: pd.DataFrame, out_path: Path) -> None:
    if points_agg.empty or "grade_mean" not in points_agg.columns:
        out_path.write_text(yaml.safe_dump({"best": None}, sort_keys=False), encoding="utf-8")
        return
    row = points_agg.sort_values("grade_mean", ascending=False).iloc[0].to_dict()
    payload = {
        "param_vector": row.get("param_vector"),
        "grade_mean": row.get("grade_mean"),
        "metrics_mean": {m: row.get(f"{m}_mean") for m in METRIC_COLUMNS},
        "metrics_std": {m: row.get(f"{m}_std") for m in METRIC_COLUMNS},
        "updated_at": datetime.now().isoformat(),
    }
    out_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _save_index(
    out_path: Path,
    source_map_dbs: list[str],
    weights: dict,
    trials_count: int,
    points_count: int,
) -> None:
    payload = {
        "schema_version": 1,
        "updated_at": datetime.now().isoformat(),
        "source_map_dbs": source_map_dbs,
        "metric_columns": METRIC_COLUMNS,
        "grade_weights": weights,
        "num_trials": trials_count,
        "num_points": points_count,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge multi-machine sweep map_db into one canonical map_db.")
    parser.add_argument("--target_map_db", required=True, help="Path to canonical map_db directory.")
    parser.add_argument(
        "--source_map_db",
        nargs="+",
        required=True,
        help="One or more source map_db directories to merge from.",
    )
    parser.add_argument("--include_target_existing", action="store_true")
    parser.add_argument("--w_vmax", type=float, default=0.5)
    parser.add_argument("--w_nuplan", type=float, default=0.5)
    parser.add_argument("--w_collision", type=float, default=0.0)
    parser.add_argument("--w_overlap", type=float, default=0.0)
    parser.add_argument("--w_oncoming", type=float, default=0.0)
    args = parser.parse_args()

    target = Path(args.target_map_db).resolve()
    target.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    source_paths: list[str] = []

    if args.include_target_existing:
        existing = _load_trials(target / "trials_long.csv")
        if not existing.empty:
            frames.append(existing)
            source_paths.append(str(target))

    for source_dir in args.source_map_db:
        source = Path(source_dir).resolve()
        trials = _load_trials(source / "trials_long.csv")
        if not trials.empty:
            frames.append(trials)
            source_paths.append(str(source))

    if not frames:
        raise RuntimeError("No trials_long.csv rows found in any provided map_db directories.")

    merged = pd.concat(frames, ignore_index=True)
    merged["grade"] = _compute_grade(
        merged,
        w_vmax=args.w_vmax,
        w_nuplan=args.w_nuplan,
        w_collision=args.w_collision,
        w_overlap=args.w_overlap,
        w_oncoming=args.w_oncoming,
    )
    merged = _dedupe_trials(merged)

    trials_out = target / "trials_long.csv"
    merged.to_csv(trials_out, index=False)

    points_agg = _build_points_agg(merged)
    points_out = target / "points_agg.csv"
    points_agg.to_csv(points_out, index=False)

    _save_best(points_agg, target / "best.yaml")
    _save_index(
        target / "index.json",
        source_map_dbs=source_paths,
        weights={
            "w_vmax": args.w_vmax,
            "w_nuplan": args.w_nuplan,
            "w_collision": args.w_collision,
            "w_overlap": args.w_overlap,
            "w_oncoming": args.w_oncoming,
        },
        trials_count=int(len(merged)),
        points_count=int(len(points_agg)),
    )

    print(f"Wrote: {trials_out}")
    print(f"Wrote: {points_out}")
    print(f"Wrote: {target / 'best.yaml'}")
    print(f"Wrote: {target / 'index.json'}")
    print(f"Merged trials: {len(merged)}")
    print(f"Unique param points: {len(points_agg)}")


if __name__ == "__main__":
    main()
