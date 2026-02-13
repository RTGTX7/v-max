#!/usr/bin/env python
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import re
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vmax.scripts.evaluate.evaluate import run_evaluation
from vmax.scripts.evaluate import utils
from vmax.simulator import datasets, make_data_generator, make_env_for_evaluation
from waymax import dynamics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate all BC model checkpoints in one run directory.")
    parser.add_argument("--run_dir", required=True, help="Path to BC run directory (contains model/ and .hydra/).")
    parser.add_argument(
        "--path_dataset",
        default=None,
        help="Dataset path/name. If omitted, use path_dataset_eval from run config, then fallback to local_womd_valid.",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Evaluation batch size.")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of model-eval threads.")
    parser.add_argument(
        "--skip_completed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip checkpoints already marked completed in existing summary.csv (default: enabled).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--max_num_objects", type=int, default=None, help="Override max objects from run config.")
    parser.add_argument("--waymo_dataset", action="store_true", help="Use Waymo dataset mode.")
    parser.add_argument("--scenario_indexes", nargs="*", type=int, default=None, help="Optional scenario index subset.")
    parser.add_argument("--out_dir", default=None, help="Optional output directory (default: <run_dir>/eval_all_bc).")
    return parser.parse_args()


def _extract_step(path: Path) -> int:
    match = re.search(r"model_(\d+)\.pkl$", path.name)
    if match:
        return int(match.group(1))
    if path.name == "model_final.pkl":
        return 10**18
    return -1


def _iter_model_files(model_dir: Path) -> list[Path]:
    files = list(model_dir.glob("model_*.pkl"))
    final_file = model_dir / "model_final.pkl"
    # model_*.pkl already matches model_final.pkl, so avoid adding it twice.
    if final_file.exists() and final_file not in files:
        files.append(final_file)
    return sorted(files, key=_extract_step)


def _mean_metrics(metrics: dict[str, list]) -> dict[str, float]:
    out = {}
    for key, values in metrics.items():
        if not values:
            continue
        out[key] = float(np.mean(values))
    return out


def _load_existing_rows(summary_path: Path) -> dict[str, dict[str, object]]:
    if not summary_path.is_file():
        return {}
    with summary_path.open("r", newline="", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        return {row["model_file"]: row for row in reader if row.get("model_file")}


def _row_step(row: dict[str, object]) -> int:
    try:
        return int(row.get("model_step", -1))
    except (TypeError, ValueError):
        return -1


def _evaluate_one_model(
    model_path: Path,
    eval_config: dict,
    max_num_objects: int,
    include_sdc_paths: bool,
    dataset_path: str,
    batch_size: int,
    scenario_indexes: list[int] | None,
    seed: int,
    out_root: Path,
) -> dict[str, object]:
    env = make_env_for_evaluation(
        max_num_objects=max_num_objects,
        dynamics_model=dynamics.InvertibleBicycleModel(normalize_actions=True),
        sdc_paths_from_data=include_sdc_paths,
        observation_type=eval_config["observation_type"],
        observation_config=eval_config["observation_config"],
        termination_keys=eval_config["termination_keys"],
        noisy_init=False,
    )

    policy = utils.load_model(env, eval_config["algorithm"]["name"], eval_config, str(model_path))
    step_fn = utils.make_step_fn(env, "ai", policy)

    data_generator = make_data_generator(
        path=dataset_path,
        max_num_objects=max_num_objects,
        include_sdc_paths=include_sdc_paths,
        batch_dims=(batch_size, 1),
        seed=seed,
        repeat=1,
    )

    run_path = out_root / model_path.stem
    run_path.mkdir(parents=True, exist_ok=True)
    eval_metrics = run_evaluation(
        env=env,
        data_generator=data_generator,
        step_fn=step_fn,
        run_path=str(run_path),
        scenario_indexes=scenario_indexes,
        termination_keys=eval_config["termination_keys"],
        render=False,
        render_pov=False,
        seed=seed,
        batch_size=batch_size,
        plot_failures=False,
    )
    metric_mean = _mean_metrics(eval_metrics or {})
    return {
        "model_file": model_path.name,
        "model_step": _extract_step(model_path),
        "status": "completed",
        **metric_mean,
    }


def main() -> None:
    args = _parse_args()

    run_dir = Path(args.run_dir).resolve()
    model_dir = run_dir / "model"
    config_path = run_dir / ".hydra" / "config.yaml"
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"Model directory not found: {model_dir}. "
            f"Check --run_dir points to a training run containing 'model/'.",
        )
    if not config_path.is_file():
        raise FileNotFoundError(f"Hydra config not found: {config_path}")

    eval_config = utils.load_yaml_config(str(config_path))
    # Keep the same layout expected by evaluate/utils.load_model.
    eval_config["encoder"] = eval_config["network"]["encoder"]
    eval_config["policy"] = eval_config["algorithm"]["network"]["policy"]
    eval_config["value"] = eval_config["algorithm"]["network"]["value"]
    eval_config["unflatten_config"] = eval_config["observation_config"]
    eval_config["action_distribution"] = eval_config["algorithm"]["network"]["action_distribution"]
    max_num_objects = args.max_num_objects if args.max_num_objects is not None else int(eval_config["max_num_objects"])
    include_sdc_paths = not args.waymo_dataset

    dataset_name_or_path = args.path_dataset or eval_config.get("path_dataset_eval") or "local_womd_valid"
    dataset_path = datasets.get_dataset(dataset_name_or_path)
    model_files = _iter_model_files(model_dir)
    if not model_files:
        raise FileNotFoundError(f"No model_*.pkl or model_final.pkl found in {model_dir}")

    out_root = Path(args.out_dir).resolve() if args.out_dir else run_dir / "eval_all_bc"
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "summary.csv"

    row_map = _load_existing_rows(summary_path)
    if args.skip_completed and row_map:
        before_count = len(model_files)
        completed_files = {
            model_file for model_file, row in row_map.items() if str(row.get("status", "")).strip() == "completed"
        }
        model_files = [model_path for model_path in model_files if model_path.name not in completed_files]
        skipped_count = before_count - len(model_files)
        if skipped_count > 0:
            print(f"-> Skipping {skipped_count} completed checkpoints from existing summary.")

    num_workers = max(1, int(args.num_workers))
    print(f"-> Evaluating {len(model_files)} checkpoints with num_workers={num_workers}")

    if num_workers == 1:
        for idx, model_path in enumerate(model_files):
            print(f"-> Evaluating {model_path.name}")
            try:
                result = _evaluate_one_model(
                    model_path=model_path,
                    eval_config=eval_config,
                    max_num_objects=max_num_objects,
                    include_sdc_paths=include_sdc_paths,
                    dataset_path=dataset_path,
                    batch_size=args.batch_size,
                    scenario_indexes=args.scenario_indexes,
                    seed=args.seed + idx,
                    out_root=out_root,
                )
                row_map[model_path.name] = result
                print(f"-> Finished {model_path.name}")
            except Exception as exc:
                print(f"-> Failed {model_path.name}: {exc}")
                row_map[model_path.name] = {
                    "model_file": model_path.name,
                    "model_step": _extract_step(model_path),
                    "status": "failed",
                    "error": str(exc),
                }
    else:
        futures = {}
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for idx, model_path in enumerate(model_files):
                future = executor.submit(
                    _evaluate_one_model,
                    model_path=model_path,
                    eval_config=eval_config,
                    max_num_objects=max_num_objects,
                    include_sdc_paths=include_sdc_paths,
                    dataset_path=dataset_path,
                    batch_size=args.batch_size,
                    scenario_indexes=args.scenario_indexes,
                    seed=args.seed + idx,
                    out_root=out_root,
                )
                futures[future] = model_path

            for future in as_completed(futures):
                model_path = futures[future]
                try:
                    row_map[model_path.name] = future.result()
                    print(f"-> Finished {model_path.name}")
                except Exception as exc:
                    print(f"-> Failed {model_path.name}: {exc}")
                    row_map[model_path.name] = {
                        "model_file": model_path.name,
                        "model_step": _extract_step(model_path),
                        "status": "failed",
                        "error": str(exc),
                    }

    rows = sorted(row_map.values(), key=_row_step)

    all_keys = sorted({k for row in rows for k in row.keys()})
    with summary_path.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(rows)

    print(f"-> Wrote summary: {summary_path}")
    if rows:
        rank_key = "vmax_aggregate_score" if any("vmax_aggregate_score" in row for row in rows) else "accuracy"
        ranked_input = [row for row in rows if row.get("status") == "completed" and row.get(rank_key) is not None]
        ranked = sorted(ranked_input, key=lambda item: float(item.get(rank_key, -1e9)), reverse=True)[:5]
        print(f"-> Top 5 by {rank_key}:")
        for idx, item in enumerate(ranked, start=1):
            print(f"   {idx}. {item['model_file']}  {rank_key}={item.get(rank_key)}")


if __name__ == "__main__":
    main()
