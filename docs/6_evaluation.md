# V-Max Evaluation Guide

This document explains how to run evaluation in the current repository, what arguments matter, and where outputs are written.

## 1. Evaluation Entry Point

The main evaluation script is:

- `vmax/scripts/evaluate/evaluate.py`

It supports:

- learned policies
- expert replay
- rule-based actors such as IDM
- batched evaluation over logged scenarios
- optional rendering

## 2. What Evaluation Does

The evaluation workflow is:

1. load the dataset
2. build the simulator environment
3. load the requested actor or model
4. roll out each scenario in closed loop
5. collect per-scenario metrics
6. aggregate results and write them to disk

If rendering is enabled, the evaluation also writes video outputs.

## 3. Command Format

The script is typically launched as:

```bash
python -m vmax.scripts.evaluate.evaluate [arguments]
```

## 4. Main Arguments

| Argument | Meaning | Notes |
|---|---|---|
| `--sdc_actor` | actor type to evaluate | `ai`, `expert`, `idm`, ... |
| `--path_dataset` | dataset path or dataset alias | required for reproducible runs |
| `--path_model` | checkpoint path for learned policy | required when `--sdc_actor ai` |
| `--batch_size` | number of scenarios processed in parallel | keep small when rendering |
| `--render` | render evaluation videos | usually set `batch_size=1` |
| `--sdc_pov` | render from ego point of view | optional |
| `--scenario_indexes` | evaluate selected scenarios only | useful for debugging |
| `--eval_name` | output directory name | controls result folder naming |
| `--seed` | evaluation seed | useful for reproducibility |
| `--plot-failures` | replay failure cases from previous results | debugging tool |

## 5. Typical Commands

### Evaluate a trained policy

```bash
python -m vmax.scripts.evaluate.evaluate \
  --sdc_actor ai \
  --path_model /path/to/model_final.pkl \
  --path_dataset /path/to/eval.tfrecord \
  --batch_size 8
```

### Evaluate the expert policy

```bash
python -m vmax.scripts.evaluate.evaluate \
  --sdc_actor expert \
  --path_dataset /path/to/eval.tfrecord
```

### Render selected scenarios

```bash
python -m vmax.scripts.evaluate.evaluate \
  --sdc_actor ai \
  --path_model /path/to/model_final.pkl \
  --path_dataset /path/to/eval.tfrecord \
  --scenario_indexes 0 3 5 \
  --render \
  --batch_size 1
```

## 6. Output Files

The evaluation pipeline typically writes:

- `evaluation_episodes.csv`: per-scenario metrics
- `evaluation_results.txt`: aggregated summary metrics
- `mp4/`: rendered videos when rendering is enabled

These files are the main artifacts to keep when reporting experimental results.

## 7. Metrics Used in Evaluation

The exact metric set depends on the collector configuration, but the standard workflow includes:

- safety metrics such as off-road, overlap, at-fault collision, and red-light violations
- progress metrics such as progression and progress ratio
- comfort and TTC-related metrics
- aggregate benchmark scores such as V-Max and nuPlan-style summaries

For metric definitions, see:

- `docs/5_metrics.md`

## 8. Practical Recommendations

- keep `batch_size=1` when rendering
- always record the exact `path_model` and `path_dataset` used for published results
- use `scenario_indexes` for targeted debugging before launching a full benchmark run
- treat `evaluation_results.txt` as the summary artifact and `evaluation_episodes.csv` as the diagnostic artifact

## 9. What to Report in a Paper

For a reproducible academic result, record at least:

- the checkpoint path or checkpoint selection rule
- the evaluation dataset or split
- the actor type
- the batch size
- whether rendering or noisy initialization was enabled
- the exact command used

Without this information, other users will have difficulty reproducing the evaluation.
