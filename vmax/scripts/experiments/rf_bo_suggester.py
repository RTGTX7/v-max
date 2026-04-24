#!/usr/bin/env python
"""Random-forest suggestion utility for reward-search experiments.

Canonical invocation:
- ``python -m vmax.scripts.experiments.rf_bo_suggester``
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import RandomForestRegressor
except Exception as exc:  # pragma: no cover
    raise SystemExit("scikit-learn is required. Install with: pip install scikit-learn") from exc


def _parse_override_value(overrides: str, key: str) -> float | None:
    if not isinstance(overrides, str):
        return None
    for part in overrides.split("|"):
        if part.startswith(key + "="):
            try:
                return float(part.split("=", 1)[1])
            except Exception:
                return None
    return None


def _extract_series(df: pd.DataFrame) -> pd.Series:
    if "overrides_str" in df.columns:
        return df["overrides_str"].astype(str)
    if "param_vector" in df.columns:
        return df["param_vector"].astype(str)
    if "overrides" in df.columns:
        # overrides is JSON list like ["a=1","b=2"]
        return df["overrides"].astype(str)
    return pd.Series([], dtype=str)


def load_data(
    trials_csv: Path,
    x_key: str,
    y_key: str,
    fail_as: float | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(trials_csv)
    # Resolve x/y from explicit columns or overrides_str
    if x_key in df.columns:
        df["x"] = pd.to_numeric(df[x_key], errors="coerce")
    else:
        overrides_series = _extract_series(df)
        df["x"] = overrides_series.apply(lambda s: _parse_override_value(s, x_key))
    if y_key in df.columns:
        df["y"] = pd.to_numeric(df[y_key], errors="coerce")
    else:
        overrides_series = _extract_series(df)
        df["y"] = overrides_series.apply(lambda s: _parse_override_value(s, y_key))

    # Grade
    if "grade" in df.columns:
        df["grade"] = pd.to_numeric(df["grade"], errors="coerce")

    # Keep failed rows for reporting but create valid set for fitting
    df_valid = df.copy()
    if fail_as is not None:
        df_valid["grade"] = df_valid["grade"].fillna(fail_as)
    df_valid = df_valid.dropna(subset=["x", "y", "grade"])

    return df, df_valid


def _to_space(x: np.ndarray, y: np.ndarray, log_x: bool, log_y: bool) -> Tuple[np.ndarray, np.ndarray]:
    if log_x:
        x = np.log10(x)
    if log_y:
        y = np.log10(y)
    return x, y


def _normalize(x: np.ndarray, low: float, high: float) -> np.ndarray:
    span = max(high - low, 1e-9)
    return (x - low) / span


def fit_rf(xy: np.ndarray, y: np.ndarray, seed: int, n_estimators: int = 200, min_samples_leaf: int = 2) -> RandomForestRegressor:
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        bootstrap=True,
        random_state=seed,
        n_jobs=-1,
    )
    rf.fit(xy, y)
    return rf


def sample_candidates(
    n: int,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    log_x: bool,
    log_y: bool,
    seed: int,
) -> np.ndarray:
    # Sample in transformed space, then convert back.
    if log_x:
        xmin_t, xmax_t = math.log10(xmin), math.log10(xmax)
    else:
        xmin_t, xmax_t = xmin, xmax
    if log_y:
        ymin_t, ymax_t = math.log10(ymin), math.log10(ymax)
    else:
        ymin_t, ymax_t = ymin, ymax

    rng = np.random.default_rng(seed)
    pts = None
    try:
        from scipy.stats import qmc  # type: ignore

        sampler = qmc.Sobol(d=2, scramble=True, seed=seed)
        pts = sampler.random(n)
        pts = qmc.scale(pts, [xmin_t, ymin_t], [xmax_t, ymax_t])
    except Exception:
        pts = rng.uniform([xmin_t, ymin_t], [xmax_t, ymax_t], size=(n, 2))

    if log_x:
        pts[:, 0] = np.power(10.0, pts[:, 0])
    if log_y:
        pts[:, 1] = np.power(10.0, pts[:, 1])
    return pts


def compute_ei(
    rf: RandomForestRegressor,
    candidates: np.ndarray,
    best: float,
    xi: float,
) -> Tuple[np.ndarray, np.ndarray]:
    # per-tree predictions
    tree_preds = np.stack([t.predict(candidates) for t in rf.estimators_], axis=0)
    std = tree_preds.std(axis=0)
    ei = np.maximum(tree_preds - best - xi, 0.0).mean(axis=0)
    return ei, std


def _min_dist_filter(
    candidates: np.ndarray,
    history: np.ndarray,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    log_x: bool,
    log_y: bool,
    min_dist: float,
) -> np.ndarray:
    if history.size == 0:
        return candidates

    # transform and normalize to [0,1] for distance
    cx, cy = _to_space(candidates[:, 0], candidates[:, 1], log_x, log_y)
    hx, hy = _to_space(history[:, 0], history[:, 1], log_x, log_y)

    if log_x:
        xmin_t, xmax_t = math.log10(xmin), math.log10(xmax)
    else:
        xmin_t, xmax_t = xmin, xmax
    if log_y:
        ymin_t, ymax_t = math.log10(ymin), math.log10(ymax)
    else:
        ymin_t, ymax_t = ymin, ymax

    cxn = _normalize(cx, xmin_t, xmax_t)
    cyn = _normalize(cy, ymin_t, ymax_t)
    hxn = _normalize(hx, xmin_t, xmax_t)
    hyn = _normalize(hy, ymin_t, ymax_t)

    kept = []
    for i in range(candidates.shape[0]):
        dx = cxn[i] - hxn
        dy = cyn[i] - hyn
        d = np.sqrt(dx * dx + dy * dy)
        if np.all(d >= min_dist):
            kept.append(candidates[i])
    return np.array(kept)


def pick_batch(
    candidates: np.ndarray,
    ei: np.ndarray,
    std: np.ndarray,
    batch_size: int,
    explore_frac: float,
) -> pd.DataFrame:
    n_explore = int(round(batch_size * explore_frac))
    n_exploit = max(batch_size - n_explore, 0)

    idx_ei = np.argsort(-ei)
    idx_std = np.argsort(-std)

    chosen = []
    used = set()

    for idx in idx_ei:
        if len(chosen) >= n_exploit:
            break
        if idx in used:
            continue
        used.add(idx)
        chosen.append((idx, "exploit"))

    for idx in idx_std:
        if len(chosen) >= batch_size:
            break
        if idx in used:
            continue
        used.add(idx)
        chosen.append((idx, "explore"))

    rows = []
    for idx, tag in chosen:
        rows.append(
            {
                "off_route": float(candidates[idx, 0]),
                "progression": float(candidates[idx, 1]),
                "tag": tag,
                "ei": float(ei[idx]),
                "uncertainty": float(std[idx]),
            }
        )
    return pd.DataFrame(rows)


def plot_maps(
    rf: RandomForestRegressor,
    df_hist: pd.DataFrame,
    df_sugg: pd.DataFrame,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    log_x: bool,
    log_y: bool,
    out_path: Path,
    grid_res: int = 100,
) -> None:
    import matplotlib.pyplot as plt

    xs = np.linspace(xmin, xmax, grid_res)
    ys = np.linspace(ymin, ymax, grid_res)
    X, Y = np.meshgrid(xs, ys)
    grid = np.c_[X.ravel(), Y.ravel()]

    if log_x:
        X_plot = np.log10(X)
    else:
        X_plot = X
    if log_y:
        Y_plot = np.log10(Y)
    else:
        Y_plot = Y

    tree_preds = np.stack([t.predict(grid) for t in rf.estimators_], axis=0)
    mean = tree_preds.mean(axis=0).reshape(grid_res, grid_res)
    std = tree_preds.std(axis=0).reshape(grid_res, grid_res)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # mean with points
    im0 = axes[0].contourf(X_plot, Y_plot, mean, levels=20, cmap="viridis")
    axes[0].scatter(
        np.log10(df_hist["x"]) if log_x else df_hist["x"],
        np.log10(df_hist["y"]) if log_y else df_hist["y"],
        s=20,
        c="white",
        edgecolors="black",
        label="history",
    )
    if not df_sugg.empty:
        is_exploit = df_sugg["tag"] == "exploit"
        axes[0].scatter(
            np.log10(df_sugg.loc[is_exploit, "off_route"]) if log_x else df_sugg.loc[is_exploit, "off_route"],
            np.log10(df_sugg.loc[is_exploit, "progression"]) if log_y else df_sugg.loc[is_exploit, "progression"],
            s=60,
            c="red",
            marker="o",
            label="suggest exploit",
        )
        axes[0].scatter(
            np.log10(df_sugg.loc[~is_exploit, "off_route"]) if log_x else df_sugg.loc[~is_exploit, "off_route"],
            np.log10(df_sugg.loc[~is_exploit, "progression"]) if log_y else df_sugg.loc[~is_exploit, "progression"],
            s=60,
            c="orange",
            marker="^",
            label="suggest explore",
        )
    axes[0].set_title("Surrogate mean")
    axes[0].legend(loc="best")
    fig.colorbar(im0, ax=axes[0])

    # uncertainty
    im1 = axes[1].contourf(X_plot, Y_plot, std, levels=20, cmap="magma")
    axes[1].set_title("Uncertainty (std)")
    fig.colorbar(im1, ax=axes[1])

    # EI map
    best = df_hist["grade"].max() if "grade" in df_hist.columns else 0.0
    ei = np.maximum(tree_preds - best, 0.0).mean(axis=0).reshape(grid_res, grid_res)
    im2 = axes[2].contourf(X_plot, Y_plot, ei, levels=20, cmap="plasma")
    axes[2].set_title("Expected Improvement")
    fig.colorbar(im2, ax=axes[2])

    for ax in axes:
        ax.set_xlabel("log10(off_route)" if log_x else "off_route")
        ax.set_ylabel("log10(progression)" if log_y else "progression")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials_csv", required=True)
    ap.add_argument("--out_csv", default="suggestions.csv")
    ap.add_argument("--batch_size", type=int, default=12)
    ap.add_argument("--candidates", type=int, default=5000)
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
    ap.add_argument("--fail_as", type=float, default=None)
    ap.add_argument("--constraint_col", type=str, default=None)
    ap.add_argument("--constraint_max", type=float, default=None)
    ap.add_argument("--map_png", default="map.png")
    ap.add_argument("--x_key", default="reward_config.off_route")
    ap.add_argument("--y_key", default="reward_config.progression")
    args = ap.parse_args()

    trials_csv = Path(args.trials_csv)
    df_raw, df_valid = load_data(trials_csv, args.x_key, args.y_key, fail_as=args.fail_as)

    if df_valid.empty:
        raise SystemExit("No valid rows after filtering. Check grade/x/y columns.")

    history = df_valid[["x", "y"]].to_numpy(dtype=float)

    # Filter by constraint if available
    if args.constraint_col and args.constraint_col in df_raw.columns and args.constraint_max is not None:
        df_valid = df_valid[df_valid[args.constraint_col] <= args.constraint_max]

    xy = df_valid[["x", "y"]].to_numpy(dtype=float)
    y = df_valid["grade"].to_numpy(dtype=float)

    # transform to model space
    mx, my = _to_space(xy[:, 0], xy[:, 1], args.log_x, args.log_y)
    XY = np.c_[mx, my]

    rf = fit_rf(XY, y, seed=args.seed, n_estimators=200, min_samples_leaf=3)

    candidates = sample_candidates(
        args.candidates,
        args.xmin,
        args.xmax,
        args.ymin,
        args.ymax,
        args.log_x,
        args.log_y,
        seed=args.seed,
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
        raise SystemExit("No candidates after min_dist filter. Lower min_dist or increase candidates.")

    cmx, cmy = _to_space(candidates[:, 0], candidates[:, 1], args.log_x, args.log_y)
    C = np.c_[cmx, cmy]

    best = float(np.nanmax(y))
    ei, std = compute_ei(rf, C, best=best, xi=args.xi)

    df_sugg = pick_batch(candidates, ei, std, args.batch_size, args.explore_frac)
    out_csv = Path(args.out_csv)
    df_sugg.to_csv(out_csv, index=False)

    print(df_sugg.to_string(index=False))

    plot_maps(
        rf,
        df_valid.assign(grade=y),
        df_sugg,
        args.xmin,
        args.xmax,
        args.ymin,
        args.ymax,
        args.log_x,
        args.log_y,
        Path(args.map_png),
    )


if __name__ == "__main__":
    main()
