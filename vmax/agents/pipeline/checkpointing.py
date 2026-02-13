# Copyright 2025 Valeo.

"""Checkpoint utilities for training pipelines."""

from __future__ import annotations

import pickle
from typing import Any

import flax
from etils import epath

class CheckpointError(RuntimeError):
    """Raised when checkpoint loading fails or is incompatible."""


def save_checkpoint(path: str, checkpoint: dict) -> None:
    """Serialize and save a checkpoint dict to disk."""
    with epath.Path(path).open("wb") as fout:
        fout.write(pickle.dumps(checkpoint))


def load_checkpoint(path: str) -> dict:
    """Load a checkpoint from disk, handling legacy params-only files."""
    with epath.Path(path).open("rb") as fin:
        obj = pickle.loads(fin.read())

    if isinstance(obj, dict) and "model_state_dict" in obj:
        return obj

    # Legacy: assume params-only file.
    return {
        "model_state_dict": obj,
        "optimizer_state_dict": None,
        "trainer_state": None,
        "rng_state": None,
        "replay_buffer_state": None,
        "format": "params_only",
    }


def merge_params(target: Any, source: Any, strict: bool = True) -> tuple[Any, dict]:
    """Merge source params into target params, optionally enforcing strict matches.

    Returns the merged params and a report dict with loaded/skipped keys.
    """
    target_flat, target_is_frozen = _flatten_params(target)
    source_flat, _ = _flatten_params(source)

    loaded_keys: list[tuple[str, ...]] = []
    missing_keys: list[tuple[str, ...]] = []
    mismatched_keys: list[tuple[str, ...]] = []

    for key, target_value in target_flat.items():
        if key not in source_flat:
            missing_keys.append(key)
            continue

        source_value = source_flat[key]
        if _compatible_leaf(target_value, source_value):
            target_flat[key] = source_value
            loaded_keys.append(key)
        else:
            mismatched_keys.append(key)

    extra_keys = [key for key in source_flat.keys() if key not in target_flat]

    if strict and (missing_keys or mismatched_keys or extra_keys):
        examples = {
            "missing": summarize_keys(missing_keys, limit=5),
            "mismatched": summarize_keys(mismatched_keys, limit=5),
            "extra": summarize_keys(extra_keys, limit=5),
        }
        raise CheckpointError(
            "Checkpoint params mismatch: "
            f"missing={len(missing_keys)}, mismatched={len(mismatched_keys)}, extra={len(extra_keys)}. "
            f"Examples: {examples}. "
            "If this is a warm-start or architecture mismatch, set resume.strict=false.",
        )

    merged = _unflatten_params(target_flat, target_is_frozen)
    report = {
        "loaded_keys": loaded_keys,
        "missing_keys": missing_keys,
        "mismatched_keys": mismatched_keys,
        "extra_keys": extra_keys,
    }
    return merged, report


def _compatible_leaf(target_value: Any, source_value: Any) -> bool:
    """Return True if source_value can replace target_value."""
    if hasattr(target_value, "shape") and hasattr(source_value, "shape"):
        return (
            target_value.shape == source_value.shape
            and getattr(target_value, "dtype", None) == getattr(source_value, "dtype", None)
        )
    return type(target_value) is type(source_value)


def _flatten_params(params: Any) -> tuple[dict, bool]:
    """Flatten params to a dict with tuple keys."""
    is_frozen = isinstance(params, flax.core.frozen_dict.FrozenDict)
    if is_frozen:
        params = flax.core.unfreeze(params)

    flat = flax.traverse_util.flatten_dict(params, sep=None)
    return flat, is_frozen


def _unflatten_params(flat: dict, is_frozen: bool) -> Any:
    """Reconstruct params from a flattened dict."""
    params = flax.traverse_util.unflatten_dict(flat)
    if is_frozen:
        return flax.core.freeze(params)
    return params


def summarize_keys(keys: list[tuple[str, ...]], limit: int = 10) -> list[str]:
    """Summarize flattened keys for logging."""
    if not keys:
        return []
    trimmed = keys[:limit]
    return ["/".join(key) for key in trimmed]
