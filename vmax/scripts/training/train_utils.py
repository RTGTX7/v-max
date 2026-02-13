# Copyright 2025 Valeo.


"""Utility functions for training scripts."""

import logging
import os
import pickle
from argparse import ArgumentParser
from datetime import datetime
from typing import Any

import jax
from etils import epath
from tensorboardX import SummaryWriter

from vmax.simulator import datasets


logger = logging.getLogger(__name__)


def resolve_output_dir(
    algorithm_name: str,
    observation_type: str,
    encoder_config: dict,
    name_run: str,
    name_exp: str,
) -> str:
    """Determine the output directory based on training parameters.

    Args:
        algorithm_name: The name of the algorithm.
        observation_type: The observation type.
        reward_type: The reward type.
        encoder_config: The encoder configuration.
        name_run: The run name.
        name_exp: The experiment name.

    Returns:
        The output directory path.

    """
    _name_exp = "runs" if name_exp is None else name_exp

    if name_run is None:
        _name_run = f"{algorithm_name.upper()}_{observation_type.upper()}"

        encoder_type = encoder_config["type"]
        if encoder_type != "none":
            _name_run += f"_{encoder_type.upper()}"

        _name_run += f"_{datetime.now().strftime('%d-%m_%H:%M:%S')}/"
    else:
        _name_run = name_run

    return f"{_name_exp}/{_name_run}"


def apply_xla_flags(config: dict) -> None:
    """Apply XLA flags for performance, debugging, and caching.

    Args:
        config: Training configuration.

    """
    xla_flags = ""

    if config["debug_flag"]:
        xla_flags += "--xla_gpu_autotune_level=0 "  # SEEDING
    if config["perf_flag"]:
        xla_flags += "--xla_gpu_enable_pipelined_reduce_scatter=true "
        xla_flags += "--xla_gpu_enable_triton_softmax_fusion=true "
        xla_flags += "--xla_gpu_triton_gemm_any=true "

    os.environ["XLA_FLAGS"] = xla_flags

    # Control JAX GPU memory fraction if provided (default 0.95 in base_config)
    gpu_mem_fraction = config.get("gpu_mem_fraction", None)
    if gpu_mem_fraction is not None:
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(gpu_mem_fraction)

    preallocate = config.get("xla_python_client_preallocate", None)
    if preallocate is not None:
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = str(preallocate).lower()

    if config["cache_flag"]:
        jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


def log_metrics(
    num_steps: int | None = None,
    metrics: dict | None = None,
    total_timesteps: int | None = None,
    writer: SummaryWriter = None,
) -> None:
    """Log and print training metrics and optionally send them to TensorBoard.

    Args:
        num_steps: Number of steps.
        metrics: Dictionary of metric names and values.
        total_timesteps: Total timesteps.
        writer: TensorBoard summary writer.

    """
    if total_timesteps is not None:
        logger.info(f"-> Step {num_steps}/{total_timesteps} - {(num_steps / total_timesteps) * 100:.2f}%")
        logger.info(f"-> Data time     : {metrics['runtime/data_time']:.2f}s")
        logger.info(f"-> Training time : {metrics['runtime/training_time']:.2f}s")
        logger.info(f"-> Log time      : {metrics['runtime/log_time']:.2f}s")
        logger.info(f"-> Eval time     : {metrics['runtime/eval_time']:.2f}s")

    for key, value in metrics.items():
        if writer:
            if total_timesteps is None:
                prefix = "evaluation/" if "/" not in key else ""
            else:
                prefix = "metrics/" if "/" not in key else ""
                if "ep_len_mean" in key or "ep_rew_mean" in key:
                    prefix = "rollout/"

            writer.add_scalar(f"{prefix}{key}", value, num_steps)
        logger.info(f"{key}: {value}")


def build_config_dicts(config: dict) -> tuple[dict, dict]:
    """Build separate configuration dictionaries for the environment and runtime.

    Args:
        config: The complete training configuration.

    Returns:
        A tuple containing the environment configuration and run configuration.

    """
    path_dataset = datasets.get_dataset(config["path_dataset"])
    path_dataset_eval = datasets.get_dataset(config["path_dataset_eval"])

    sdc_paths_from_data = not config["waymo_dataset"]

    env_config = {
        "path_dataset": path_dataset,
        "path_dataset_eval": path_dataset_eval,
        "sdc_paths_from_data": sdc_paths_from_data,
        "termination_keys": config["termination_keys"],
        "max_num_objects": config["max_num_objects"],
        "reward_type": config["reward_type"],
        "reward_config": config["reward_config"],
        "observation_type": config["observation_type"],
        "observation_config": config["observation_config"],
        "num_envs": config["num_envs"],
        "num_episode_per_epoch": config["num_episode_per_epoch"],
        "num_scenario_per_eval": config["num_scenario_per_eval"],
        "seed": config["seed"],
    }

    if config["network"]["encoder"]["type"] != "none":
        config["network"]["unflatten_config"] = config["observation_config"]

    training_config = config.get("training", {})
    legacy_mixed_precision = bool(training_config.get("mixed_precision", False))
    legacy_mp_dtype = str(training_config.get("mp_dtype", "bf16")).lower()
    default_compute_dtype = legacy_mp_dtype if legacy_mixed_precision else "fp32"
    compute_dtype = str(training_config.get("compute_dtype", default_compute_dtype)).lower()
    encoder_compute_dtype = str(training_config.get("encoder_compute_dtype", compute_dtype)).lower()
    cast_encoder_inputs = bool(
        training_config.get("cast_encoder_inputs", encoder_compute_dtype in ("bf16", "bfloat16")),
    )

    network_config = config["network"]
    network_config["dtype_policy"] = {
        # Legacy keys are kept for backward compatibility with existing scripts.
        "mixed_precision": legacy_mixed_precision,
        "mp_dtype": legacy_mp_dtype,
        # New keys support split precision between encoder and policy/value heads.
        "compute_dtype": compute_dtype,
        "encoder_compute_dtype": encoder_compute_dtype,
        "cast_encoder_inputs": cast_encoder_inputs,
    }
    del config["algorithm"]["network"]

    if network_config["value"]["layer_sizes"] is None:
        del network_config["value"]

    run_config = {
        "total_timesteps": config["total_timesteps"],
        "scenario_length": config["scenario_length"],
        "log_freq": config["log_freq"],
        "save_freq": config["save_freq"],
        "num_envs": config["num_envs"],
        "num_episode_per_epoch": config["num_episode_per_epoch"],
        "num_scenario_per_eval": config["num_scenario_per_eval"],
        "seed": config["seed"],
        "eval_freq": config["eval_freq"],
        **config["algorithm"],
        "network_config": network_config,
    }
    del run_config["name"]

    return env_config, run_config


def print_hyperparameters(args: dict) -> None:
    """Print hyperparameters in a structured and readable format.

    Args:
        args: Dictionary of hyperparameters.

    """
    print(" Experiment Summary ".center(40, "="))
    print(f"- Algorithm          : {args['algorithm']['name']}")
    print(f"- Observation Type   : {args['observation_type']}")
    print(f"- Encoder            : {args['network']['encoder']['type']}")
    mp_cfg = args.get("training", {})
    legacy_mp_enabled = bool(mp_cfg.get("mixed_precision", False))
    legacy_mp_dtype = str(mp_cfg.get("mp_dtype", "bf16")).lower()
    default_compute_dtype = legacy_mp_dtype if legacy_mp_enabled else "fp32"
    compute_dtype = str(mp_cfg.get("compute_dtype", default_compute_dtype)).lower()
    encoder_compute_dtype = str(mp_cfg.get("encoder_compute_dtype", compute_dtype)).lower()
    print(f"- Compute DType      : {compute_dtype}")
    print(f"- Encoder DType      : {encoder_compute_dtype}")
    print(f"- Dataset Path       : {args['path_dataset']}")
    print(f"- Total Timesteps    : {args['total_timesteps']}")


def get_and_print_device_info() -> int:
    """Display and return the count of local JAX devices."""
    print(" Devices ".center(40, "="))
    print(f"- Backend: {jax.default_backend()}")
    print(f"- Devices: {jax.local_device_count()} -> {jax.local_devices()}")


def save_params(path: str, params: Any) -> None:
    """Serialize and save model parameters to a specified file."""
    with epath.Path(path).open("wb") as fout:
        fout.write(pickle.dumps(params))


def setup_tensorboard(run_path: str) -> SummaryWriter:
    """Initialize and return a TensorBoard summary writer."""
    return SummaryWriter(log_dir=run_path)


def str2bool(v) -> bool:
    """Convert a string literal to a boolean."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentParser.ArgumentTypeError("Boolean value expected.")
