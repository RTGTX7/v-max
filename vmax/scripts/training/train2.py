# Copyright 2025 Valeo.

"""Script to run (and resume) the training process."""

import os
import sys
import pickle
import logging
import inspect
from functools import partial
from typing import Optional, Dict, Any

import hydra
from omegaconf import DictConfig, OmegaConf
from waymax import dynamics

from vmax import PATH_TO_APP, simulator
from vmax.agents import learning
from vmax.scripts.training import train_utils

OmegaConf.register_new_resolver("output_dir", train_utils.resolve_output_dir)

LOG = logging.getLogger("vmax.train2")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def _find_resume_path(config: Dict[str, Any]) -> Optional[str]:
    """Figure out where to load model_final.pkl from."""
    # 1) Highest priority: explicit resume_path in config (hydra override: resume_path=...)
    explicit = config.get("resume_path")
    if explicit and os.path.isfile(explicit):
        return explicit

    # 2) Common places users keep checkpoints
    candidates = [
        "model_final.pkl",
        os.path.join(os.getcwd(), "model_final.pkl"),
        os.path.join("model", "model_final.pkl"),
        os.path.join("checkpoints", "model_final.pkl"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p

    return None


def _choose_resume_kwarg(train_fn) -> Optional[str]:
    """Probe train_fn signature to select the correct kwarg to pass a checkpoint path."""
    sig = inspect.signature(train_fn)
    params = set(sig.parameters.keys())
    # Try common names in order of likelihood
    for name in ("resume_from", "checkpoint_path", "load_path", "resume_path", "init_checkpoint"):
        if name in params:
            return name
    return None


@hydra.main(version_base=None, config_name="base_config", config_path=PATH_TO_APP + "/config")
def run(cfg: DictConfig) -> None:
    """Run the training process with optional resume from model_final.pkl.

    Args:
        cfg: Configuration for the training process (Hydra/OmegaConf).
    """
    # Resolve config to plain dict
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    # Environment / device setup
    train_utils.apply_xla_flags(config)
    train_utils.print_hyperparameters(config)
    train_utils.get_and_print_device_info()

    # Build env/run configs
    env_config, run_config = train_utils.build_config_dicts(config)

    # Data generators
    data_generator = simulator.make_data_generator(
        path=env_config["path_dataset"],
        max_num_objects=env_config["max_num_objects"],
        include_sdc_paths=env_config["sdc_paths_from_data"],
        batch_dims=(env_config["num_envs"], env_config["num_episode_per_epoch"]),
        seed=env_config["seed"],
        distributed=True,
    )

    if config.get("eval_freq", 0) > 0 and env_config.get("path_dataset_eval"):
        eval_data_generator = simulator.make_data_generator(
            path=env_config["path_dataset_eval"],
            max_num_objects=env_config["max_num_objects"],
            include_sdc_paths=env_config["sdc_paths_from_data"],
            batch_dims=(8, int(config["num_scenario_per_eval"]) // 8),
            seed=69,
            distributed=True,
        )
        eval_scenario = next(eval_data_generator)
        del eval_data_generator
    else:
        eval_scenario = None

    # Environment
    env = simulator.make_env_for_training(
        max_num_objects=env_config["max_num_objects"],
        dynamics_model=dynamics.InvertibleBicycleModel(normalize_actions=True),
        sdc_paths_from_data=env_config["sdc_paths_from_data"],
        observation_type=env_config["observation_type"],
        observation_config=env_config["observation_config"],
        reward_type=env_config["reward_type"],
        reward_config=env_config["reward_config"],
        termination_keys=env_config["termination_keys"],
    )

    # Output dirs
    absolute_run_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    relative_run_path = "/".join(absolute_run_path.split("/")[-2:])
    model_path = os.path.join(relative_run_path, "model")
    os.makedirs(model_path, exist_ok=True)

    writer = train_utils.setup_tensorboard(relative_run_path)
    progress = partial(train_utils.log_metrics, writer=writer)

    # Get train function and detect resume interface
    train_fn = learning.get_train_fn(config["algorithm"]["name"])
    resume_kwarg = _choose_resume_kwarg(train_fn)

    # Locate model_final.pkl
    resume_path = _find_resume_path(config)

    resume_kwargs: Dict[str, Any] = {}
    if resume_path and resume_kwarg:
        LOG.info(f"Resuming training using {resume_kwarg}='{resume_path}'")
        resume_kwargs[resume_kwarg] = resume_path
    elif resume_path and not resume_kwarg:
        # Fall back: try to load params to warm-start if the API doesn't support a path.
        # We don't know the internal structure, so we only log a hint and proceed cleanly.
        LOG.warning(
            "Found 'model_final.pkl' at '%s' but the trainer doesn't accept a resume path "
            "argument. Continuing without resume. If your trainer supports warm-starting "
            "via parameters, expose a compatible kwarg (e.g., 'initial_params').", resume_path
        )
        # If you know your pickle structure, you could deserialize here:
        # with open(resume_path, 'rb') as f:
        #     ckpt = pickle.load(f)
        # and pass pieces if train_fn supports them.
    else:
        LOG.info("No 'model_final.pkl' found and/or no compatible resume argument; starting fresh.")

    # TRAIN
    train_fn(
        env=env,
        data_generator=data_generator,
        eval_scenario=eval_scenario,
        **run_config,
        progress_fn=progress,
        checkpoint_logdir=model_path,
        disable_tqdm=not sys.stdout.isatty(),
        **resume_kwargs,
    )


if __name__ == "__main__":
    run()
