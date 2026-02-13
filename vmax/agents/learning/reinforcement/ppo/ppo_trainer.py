# Copyright 2025 Valeo.


"""Proximal Policy Optimization (PPO) trainer."""

from __future__ import annotations

import typing
from collections.abc import Callable
from functools import partial
import logging
import os
import random
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from vmax.agents import datatypes, pipeline
from vmax.agents.learning.reinforcement import ppo
from vmax.agents.pipeline import checkpointing, inference, pmap
from vmax.scripts.training import train_utils
from vmax.simulator import metrics as _metrics


if typing.TYPE_CHECKING:
    from waymax import datatypes as waymax_datatypes
    from waymax import env as waymax_env


LOG = logging.getLogger(__name__)


def _inverse_softplus(y: float) -> float:
    """Compute an inverse softplus value for positive y."""
    y = float(max(y, 1e-6))
    return float(np.log(np.expm1(y)))


def _adapt_policy_head_for_stochastic_policy(
    target_policy: datatypes.Params,
    source_policy: datatypes.Params,
    init_std: float = 0.05,
    min_std: float = 1e-3,
) -> tuple[datatypes.Params, bool]:
    """Adapt deterministic policy head weights to PPO gaussian head shape.

    If source output layer has size A and target has size 2*A, copy source
    weights into the mean half and initialize scale logits with a small std.
    """
    target_flat, _ = checkpointing._flatten_params(target_policy)
    source_flat, source_is_frozen = checkpointing._flatten_params(source_policy)

    changed = False
    head_keys = (("params", "Dense_0", "kernel"), ("params", "Dense_0", "bias"))
    log_std_logit = _inverse_softplus(max(float(init_std) - float(min_std), 1e-6))

    for key in head_keys:
        target_value = target_flat.get(key)
        source_value = source_flat.get(key)

        if target_value is None or source_value is None:
            continue
        if not (hasattr(target_value, "shape") and hasattr(source_value, "shape")):
            continue
        if target_value.shape == source_value.shape:
            continue

        # Kernel: (in_dim, 2 * action_dim) <- (in_dim, action_dim)
        if (
            key[-1] == "kernel"
            and len(target_value.shape) == 2
            and len(source_value.shape) == 2
            and target_value.shape[0] == source_value.shape[0]
            and target_value.shape[1] == 2 * source_value.shape[1]
        ):
            action_dim = source_value.shape[1]
            scale_kernel = jnp.zeros(
                (source_value.shape[0], action_dim),
                dtype=getattr(target_value, "dtype", source_value.dtype),
            )
            source_flat[key] = jnp.concatenate(
                [
                    source_value.astype(getattr(target_value, "dtype", source_value.dtype)),
                    scale_kernel,
                ],
                axis=-1,
            )
            changed = True
            continue

        # Bias: (2 * action_dim,) <- (action_dim,)
        if (
            key[-1] == "bias"
            and len(target_value.shape) == 1
            and len(source_value.shape) == 1
            and target_value.shape[0] == 2 * source_value.shape[0]
        ):
            action_dim = source_value.shape[0]
            scale_bias = jnp.full(
                (action_dim,),
                log_std_logit,
                dtype=getattr(target_value, "dtype", source_value.dtype),
            )
            source_flat[key] = jnp.concatenate(
                [
                    source_value.astype(getattr(target_value, "dtype", source_value.dtype)),
                    scale_bias,
                ],
                axis=-1,
            )
            changed = True

    if not changed:
        return source_policy, False

    adapted_source = checkpointing._unflatten_params(source_flat, source_is_frozen)
    return adapted_source, True


def train(
    env: waymax_env.PlanningAgentEnvironment,
    data_generator: typing.Iterator[waymax_datatypes.SimulatorState],
    eval_scenario: waymax_datatypes.SimulatorState,
    num_scenario_per_eval: int,
    total_timesteps: int,
    num_envs: int,
    num_episode_per_epoch: int,
    scenario_length: int,
    log_freq: int,
    seed: int,
    value_coef: float,
    entropy_coef: float,
    discount: float,
    gae_lambda: float,
    eps_clip: float,
    normalize_advantages: bool,
    save_freq: int,
    eval_freq: int,
    learning_rate: float,
    grad_updates_per_step: int,
    batch_size: int,
    unroll_length: int,
    num_minibatches: int,
    network_config: dict,
    progress_fn: Callable[[int, datatypes.Metrics], None] = lambda *args: None,
    checkpoint_logdir: str = "",
    resume: dict | None = None,
    disable_tqdm: bool = False,
) -> None:
    """Train a PPO agent.

    Args:
        env: An instance of the planning environment.
        data_generator: Iterator yielding simulator state samples.
        eval_scenario: Simulator state used for evaluation.
        num_scenario_per_eval: Number of evaluation scenarios.
        total_timesteps: Total training timesteps.
        num_envs: Number of parallel environments.
        num_episode_per_epoch: Episodes per epoch.
        scenario_length: Number of steps per scenario.
        log_freq: Frequency of logging.
        seed: Random seed.
        value_coef: Coefficient for value loss.
        entropy_coef: Coefficient for entropy loss.
        discount: Discount factor.
        gae_lambda: Lambda for Generalized Advantage Estimation.
        eps_clip: PPO clipping parameter.
        normalize_advantages: Flag for normalizing advantages.
        save_freq: Frequency to save model checkpoints.
        eval_freq: Evaluation frequency.
        learning_rate: Learning rate for optimizers.
        grad_updates_per_step: Gradient update iterations per step.
        batch_size: Batch size.
        unroll_length: Unroll length for trajectory generation.
        num_minibatches: Number of minibatches.
        network_config: Dictionary for network configurations.
        progress_fn: Callback function for reporting progress.
        checkpoint_logdir: Directory path for saving checkpoints.
        resume: Optional resume configuration dictionary.
        disable_tqdm: Flag to disable tqdm progress bar.

    """
    print(" PPO ".center(40, "="))

    rng = jax.random.PRNGKey(seed)
    num_devices = jax.local_device_count()

    do_save = save_freq > 1 and checkpoint_logdir is not None
    do_evaluation = eval_freq >= 1

    env_step_per_training_step = batch_size * unroll_length * num_minibatches
    total_iters = (total_timesteps // env_step_per_training_step) + 1

    observation_size = env.observation_spec()
    action_size = env.action_spec().data.shape[0]

    rng, network_key = jax.random.split(rng)

    print("-> Initializing networks...")
    dtype_policy = network_config.get("dtype_policy", {})
    legacy_mixed_precision = bool(dtype_policy.get("mixed_precision", False))
    legacy_mp_dtype = str(dtype_policy.get("mp_dtype", "bf16")).lower()
    default_compute_dtype = legacy_mp_dtype if legacy_mixed_precision else "fp32"
    compute_dtype = str(dtype_policy.get("compute_dtype", default_compute_dtype)).lower()
    encoder_compute_dtype = str(dtype_policy.get("encoder_compute_dtype", compute_dtype)).lower()
    cast_encoder_inputs = bool(dtype_policy.get("cast_encoder_inputs", encoder_compute_dtype in ("bf16", "bfloat16")))
    LOG.info(
        "DType policy: param=float32 compute=%s encoder_compute=%s output=float32 cast_encoder_inputs=%s",
        compute_dtype,
        encoder_compute_dtype,
        cast_encoder_inputs,
    )

    network, training_state, policy_fn = ppo.initialize(
        action_size,
        observation_size,
        env,
        learning_rate,
        network_config,
        num_devices,
        network_key,
    )
    learning_fn = ppo.make_sgd_step(
        ppo_network=network,
        num_minibatches=num_minibatches,
        gae_lambda=gae_lambda,
        discount=discount,
        eps_clip=eps_clip,
        value_coef=value_coef,
        entropy_coef=entropy_coef,
        normalize_advantages=normalize_advantages,
        encoder_compute_dtype=encoder_compute_dtype,
        cast_encoder_inputs=cast_encoder_inputs,
    )
    step_fn = partial(inference.policy_step, extra_fields=("truncation", "steps", "rewards"))
    print("-> Initializing networks... Done.")

    resume = resume or {}
    resume_enabled = bool(resume.get("enabled", False))
    resume_mode = resume.get("mode", "full")
    resume_strict = bool(resume.get("strict", True))
    resume_ckpt_path = resume.get("ckpt_path")
    resume_load_replay_buffer = bool(resume.get("load_replay_buffer", True))
    resume_policy_init_std = float(resume.get("policy_init_std", 0.05))

    if resume_enabled and not resume_ckpt_path:
        raise ValueError("resume.enabled is true but resume.ckpt_path is empty.")

    if resume_mode not in (
        "full",
        "weights_only",
        "weights_reset_opt",
        "weights_reset_opt_and_buffer",
    ):
        raise ValueError(f"Unknown resume.mode '{resume_mode}'.")

    iteration_offset = 0
    eval_count = 0
    loaded_parts: list[str] = []
    skipped_keys: list[str] = []

    if resume_enabled and resume_ckpt_path:
        if not os.path.isfile(resume_ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {resume_ckpt_path}")

        ckpt = checkpointing.load_checkpoint(resume_ckpt_path)
        model_state = ckpt.get("model_state_dict")
        optimizer_state = ckpt.get("optimizer_state_dict")
        trainer_state = ckpt.get("trainer_state") or {}
        rng_state = ckpt.get("rng_state") or {}

        training_state_host = pmap.unpmap(training_state)

        policy_source = getattr(model_state, "policy", None)
        if policy_source is None and isinstance(model_state, dict):
            policy_source = model_state.get("policy")

        value_source = getattr(model_state, "value", None)
        if value_source is None and isinstance(model_state, dict):
            value_source = model_state.get("value")

        policy_source_raw = policy_source

        if policy_source is None:
            if resume_strict:
                raise checkpointing.CheckpointError("Checkpoint is missing policy parameters.")
        else:
            policy_source, head_adapted = _adapt_policy_head_for_stochastic_policy(
                training_state_host.params.policy,
                policy_source,
                init_std=resume_policy_init_std,
            )
            if head_adapted:
                loaded_parts.append("policy_head_adapted")
            merged_policy, report = checkpointing.merge_params(
                training_state_host.params.policy,
                policy_source,
                strict=resume_strict,
            )
            training_state_host = training_state_host.replace(
                params=type(training_state_host.params)(
                    policy=merged_policy,
                    value=training_state_host.params.value,
                ),
            )
            loaded_parts.append("model")
            skipped_keys.extend(checkpointing.summarize_keys(report["missing_keys"]))
            skipped_keys.extend(checkpointing.summarize_keys(report["mismatched_keys"]))
            skipped_keys.extend(checkpointing.summarize_keys(report["extra_keys"]))

        if value_source is not None:
            merged_value, report = checkpointing.merge_params(
                training_state_host.params.value,
                value_source,
                strict=resume_strict,
            )
            training_state_host = training_state_host.replace(
                params=type(training_state_host.params)(
                    policy=training_state_host.params.policy,
                    value=merged_value,
                ),
            )
            skipped_keys.extend(checkpointing.summarize_keys(report["missing_keys"]))
            skipped_keys.extend(checkpointing.summarize_keys(report["mismatched_keys"]))
            skipped_keys.extend(checkpointing.summarize_keys(report["extra_keys"]))
        elif resume_mode != "full" and policy_source_raw is not None:
            # Warm-start value network trunk/encoder from policy weights when checkpoint has no value params (e.g., BC).
            merged_value, report = checkpointing.merge_params(
                training_state_host.params.value,
                policy_source_raw,
                strict=False,
            )
            if report["loaded_keys"]:
                training_state_host = training_state_host.replace(
                    params=type(training_state_host.params)(
                        policy=training_state_host.params.policy,
                        value=merged_value,
                    ),
                )
                loaded_parts.append("value_from_policy")
                skipped_keys.extend(checkpointing.summarize_keys(report["mismatched_keys"]))
                skipped_keys.extend(checkpointing.summarize_keys(report["extra_keys"]))
        elif resume_strict and resume_mode == "full":
            raise checkpointing.CheckpointError("Checkpoint is missing value parameters for full resume.")

        if resume_mode == "full":
            if optimizer_state is None and resume_strict:
                raise checkpointing.CheckpointError("Checkpoint is missing optimizer state.")
            if optimizer_state is not None:
                training_state_host = training_state_host.replace(optimizer_state=optimizer_state)
                loaded_parts.append("optimizer")

            if trainer_state:
                training_state_host = training_state_host.replace(
                    env_steps=int(trainer_state.get("env_steps", training_state_host.env_steps)),
                    rl_gradient_steps=int(trainer_state.get("rl_gradient_steps", training_state_host.rl_gradient_steps)),
                )
                iteration_offset = int(trainer_state.get("iteration", 0))
                eval_count = int(trainer_state.get("eval_count", 0))
                loaded_parts.append("trainer_state")
            elif resume_strict:
                raise checkpointing.CheckpointError("Checkpoint is missing trainer_state.")

            if rng_state and "jax" in rng_state:
                rng = rng_state["jax"]
                if "python" in rng_state:
                    random.setstate(rng_state["python"])
                if "numpy" in rng_state:
                    np.random.set_state(rng_state["numpy"])
                loaded_parts.append("rng")
            elif resume_strict:
                raise checkpointing.CheckpointError("Checkpoint is missing rng_state.")
        else:
            training_state_host = training_state_host.replace(
                optimizer_state=network.optimizer.init(training_state_host.params),
            )
            if resume_mode == "weights_only":
                training_state_host = training_state_host.replace(env_steps=0, rl_gradient_steps=0)
            else:
                training_state_host = training_state_host.replace(
                    env_steps=int(trainer_state.get("env_steps", 0)),
                    rl_gradient_steps=int(trainer_state.get("rl_gradient_steps", 0)),
                )
                iteration_offset = int(trainer_state.get("iteration", 0))
                eval_count = int(trainer_state.get("eval_count", 0))

        if resume_load_replay_buffer and resume_mode != "full":
            LOG.warning("resume.load_replay_buffer is only used for full mode; ignoring.")

        training_state = jax.device_put_replicated(training_state_host, jax.local_devices()[:num_devices])

        LOG.info(
            "Resuming from %s mode=%s loaded=(%s) skipped_keys=%s",
            resume_ckpt_path,
            resume_mode,
            "/".join(loaded_parts) if loaded_parts else "none",
            ",".join(skipped_keys) if skipped_keys else "none",
        )

    unroll_fn = partial(
        inference.generate_unroll,
        unroll_length=unroll_length,
        env=env,
        step_fn=step_fn,
    )

    run_training = partial(
        pipeline.run_training_on_policy,
        env=env,
        learning_fn=learning_fn,
        policy_fn=policy_fn,
        unroll_fn=unroll_fn,
        scan_length=batch_size * num_minibatches // num_envs,
        grad_updates_per_step=grad_updates_per_step,
    )
    run_evaluation = partial(
        pipeline.run_evaluation,
        env=env,
        policy_fn=policy_fn,
        step_fn=step_fn,
        scan_length=scenario_length * num_scenario_per_eval,
    )

    run_training = jax.pmap(run_training, axis_name="batch")
    run_evaluation = jax.pmap(run_evaluation, axis_name="batch")

    time_training = perf_counter()

    current_step = int(pmap.unpmap(training_state.env_steps))
    remaining_timesteps = max(total_timesteps - current_step, 0)
    if remaining_timesteps == 0:
        total_iters = 0
    else:
        total_iters = (remaining_timesteps // env_step_per_training_step) + 1

    print("-> Ground Control to Major Tom...")
    for iter_idx in tqdm(
        range(total_iters),
        desc="Training",
        total=total_iters,
        dynamic_ncols=True,
        disable=disable_tqdm,
    ):
        iteration = iteration_offset + iter_idx
        rng, iter_key = jax.random.split(rng)
        iter_keys = jax.random.split(iter_key, num_devices)

        # Batch data generation
        t = perf_counter()
        batch_scenarios = next(data_generator)
        epoch_data_time = perf_counter() - t

        # Training step
        t = perf_counter()
        should_log = not iteration % log_freq
        should_save = do_save and not iteration % save_freq
        should_eval = do_evaluation and not iteration % eval_freq
        should_sync = should_log or should_save or should_eval or iter_idx == total_iters - 1

        training_state, training_metrics = run_training(batch_scenarios, training_state, iter_keys)
        if should_sync:
            jax.block_until_ready(training_state.env_steps)
            epoch_training_time = perf_counter() - t
            current_step = int(pmap.unpmap(training_state.env_steps))
        else:
            epoch_training_time = 0.0
            current_step += env_step_per_training_step

        metrics = {}
        if should_log:
            t = perf_counter()
            training_metrics = pmap.flatten_tree(training_metrics)
            training_metrics = jax.device_get(training_metrics)
            training_metrics = _metrics.collect(training_metrics, "ep_len_mean")
            metrics = {
                "runtime/sps": int(env_step_per_training_step / max(epoch_training_time, 1e-8)),
                **{f"{name}": value for name, value in training_metrics.items()},
            }
            epoch_log_time = perf_counter() - t
        else:
            epoch_log_time = 0.0

        if should_save:
            t = perf_counter()
            path = f"{checkpoint_logdir}/model_{current_step}.pkl"
            train_utils.save_params(path, pmap.unpmap(training_state.params))
            checkpointing.save_checkpoint(
                f"{checkpoint_logdir}/checkpoint_{current_step}.pkl",
                {
                    "version": 1,
                    "model_state_dict": pmap.unpmap(training_state.params),
                    "optimizer_state_dict": pmap.unpmap(training_state.optimizer_state),
                    "trainer_state": {
                        "global_step": int(current_step),
                        "env_steps": int(current_step),
                        "rl_gradient_steps": int(pmap.unpmap(training_state.rl_gradient_steps)),
                        "iteration": int(iteration),
                        "eval_count": int(eval_count),
                    },
                    "rng_state": {
                        "python": random.getstate(),
                        "numpy": np.random.get_state(),
                        "jax": jax.device_get(rng),
                    },
                    "replay_buffer_state": None,
                },
            )
            epoch_log_time = perf_counter() - t if not should_log else epoch_log_time

        # Evaluation
        t = perf_counter()
        if should_eval:
            eval_metrics = run_evaluation(eval_scenario, training_state)
            eval_metrics = pmap.flatten_tree(eval_metrics)
            eval_metrics = jax.device_get(eval_metrics)
            eval_metrics = _metrics.collect(eval_metrics, "ep_len_mean")
            progress_fn(current_step, eval_metrics)
            eval_count += 1

        epoch_eval_time = perf_counter() - t

        if should_log:
            metrics["runtime/data_time"] = epoch_data_time
            metrics["runtime/training_time"] = epoch_training_time
            metrics["runtime/log_time"] = epoch_log_time
            metrics["runtime/eval_time"] = epoch_eval_time
            metrics["runtime/iter_time"] = epoch_data_time + epoch_training_time + epoch_log_time + epoch_eval_time
            metrics["runtime/wall_time"] = perf_counter() - time_training
            metrics["train/rl_gradient_steps"] = int(pmap.unpmap(training_state.rl_gradient_steps))
            metrics["train/env_steps"] = current_step

            progress_fn(current_step, metrics, total_timesteps)

            if disable_tqdm:
                print(f"-> Step {current_step}/{total_timesteps} - {(current_step / total_timesteps) * 100:.2f}%")

    print(f"-> Training took {perf_counter() - time_training:.2f}s")
    assert current_step >= total_timesteps

    if checkpoint_logdir:
        path = f"{checkpoint_logdir}/model_final.pkl"
        train_utils.save_params(path, pmap.unpmap(training_state.params))
        checkpointing.save_checkpoint(
            f"{checkpoint_logdir}/checkpoint_final.pkl",
            {
                "version": 1,
                "model_state_dict": pmap.unpmap(training_state.params),
                "optimizer_state_dict": pmap.unpmap(training_state.optimizer_state),
                "trainer_state": {
                    "global_step": int(current_step),
                    "env_steps": int(current_step),
                    "rl_gradient_steps": int(pmap.unpmap(training_state.rl_gradient_steps)),
                    "iteration": int(iteration_offset + max(total_iters - 1, 0)),
                    "eval_count": int(eval_count),
                },
                "rng_state": {
                    "python": random.getstate(),
                    "numpy": np.random.get_state(),
                    "jax": jax.device_get(rng),
                },
                "replay_buffer_state": None,
            },
        )

    pmap.assert_is_replicated(training_state)
    pmap.synchronize_hosts()
