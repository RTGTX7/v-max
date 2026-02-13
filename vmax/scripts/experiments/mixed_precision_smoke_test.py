#!/usr/bin/env python3

"""Smoke test for PPO mixed precision (FP32 vs BF16)."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from vmax.agents import datatypes
from vmax.agents.learning.reinforcement.ppo import ppo_factory


@dataclass
class CaseResult:
    mode: str
    losses: list[float]
    encoder_input_dtype: str
    param_float_dtypes: set[str]
    opt_float_dtypes: set[str]


def _build_fake_unflatten(dtype_probe: dict[str, str]):
    def _unflatten(obs: jax.Array):
        dtype_probe["encoder_input_dtype"] = str(obs.dtype)

        batch_size = obs.shape[0]

        sdc = obs[:, 0:4].reshape(batch_size, 1, 4)
        other = obs[:, 4:8].reshape(batch_size, 1, 4)
        rg = obs[:, 8:12].reshape(batch_size, 1, 4)
        tl = obs[:, 12:16].reshape(batch_size, 1, 4)
        gps = obs[:, 16:20].reshape(batch_size, 1, 4)

        sdc_mask = jnp.ones((batch_size, 1), dtype=bool)
        other_mask = jnp.ones((batch_size, 1), dtype=bool)
        rg_mask = jnp.ones((batch_size, 1), dtype=bool)
        tl_mask = jnp.ones((batch_size, 1), dtype=bool)

        return (sdc, other, rg, tl, gps), (sdc_mask, other_mask, rg_mask, tl_mask)

    return _unflatten


def _make_network_config(mixed_precision: bool) -> dict:
    return {
        "encoder": {
            "type": "mlp",
            "embedding_layer_sizes": [16, 16],
            "embedding_activation": "relu",
            "dk": 8,
            "concat_layer_sizes": [16],
            "concat_activation": "relu",
        },
        "policy": {
            "type": "mlp",
            "layer_sizes": [16, 16],
            "activation": "tanh",
            "final_activation": "none",
        },
        "value": {
            "type": "mlp",
            "layer_sizes": [16, 16],
            "activation": "tanh",
            "final_activation": "none",
            "num_networks": 1,
            "shared_encoder": True,
        },
        "action_distribution": "gaussian",
        "dtype_policy": {
            "mixed_precision": mixed_precision,
            "mp_dtype": "bf16",
        },
    }


def _make_fake_batch(
    key: jax.Array,
    batch_items: int,
    unroll_length: int,
    obs_size: int,
    action_size: int,
) -> datatypes.RLTransition:
    key_obs, key_next_obs, key_action, key_raw_action, key_log_prob, key_reward = jax.random.split(key, 6)

    obs = jax.random.normal(key_obs, (batch_items, unroll_length, obs_size), dtype=jnp.float32)
    next_obs = jax.random.normal(key_next_obs, (batch_items, unroll_length, obs_size), dtype=jnp.float32)
    action = jax.random.uniform(key_action, (batch_items, unroll_length, action_size), minval=-1.0, maxval=1.0)
    raw_action = jax.random.normal(key_raw_action, (batch_items, unroll_length, action_size), dtype=jnp.float32)
    log_prob = jax.random.normal(key_log_prob, (batch_items, unroll_length), dtype=jnp.float32) * 0.1
    reward = jax.random.normal(key_reward, (batch_items, unroll_length), dtype=jnp.float32) * 0.1
    flag = jnp.zeros((batch_items, unroll_length), dtype=jnp.float32)
    done = jnp.zeros((batch_items, unroll_length), dtype=jnp.float32)

    extras = {
        "state_extras": {
            "truncation": jnp.zeros((batch_items, unroll_length), dtype=jnp.float32),
        },
        "policy_extras": {
            "raw_action": raw_action,
            "log_prob": log_prob,
        },
    }

    return datatypes.RLTransition(
        observation=obs,
        action=action,
        reward=reward,
        flag=flag,
        next_observation=next_obs,
        done=done,
        extras=extras,
    )


def _floating_dtypes(tree) -> set[str]:
    leaves = jax.tree_util.tree_leaves(tree)
    return {
        str(x.dtype)
        for x in leaves
        if hasattr(x, "dtype") and jnp.issubdtype(jnp.asarray(x).dtype, jnp.floating)
    }


def run_case(steps: int, seed: int, mixed_precision: bool) -> CaseResult:
    obs_size = 20
    action_size = 2
    batch_items = 8
    unroll_length = 4
    num_minibatches = 2

    probe = {"encoder_input_dtype": "unknown"}
    unflatten_fn = _build_fake_unflatten(probe)
    network_config = _make_network_config(mixed_precision)

    network = ppo_factory.make_networks(
        observation_size=obs_size,
        action_size=action_size,
        unflatten_fn=unflatten_fn,
        learning_rate=3e-4,
        network_config=network_config,
    )

    key = jax.random.PRNGKey(seed)
    key_policy, key_value, key_rollout = jax.random.split(key, 3)

    params = ppo_factory.PPONetworkParams(
        policy=network.policy_network.init(key_policy),
        value=network.value_network.init(key_value),
    )
    optimizer_state = network.optimizer.init(params)

    state = ppo_factory.PPOTrainingState(
        params=params,
        optimizer_state=optimizer_state,
        env_steps=0,
        rl_gradient_steps=0,
    )

    # Trigger a forward pass so probe captures the encoder input dtype.
    dummy_obs = jnp.zeros((2, obs_size), dtype=jnp.float32)
    _ = network.policy_network.apply(params.policy, dummy_obs)

    sgd_step = ppo_factory.make_sgd_step(
        ppo_network=network,
        num_minibatches=num_minibatches,
        gae_lambda=0.95,
        discount=0.99,
        eps_clip=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        normalize_advantages=True,
        mixed_precision=mixed_precision,
        mp_dtype="bf16",
        pmap_axis_name=None,
    )

    losses: list[float] = []
    carry = (state, key_rollout)
    for step_idx in range(steps):
        key_rollout, batch_key = jax.random.split(carry[1])
        transitions = _make_fake_batch(
            key=batch_key,
            batch_items=batch_items,
            unroll_length=unroll_length,
            obs_size=obs_size,
            action_size=action_size,
        )
        carry, metrics = sgd_step((carry[0], key_rollout), step_idx, transitions)
        total_loss = metrics["total_loss"]
        loss_scalar = float(np.asarray(jnp.mean(total_loss)))
        if not np.isfinite(loss_scalar):
            raise RuntimeError(f"Non-finite loss encountered for mode={mixed_precision}: {loss_scalar}")
        losses.append(loss_scalar)

    final_state = carry[0]

    return CaseResult(
        mode="bf16" if mixed_precision else "fp32",
        losses=losses,
        encoder_input_dtype=probe["encoder_input_dtype"],
        param_float_dtypes=_floating_dtypes(final_state.params),
        opt_float_dtypes=_floating_dtypes(final_state.optimizer_state),
    )


def _validate(result: CaseResult) -> None:
    expected_encoder_dtype = "bfloat16" if result.mode == "bf16" else "float32"
    if expected_encoder_dtype not in result.encoder_input_dtype:
        raise RuntimeError(
            f"[{result.mode}] encoder input dtype mismatch: expected {expected_encoder_dtype}, got {result.encoder_input_dtype}"
        )

    if result.param_float_dtypes != {"float32"}:
        raise RuntimeError(f"[{result.mode}] parameters are not FP32-only: {sorted(result.param_float_dtypes)}")

    if result.opt_float_dtypes != {"float32"}:
        raise RuntimeError(f"[{result.mode}] optimizer states are not FP32-only: {sorted(result.opt_float_dtypes)}")

    if not np.all(np.isfinite(result.losses)):
        raise RuntimeError(f"[{result.mode}] non-finite losses found.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PPO mixed precision smoke tests.")
    parser.add_argument("--steps", type=int, default=10, help="Number of SGD steps per mode.")
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed.")
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["both", "fp32", "bf16"],
        help="Which mode to run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    modes = [False, True] if args.mode == "both" else [args.mode == "bf16"]
    results = [run_case(args.steps, args.seed, mixed_precision=m) for m in modes]

    for result in results:
        _validate(result)
        print(
            f"[{result.mode}] losses(min/mean/max)=({min(result.losses):.6f}/"
            f"{float(np.mean(result.losses)):.6f}/{max(result.losses):.6f}) "
            f"encoder_input_dtype={result.encoder_input_dtype} "
            f"param_float_dtypes={sorted(result.param_float_dtypes)} "
            f"opt_float_dtypes={sorted(result.opt_float_dtypes)}"
        )

    print("Smoke test passed.")


if __name__ == "__main__":
    main()
