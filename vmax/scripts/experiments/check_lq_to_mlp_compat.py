#!/usr/bin/env python3
# Copyright 2025 Valeo.

"""Check LQ -> MLP PPO parameter compatibility without training."""

from __future__ import annotations

import argparse
from pathlib import Path
import json

import jax
from omegaconf import OmegaConf
import jax.numpy as jnp

from vmax import PATH_TO_APP
from vmax.agents.learning.reinforcement import ppo
from vmax.agents.pipeline import checkpointing, pmap
from vmax.scripts.training import train_utils
from vmax import simulator
from waymax import dynamics
from vmax.agents.networks import encoders, network_utils


def _load_config() -> dict:
    cfg = OmegaConf.load(Path(PATH_TO_APP) / "config" / "base_config.yaml")
    return OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)


def _build_env(config: dict):
    env_config, _ = train_utils.build_config_dicts(config)
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
    return env


def _init_model(config: dict):
    env = _build_env(config)
    observation_size = env.observation_spec()
    action_size = env.action_spec().data.shape[0]
    key = jax.random.PRNGKey(0)
    num_devices = jax.local_device_count()
    network, training_state, _ = ppo.initialize(
        action_size,
        observation_size,
        env,
        config["algorithm"]["learning_rate"],
        config["network"],
        num_devices,
        key,
    )
    params = pmap.unpmap(training_state.params)
    return params


def _encoder_output_dim(config: dict) -> int:
    env = _build_env(config)
    observation_size = env.observation_spec()
    dummy_obs = jnp.zeros((1, observation_size))
    enc_cfg = config["network"]["encoder"]
    enc_type = enc_cfg["type"]
    enc_cfg = network_utils.parse_config(enc_cfg, ["type"])
    encoder = encoders.get_encoder(enc_type)(env.get_wrapper_attr("features_extractor").unflatten_features, **enc_cfg)
    params = encoder.init(jax.random.PRNGKey(0), dummy_obs)
    out = encoder.apply(params, dummy_obs)
    return int(out.shape[-1])


def _flatten(params):
    flat, _ = checkpointing._flatten_params(params)  # internal helper
    return flat


def _diff_params(a, b):
    a_flat = _flatten(a)
    b_flat = _flatten(b)
    a_keys = set(a_flat.keys())
    b_keys = set(b_flat.keys())
    missing = sorted(a_keys - b_keys)
    extra = sorted(b_keys - a_keys)
    mismatched = []
    for key in sorted(a_keys & b_keys):
        av = a_flat[key]
        bv = b_flat[key]
        if hasattr(av, "shape") and hasattr(bv, "shape"):
            if av.shape != bv.shape:
                mismatched.append((key, av.shape, bv.shape))
    return missing, extra, mismatched


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", required=False)
    args = parser.parse_args()

    base = _load_config()

    mlp_cfg = json.loads(json.dumps(base))
    mlp_cfg["network"]["encoder"] = OmegaConf.to_container(
        OmegaConf.load(Path(PATH_TO_APP) / "config" / "network" / "encoder" / "mlp.yaml"),
        resolve=True,
    )

    lq_cfg = json.loads(json.dumps(base))
    lq_cfg["network"]["encoder"] = OmegaConf.to_container(
        OmegaConf.load(Path(PATH_TO_APP) / "config" / "network" / "encoder" / "lq.yaml"),
        resolve=True,
    )

    print("Encoder output dim (MLP):", _encoder_output_dim(mlp_cfg))
    print("Encoder output dim (LQ):", _encoder_output_dim(lq_cfg))

    print("Initializing MLP params...")
    mlp_params = _init_model(mlp_cfg)
    print("Initializing LQ params...")
    lq_params = _init_model(lq_cfg)

    missing, extra, mismatched = _diff_params(lq_params, mlp_params)
    print(f"Keys missing in MLP (present in LQ): {len(missing)}")
    print(f"Extra keys in MLP: {len(extra)}")
    print(f"Shape mismatches: {len(mismatched)}")
    if mismatched:
        print("First 20 mismatches:")
        for key, ashape, bshape in mismatched[:20]:
            print("/".join(key), ashape, "->", bshape)

    if args.ckpt_path:
        ckpt = checkpointing.load_checkpoint(args.ckpt_path)
        model_state = ckpt.get("model_state_dict")
        print("Attempting weights_only merge (strict=False)...")
        merged_policy, report = checkpointing.merge_params(
            mlp_params.policy,
            model_state.policy if hasattr(model_state, "policy") else model_state.get("policy"),
            strict=False,
        )
        print(f"Loaded keys: {len(report['loaded_keys'])}")
        print(f"Missing keys: {len(report['missing_keys'])}")
        print(f"Mismatched keys: {len(report['mismatched_keys'])}")
        print(f"Extra keys: {len(report['extra_keys'])}")


if __name__ == "__main__":
    main()
