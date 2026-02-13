# Copyright 2025 Valeo.


"""Module containing utility functions for constructing MLP embeddings."""

import jax.numpy as jnp
from flax import linen as nn


def build_mlp_embedding(
    input_features,
    output_size,
    hidden_sizes,
    activation_fn,
    name_prefix,
    *,
    param_dtype=jnp.float32,
    compute_dtype=jnp.float32,
    output_dtype=jnp.float32,
):
    """Build an MLP embedding network.

    Args:
        input_features: The input tensor.
        output_size: The final output size.
        hidden_sizes: A sequence of hidden layer sizes.
        activation_fn: Activation function to use between layers.
        name_prefix: Prefix for naming layers.

    Returns:
        The output tensor after applying the MLP.

    """
    x = input_features.astype(compute_dtype)
    for i, hidden_size in enumerate(hidden_sizes):
        x = nn.Dense(
            hidden_size,
            param_dtype=param_dtype,
            dtype=compute_dtype,
            name=f"{name_prefix}_layer_{i}",
        )(x)
        if activation_fn:
            x = activation_fn(x)

    output = nn.Dense(
        output_size,
        param_dtype=param_dtype,
        dtype=compute_dtype,
        name=f"{name_prefix}_output",
    )(x)

    return output.astype(output_dtype)
