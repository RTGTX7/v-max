# Copyright 2025 Valeo.


"""Multi-layer perceptron network module."""

from collections.abc import Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn

from vmax.agents import datatypes


class MLP(nn.Module):
    """Multi-layer perceptron network composed of dense layers with optional dropout."""

    layer_sizes: Sequence[int] = (256, 256)
    activation: datatypes.ActivationFn = nn.relu
    dropout_rate: float | None = None
    kernel_init: datatypes.Initializer = nn.initializers.lecun_uniform()
    param_dtype: jnp.dtype = jnp.float32
    compute_dtype: jnp.dtype = jnp.float32
    output_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jax.Array, training: bool = False) -> jax.Array:
        """Run a forward pass through the MLP.

        Args:
            x: The input tensor.
            training: Boolean flag to enable dropout.

        Returns:
            The output tensor after processing through the MLP.

        """
        x = x.astype(self.compute_dtype)
        for i, size in enumerate(self.layer_sizes):
            x = nn.Dense(
                size,
                kernel_init=self.kernel_init,
                param_dtype=self.param_dtype,
                dtype=self.compute_dtype,
                name=f"hidden_{i}",
            )(x)

            if i != len(self.layer_sizes) - 1:
                x = self.activation(x)
            if self.dropout_rate is not None:
                x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)

        return x.astype(self.output_dtype)
