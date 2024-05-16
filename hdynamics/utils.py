"""Utility functions."""

import jax.numpy as jnp


def symplectic_form(x):
    """Returns symplectic form of a function.

    Input: x (M,)
    Returns: symplectic form (M,)
    """
    assert len(x.shape) == 1, f"symplectic form expects a Jacobian of shape (M,). Got: {x.shape}."
    assert (len(x) % 2) == 0, f"input shape should be even. Got {x.shape}."

    D = x.shape[0] // 2

    q, p = x[:D], x[D:]

    return jnp.concatenate([p, -q])  # Hamilton's equation of motion
