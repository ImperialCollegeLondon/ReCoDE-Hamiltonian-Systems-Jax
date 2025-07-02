"""Data generation utilities for Hamiltonian dynamics experiments.

This module provides functions to generate and load datasets of simulated trajectories
for various dynamical systems using JAX.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
from tqdm import tqdm

from hdynamics.odeint import ode_int
from hdynamics.utils import symplectic_form


def generate_dataset(dynamics, save_name=None, n_trajectories=1000, n_steps=50, stepsize=0.1, seed=100):
    """Generate or load a dataset of simulated trajectories for a given dynamical system.

    Args:
        dynamics: An object with attributes `H` (Hamiltonian function) and
            `initial_phase` (function for initial conditions).
        save_name (str, optional): Name to use for saving/loading the dataset file.
        n_trajectories (int, optional): Number of trajectories to generate.
        n_steps (int, optional): Number of time steps per trajectory.
        stepsize (float, optional): Step size for integration.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        trajectories (jax.numpy.ndarray): Simulated trajectories of shape (n_trajectories, n_steps+1, M).
        t_span (jax.numpy.ndarray): Time points for the simulation.
    """
    dataset_dir = "data"
    dataset_name = f"data_{save_name}_ntrajectories={n_trajectories}_steps={n_steps}_stepsize={stepsize}_seed={seed}"

    Path(dataset_dir).mkdir(parents=True, exist_ok=True)

    key = jax.random.PRNGKey(seed)

    try:
        trajectories = jnp.load(f"{dataset_dir}/{dataset_name}.npy")

        t_start = 0.0
        t_end = n_steps * stepsize
        t_span = jnp.linspace(t_start, t_end, n_steps + 1)

        print(f"Loaded dataset: {dataset_name}")

        return trajectories, t_span
    except Exception:
        print(f"Failed to load dataset: {dataset_name}")

    print(f"Generating dataset... {dataset_name}")
    trajectories = []

    hamiltonian = dynamics.H
    initial_fn = dynamics.initial_phase

    def grad_x(x, _):
        return symplectic_form(jax.grad(hamiltonian)(x))

    for i in tqdm(range(n_trajectories)):
        # Initial conditions
        key, subkey = jax.random.split(key)
        x_start = initial_fn(subkey)
        x_start = x_start.reshape(-1)  # M = 2 * n_objects * dim

        # Simulate
        trajectory, t_span = generate_trajectory(grad_x, x_start, stepsize, n_steps=n_steps)  # (T, M)

        # Append to trajectory list
        trajectories.append(trajectory)

    trajectories = jnp.stack(trajectories)

    jnp.save(f"{dataset_dir}/{dataset_name}.npy", trajectories)

    return trajectories, t_span


def generate_trajectory(grad_x, x_start, stepsize, n_steps=500):
    """Simulate a single trajectory using the provided gradient function and initial state.

    Args:
        grad_x (callable): Function that computes the time derivative of the state.
        x_start (array-like): Initial state vector.
        stepsize (float): Step size for integration.
        n_steps (int, optional): Number of time steps.

    Returns:
        solution (jax.numpy.ndarray): Simulated trajectory of shape (n_steps+1, M).
        t_span (jax.numpy.ndarray): Time points for the simulation.
    """
    t_start = 0.0
    t_end = n_steps * stepsize

    t_span = jnp.linspace(t_start, t_end, n_steps + 1)

    solution = ode_int(grad_x, x_start, t_span, atol=1e-10, rtol=1e-10)  # (T, M)

    return solution, t_span
