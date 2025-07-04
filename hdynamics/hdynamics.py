"""Implementation of dynamics (e.g. specification of Hamiltonians)."""

from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp

from hdynamics.odeint import ode_int
from hdynamics.utils import symplectic_form


class Dynamics(ABC):
    """Abstract class for a dynamical system."""

    def __init__(self, cdim):
        """Initialisation of dynamical system.

        cdim is the dimensionality of q and p,
        pdim=2*cdim is the dimensionality of the phase space.
        """
        self.cdim = cdim
        self.pdim = cdim * 2

    @abstractmethod
    def H(self, x):
        """Return scalar energy given phase space location x of shape (pdim,)."""
        pass

    def initial_phase(self, key, q_scale=1.0, p_scale=1.0, q_trans=0.0, p_trans=0.0):
        """Return initial location of shape (2, cdim)."""
        x_start = jax.random.normal(key, shape=(2, self.cdim))

        qp_scale = jnp.array([q_scale, p_scale]).reshape(2, 1)
        qp_trans = jnp.array([q_trans, p_trans]).reshape(2, 1)
        x_start = x_start * qp_scale + qp_trans

        return x_start

    def jac_h(self):
        """Return the gradient of the Hamiltonian with respect to x."""
        return jax.grad(self.H)

    def get_rate_of_change(self):
        """Return a function grad_x(x, t) suitable for ODE integration."""
        jac_h = self.jac_h()

        def grad_x(x, t):
            return symplectic_form(jac_h(x))

        return grad_x

    def generate_trajectory(self, x_start, stepsize, n_steps=500):
        """Simulate a single trajectory using the provided gradient function and initial state.

        Args:
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

        grad_x = self.get_rate_of_change()

        solution = ode_int(grad_x, x_start, t_span, atol=1e-10, rtol=1e-10)

        return solution, t_span

    def plot_trajectory(self, trajectory, t_span, ax):
        """Plot trajectory within specified t_span range."""
        raise NotImplementedError("No trajectory plotting available for this dynamical system.")
