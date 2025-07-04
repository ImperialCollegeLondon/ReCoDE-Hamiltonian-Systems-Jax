"""Implementation of dynamics (e.g. specification of Hamiltonians)."""

from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint


class Dynamics(ABC):
    """Abstract base class for a Hamiltonian dynamical system.

    Attributes:
        cdim (int): Dimensionality of q and p (configuration and momentum space).
        pdim (int): Dimensionality of the phase space (2 * cdim).
    """

    def __init__(self, cdim):
        """Initialise the dynamical system.

        Args:
            cdim (int): Dimensionality of q and p (configuration and momentum space).
        """
        self.cdim = cdim
        self.pdim = cdim * 2

    @abstractmethod
    def H(self, x):
        """Return the Hamiltonian (energy) at phase space location x.

        Args:
            x (array-like): Phase space vector of shape (pdim,).

        Returns:
            float: Scalar energy value at x.
        """
        pass

    def jac_h(self):
        """Return a function that computes the gradient of the Hamiltonian with respect to x.

        Returns:
            Callable: A function grad(x) that returns the gradient of H at x.
        """
        return jax.grad(self.H)

    @staticmethod
    def symplectic_form(x):
        """Return the canonical symplectic form for a phase space vector x = [q, p].

        Args:
            x (array-like): Phase space vector of shape (2*D,).

        Returns:
            jax.numpy.ndarray: Symplectic form vector of shape (2*D,).

        Raises:
            ValueError: If x is not a 1D array or does not have even length.
        """
        if len(x.shape) != 1:
            raise ValueError(f"symplectic form expects a vector of shape (M,). Got: {x.shape}.")
        if (len(x) % 2) != 0:
            raise ValueError(f"input shape should be even. Got {x.shape}.")
        D = x.shape[0] // 2
        q, p = x[:D], x[D:]
        return jnp.concatenate([p, -q])

    def get_rate_of_change(self):
        """Return a function grad_x(x, t) suitable for ODE integration.

        Returns:
            Callable: A function grad_x(x, t) that returns the time derivative of x.
        """
        jac_h = self.jac_h()

        def grad_x(x, t):
            return self.symplectic_form(jac_h(x))

        return grad_x

    def generate_trajectory(self, initial_conditions, stepsize, n_steps=500, rtol=1e-5, atol=1e-5):
        """Simulate a single trajectory using the system's ODE and initial state.

        Args:
            initial_conditions (array-like): Initial state vector of shape (M,), where M is the phase space dimension.
            stepsize (float): Step size for integration.
            n_steps (int, optional): Number of time steps (default: 500).
            rtol (float, optional): Relative tolerance for the ODE solver (default: 1e-5).
            atol (float, optional): Absolute tolerance for the ODE solver (default: 1e-5).

        Returns:
            solution (jax.numpy.ndarray): Simulated trajectory of shape (n_steps+1, M).
            t_span (jax.numpy.ndarray): Array of time points of shape (n_steps+1,).
        """
        t_start = 0.0
        t_end = n_steps * stepsize

        t_span = jnp.linspace(t_start, t_end, n_steps + 1)

        grad_x = self.get_rate_of_change()

        solution = odeint(grad_x, initial_conditions, t_span, rtol=rtol, atol=atol)

        return solution, t_span

    def plot_trajectory(self, trajectory, t_span, ax):
        """Plot trajectory within specified t_span range."""
        raise NotImplementedError("No trajectory plotting available for this dynamical system.")
