"""Implementation of dynamics (e.g. specification of Hamiltonians)."""

from abc import ABC
from abc import abstractmethod

import jax
import jax.numpy as jnp
import matplotlib


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

    def plot_trajectory(self, trajectory, t_span, ax):
        """Plot trajectory within specified t_span range."""
        raise NotImplementedError("No trajectory plotting available for selected dynamical system.")

    def plot_H(self, ax):
        """Plot the Hamiltonian."""
        raise NotImplementedError("No energy plotting available for selected dynamical system.")


