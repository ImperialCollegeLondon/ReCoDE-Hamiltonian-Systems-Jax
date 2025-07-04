"""Implementation of simple harmonic oscillator."""

import jax.numpy as jnp

from hdynamics.hdynamics import Dynamics


class HarmonicOscillator(Dynamics):
    """Dynamics of simple harmonic oscillator."""

    def __init__(self, omega=1.0):
        """Initialise harmonic oscillator."""
        cdim = 1
        super().__init__(cdim)

        self.omega = omega

    def H(self, x, eps=1.0):
        """Hamiltonian of harmonic oscillator."""
        assert x.shape[0] == 2, f"x does have have correct shape: {x}"

        q, p = x

        pot_energy = 0.5 * (q**2)
        kin_energy = 0.5 * (p**2) * (self.omega**2)

        return pot_energy + kin_energy

    def plot_trajectory(
        self,
        trajectory,
        t_span,
        ax,
    ):
        """Plot 2d trajectory.

        Input:
            trajectory: (T, pdim)
        """
        # Draw line
        ax.plot(t_span, trajectory[:, 0])

        # Draw point at end of line
        ax.scatter([t_span[-1]], [trajectory[-1, 0]], s=20, marker="o")

    def plot_phase_energy(
        self,
        ax,
        q_min=-1.0,
        q_max=1.0,
        p_min=-1.0,
        p_max=1.0,
        num_points=100,
        **contour_kwargs,
    ):
        """Plot energy contours in phase space (q, p).

        Args:
            ax (matplotlib.axes.Axes): Matplotlib Axes object to plot on.
            q_min (float): Minimum value for position axis.
            q_max (float): Maximum value for position axis.
            p_min (float): Minimum value for momentum axis.
            p_max (float): Maximum value for momentum axis.
            num_points (int): Number of grid points per axis.
            **contour_kwargs: Additional keyword arguments for ax.contour.

        The Hamiltonian is evaluated at each (q, p) grid point.
        """
        q = jnp.linspace(q_min, q_max, num_points)
        p = jnp.linspace(p_min, p_max, num_points)
        Q, P = jnp.meshgrid(q, p)
        # Evaluate H at each (q, p) pair
        H_grid = jnp.vectorize(lambda q, p: self.H(jnp.array([q, p])))(Q, P)

        contour = ax.contour(Q, P, H_grid, **contour_kwargs)
        ax.set_xlabel("q")
        ax.set_ylabel("p")
        return contour
