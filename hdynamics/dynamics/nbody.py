"""Implementation of N-body system."""

import jax.numpy as jnp
import matplotlib.pyplot as plt

from hdynamics.hdynamics import Dynamics


class Nbody(Dynamics):
    """Dynamics of N-body system."""

    def __init__(self, dim, n_bodies, gravity=1.0, masses=1.0):
        """Initialise N-body dynamics."""
        cdim = dim * n_bodies
        super().__init__(cdim)

        self.dim = dim
        self.n_bodies = n_bodies

        self.gravity = gravity

        if isinstance(masses, (int, float)):
            self.masses = jnp.array([masses for _ in range(self.n_bodies)])
        else:
            self.masses = masses

        self.masses_outer = jnp.outer(self.masses, self.masses)  # (n_bodies, n_bodies)

    def H(self, x, eps=1.0):
        """Hamiltonian of Nbody system.

        Input:
            eps (float): controls the amount of smooth relaxation
        """
        assert len(x) == self.pdim, f"x does not have correct shape of {self.pdim}. Got x of shape {x.shape}."

        q, p = x.reshape(2, self.n_bodies, self.dim)

        H_kinetic = jnp.sum(jnp.sum(p**2, axis=1) / (2 * self.masses))

        q_dists = q.reshape(self.n_bodies, 1, self.dim) - q.reshape(
            1, self.n_bodies, self.dim
        )  # (n_bodies, n_bodies, dim)
        q_quads = jnp.sum(q_dists**2, axis=2)  # (n_bodies, n_bodies)

        H_potential = -jnp.sum(jnp.tril(self.masses_outer / jnp.sqrt(q_quads + (eps**2)), -1))

        H = H_kinetic + H_potential

        return H

    def plot_trajectory(
        self,
        trajectory,
        ax,
    ):
        """Plot 2d trajectory.

        Input:
            trajectory: (T, 2, n_bodies, dim), where T denotes time.
        """
        if self.dim != 2:
            raise NotImplementedError("N-body system plotting currently only supported for 2d systems")

        # Get the default colour cycle from matplotlib
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        for object_i in range(self.n_bodies):
            # color
            color = colors[object_i % len(colors)]

            points_x, points_y = trajectory[:, object_i * 2], trajectory[:, object_i * 2 + 1]

            # Draw line
            ax.plot(points_x, points_y, "-", linewidth=1, color=color, label=f"Object {object_i + 1}")

            # Draw points at end of line
            ax.scatter(points_x[-1], points_y[-1], s=20, marker="o", color=color)

        x_min = jnp.min(trajectory[:, : self.n_bodies * 2 : 2])
        x_max = jnp.max(trajectory[:, : self.n_bodies * 2 : 2])
        y_min = jnp.min(trajectory[:, 1 : self.n_bodies * 2 : 2])
        y_max = jnp.max(trajectory[:, 1 : self.n_bodies * 2 : 2])

        x_range = x_max - x_min
        y_range = y_max - y_min

        if x_range == 0:
            x_lim_min = x_min - 1
            x_lim_max = x_min + 1
        else:
            x_lim_min = x_min - x_range * 0.1
            x_lim_max = x_max + x_range * 0.1

        if y_range == 0:
            y_lim_min = y_min - 1
            y_lim_max = y_min + 1
        else:
            y_lim_min = y_min - y_range * 0.1
            y_lim_max = y_max + y_range * 0.1

        ax.set_xlim(x_lim_min, x_lim_max)
        ax.set_ylim(y_lim_min, y_lim_max)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        ax.legend()
