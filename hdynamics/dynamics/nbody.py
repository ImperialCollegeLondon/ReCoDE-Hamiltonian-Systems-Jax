"""Implementation of N-body system."""

import jax.numpy as jnp

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
        self.masses = masses

    def H(self, x, eps=1.0):
        """Hamiltonian of Nbody system.

        Input:
            eps (float): controls the amount of smooth relaxation
        """
        assert len(x) == self.pdim, f"x does not have correct shape of {self.pdim}. Got x of shape {x.shape}."

        masses = (
            jnp.array([self.masses for _ in range(self.n_bodies)]) if isinstance(self.masses, float) else self.masses
        )

        q, p = x.reshape(2, self.n_bodies, self.dim)

        H_kinetic = jnp.sum((jnp.linalg.norm(p, axis=1) ** 2) / (2 * (masses**2)))

        q_dists = q.reshape(self.n_bodies, 1, self.dim) - q.reshape(
            1, self.n_bodies, self.dim
        )  # (n_bodies, n_bodies, dim)
        q_quads = jnp.sum(q_dists**2, axis=2)  # (n_bodies, n_bodies)
        masses_outer = jnp.outer(masses, masses)  # (n_bodies, n_bodies)
        H_potential = -jnp.sum(jnp.tril(masses_outer / jnp.sqrt(q_quads + (eps**2)), -1))

        H = H_kinetic + H_potential

        return H

    def plot_trajectory(
        self,
        trajectory,
        t_span,
        ax,
    ):
        """Plot 2d trajectory.

        Input:
            trajectory: (T, 2, n_bodies, dim), where T denotes time.
        """
        if self.dim != 2:
            raise NotImplementedError("N-body system plotting currently only supported for 2d systems")

        colors = [
            "tab:blue",
            "tab:red",
            "tab:orange",
            "tab:purple",
            "tab:green",
            "tab:brown",
            "tab:pink",
        ]

        for object_i in range(self.n_bodies):
            # color
            color = colors[object_i % len(colors)]

            points_x, points_y = trajectory[:, object_i * 2], trajectory[:, object_i * 2 + 1]

            # draw line
            ax.plot(points_x, points_y, "-", linewidth=1, color=color)

            ax.plot()

            # draw line
            ax.plot(
                points_x[-1:],
                points_y[-1:],
                ".",
                markersize=8,
                color=color,
            )

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
        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_aspect("equal")
