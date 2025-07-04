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
        T = trajectory.shape[0]

        trajectory = trajectory.reshape(T, 2, 1)

        time = t_span

        points = trajectory.reshape(T, 2)[:, 0]

        # Draw line
        ax.plot(time, points, "-", linewidth=2, markersize=5)

        # Draw point at end of line
        ax.scatter([time[-1]], [points[-1]], s=20, marker="o")

        y_min = jnp.min(points)
        y_max = jnp.max(points)

        y_range = y_max - y_min

        if y_range == 0:
            y_lim_min = y_min - 1.0
            y_lim_max = y_min + 1.0
        else:
            y_lim_min = y_min - 0.1 * y_range
            y_lim_max = y_max + 0.1 * y_range

        ax.set_xlim(0, t_span[-1] * 1.1)
        ax.set_ylim(y_lim_min, y_lim_max)

    def plot_phase_energy(self, grid_energy, ax, lim=1.0):
        """Plot energy on phase space.

        Input:
            energy grid: (H, W)
        """
        H, W = grid_energy.shape
        y = jnp.linspace(-lim, lim, H)
        x = jnp.linspace(-lim, lim, W)
        xx, yy = jnp.meshgrid(x, y)
        ax.contour(xx, yy, grid_energy)

        ax.set_xticks([-lim, 0, lim])
        ax.set_yticks([-lim, 0, lim])

        ax.set_aspect("equal")

    def plot_phase_trajectories(
        self,
        trajectories,
        ax,
        lim=3,
        alpha=1.0,
        transparent=False,
    ):
        """Plot 2d trajectory.

        Input:
            trajectories: (B, T, pdim)
        """
        B, T, pdim = trajectories.shape

        if pdim != 2:
            raise NotImplementedError("Can only plot phase trajectories for 2-dimensional phase spaces")

        for trajectory in trajectories:
            # color
            color = "black"

            q_points = trajectory.reshape(T, 2)[:, 0]
            p_points = trajectory.reshape(T, 2)[:, 1]

            if transparent:
                # draw dots
                for t_i in range(T):
                    ax.scatter(
                        [q_points[t_i]],
                        [p_points[t_i]],
                        s=3,
                        markersize=10,
                        marker=".",
                        color=color,
                        alpha=alpha,
                    )
            else:
                # draw line
                ax.plot(
                    q_points,
                    p_points,
                    ".-",
                    markersize=10,
                    color=color,
                    alpha=0.5 * alpha,
                )

                # draw ball
                ax.plot(
                    [q_points[-1]],
                    [p_points[-1]],
                    ">",
                    markersize=5,
                    color=color,
                    alpha=alpha,
                )

        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)

        ax.set_xlabel("q")
        ax.set_ylabel("p")

        ax.set_aspect("equal")
