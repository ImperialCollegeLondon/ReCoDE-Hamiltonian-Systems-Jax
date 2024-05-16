"""Implementation of dynamics (e.g. specification of Hamiltonians)."""

from abc import abstractmethod

import jax
import jax.numpy as jnp
import matplotlib


class Dynamics(object):
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

        masses = [self.masses for _ in range(self.n_bodies)] if isinstance(self.masses, float) else self.masses

        q, p = x.reshape(2, self.n_bodies, self.dim)

        H = 0.0
        for i in range(self.n_bodies):
            H += (jnp.linalg.norm(p[i]) ** 2) / (2 * masses[i])

            for j in range(i + 1, self.n_bodies):
                H -= (self.gravity * masses[i] * masses[j]) / jnp.sqrt(jnp.sum(jnp.power(q[i] - q[j], 2)) + (eps**2))

        return H

    def plot_trajectory(
        self,
        trajectory,
        t_span,
        ax,
        n_line_segments=500,
        xlim=3,
        ylim=3,
        H=None,
        JH=None,
        alpha=1.0,
        transparent=False,
    ):
        """Plot 2d trajectory.

        Input:
            trajectory: (T, 2, n_bodies, dim)
        """
        if self.dim != 2:
            raise NotImplementedError("N-body system plotting currently only supported for 2d systems")

        T = trajectory.shape[0]

        trajectory = trajectory.reshape(T, 2, self.n_bodies, self.dim)

        colors = [
            "tab:blue",
            "tab:red",
            "tab:orange",
            "tab:purple",
            "tab:green",
            "tab:brown",
            "tab:pink",
        ]
        if transparent:
            colors = [tuple([x + (1 - x) * 0.6 for x in matplotlib.colors.to_rgb(color)]) for color in colors]

        for object_i in range(self.n_bodies):
            # color
            color = colors[object_i % len(colors)]

            points = trajectory.reshape(T, 2, self.n_bodies, self.dim)[:, 0, object_i]

            points_x, points_y = points[:, 0], points[:, 1]

            if transparent:
                # draw dots
                ax.plot(points_x, points_y, "-", linewidth=1, color=color, alpha=alpha)
            else:
                # draw line
                ax.plot(points_x, points_y, "-", linewidth=1, color=color, alpha=alpha)

                # draw line
                ax.plot(
                    points_x[-1:],
                    points_y[-1:],
                    ".",
                    markersize=8,
                    color=color,
                    alpha=alpha,
                )

        ax.set_xlim(-xlim, xlim)
        ax.set_ylim(-ylim, ylim)
        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_aspect("equal")


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
        n_line_segments=500,
        ylim=3,
        H=None,
        JH=None,
        alpha=1.0,
        transparent=False,
    ):
        """Plot 2d trajectory.

        Input:
            trajectory: (T, pdim)
        """
        T = trajectory.shape[0]

        trajectory = trajectory.reshape(T, 2, 1)

        # color
        color = "black"

        time = t_span

        points = trajectory.reshape(T, 2)[:, 0]

        if transparent:
            # draw brighter
            ax.plot(time, points, "-", s=10, linewidth=2, color=color, alpha=alpha)
        else:
            # draw line
            ax.plot(time, points, "-", linewidth=2, markersize=5, color=color, alpha=alpha)

            # draw ball
            ax.plot([time[-1]], [points[-1]], ".", markersize=6, color=color, alpha=alpha)

        ax.set_xlim(0, t_span[-1] * 1.1)
        ax.set_ylim(-ylim, ylim)

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
        n_line_segments=500,
        lim=3,
        H=None,
        JH=None,
        alpha=1.0,
        transparent=False,
    ):
        """Plot 2d trajectory.

        Input:
            trajectories: (B, T, pdim)
        """
        B, T, pdim = trajectories.shape

        if pdim != 2:
            raise NotImplementedError("Can onnly plot phase trajectories for 2-dimensional phase spaces")

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
