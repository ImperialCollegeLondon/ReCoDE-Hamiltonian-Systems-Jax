from abc import abstractmethod # TODO: docstrings!
from functools import partial

import jax
import jax.numpy as jnp

import matplotlib
import matplotlib.pyplot as plt


class Dynamics(object): 
    def __init__(self, cdim):
        self.cdim = cdim
        self.pdim = cdim * 2

    @abstractmethod
    def H(self, x): 
        """ Return scalar energy given phase space location x of shape (pdim,) """
        pass

    def initial_phase(self, key, q_scale=1.0, p_scale=1.0, q_trans=0.0, p_trans=0.0):
        """ Return initial location of shape (2, cdim). """
        x_start = jax.random.normal(key, shape=(2, self.cdim))

        qp_scale = jnp.array([q_scale, p_scale]).reshape(2, 1)
        qp_trans = jnp.array([q_trans, p_trans]).reshape(2, 1)
        x_start = x_start * qp_scale + qp_trans
        
        return x_start

    def plot_trajectory(self, trajectory, t_span, ax):
        raise NotImplementedError(f"No trajectory plotting available for selected dynamical system.")

    def plot_H(self, ax):
        raise NotImplementedError(f"No energy plotting available for selected dynamical system.")


class Nbody(Dynamics):
    def __init__(self, dim, n_bodies, gravity=1.0, masses=1.0):
        cdim = dim * n_bodies
        super().__init__(cdim)

        self.dim = dim
        self.n_bodies = n_bodies

        self.gravity = gravity
        self.masses = masses

    def H(self, x, eps=1.0):
        assert len(x) == self.pdim, f"x does not have correct shape of {self.pdim}. Got x of shape {x.shape}."

        masses = [self.masses for _ in range(self.n_bodies)] if type(self.masses) == float else self.masses
        
        q, p = x.reshape(2, self.n_bodies, self.dim)
        
        H = 0.0
        for i in range(self.n_bodies):
            H += (jnp.linalg.norm(p[i]) ** 2) / (2 * masses[i])
        
            for j in range(i + 1, self.n_bodies):
                H -= (self.gravity * masses[i] * masses[j]) / jnp.sqrt(jnp.sum(jnp.power(q[i] - q[j], 2)) + (eps ** 2))
        
        return H

    def plot_trajectory(self, trajectory, t_span, ax, n_line_segments=500, xlim=3, ylim=3, H=None, JH=None, alpha=1.0, brighter=0.0):
        """ Plot 2d trajectory

            Input:
                trajectory: (T, 2, n_bodies, dim)
        """

        if self.dim != 2:
            raise NotImplementedError(f"N-body system plotting currently only supported for 2d systems")
        
        T = trajectory.shape[0]

        trajectory = trajectory.reshape(T, 2, self.n_bodies, self.dim)

        colors = ['tab:blue', 'tab:red', 'tab:orange', 'tab:purple', 'tab:green', 'tab:brown', 'tab:pink']
        colors = [tuple([x + (1 - x) * brighter for x in matplotlib.colors.to_rgb(color)]) for color in colors]
        
        for object_i in range(self.n_bodies):
            # color
            color = colors[object_i % len(colors)]

            points = trajectory.reshape(T, 2, self.n_bodies, self.dim)[:, 0, object_i]

            points_x, points_y = points[:, 0], points[:, 1]

            if brighter > 0.0:
                # draw dots
                ax.scatter(points_x, points_y, s=7, marker='.', color=color, alpha=alpha)
            else:
                # draw line
                ax.plot(points_x, points_y, '-', marker='.', markersize=5, color=color, alpha=alpha*0.4)

                # draw line
                ax.plot(points_x[-1:], points_y[-1:], '.', markersize=5, color=color, alpha=alpha)

        ax.set_xlim(-xlim, xlim)
        ax.set_ylim(-ylim, ylim)
        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_aspect('equal')



