"""Implementation of N-body system."""

import jax.numpy as jnp

from hdynamics.hdynamics import Dynamics


class Nbody(Dynamics):
    """Dynamics of an N-body gravitational system.

    Simulates a system of N bodies interacting via Newtonian gravity in a given dimension.
    Positions and momenta are packed as (positions, momenta) in the phase space vector.
    """

    def __init__(self, dim, n_bodies, gravity=1.0, masses=1.0):
        """Initialise N-body dynamics.

        Args:
            dim (int): Number of spatial dimensions.
            n_bodies (int): Number of bodies in the system.
            gravity (float): Gravitational constant (default 1.0).
            masses (float or array): Mass of each body (scalar or array of length n_bodies).

        Raises:
            ValueError: If dim < 2 or n_bodies < 1.
        """
        if dim < 2:
            raise ValueError("dim must be at least 2 for N-body dynamics.")
        if n_bodies < 1:
            raise ValueError("n_bodies must be at least 1 for N-body dynamics.")

        phase_space_dim = dim * n_bodies
        super().__init__(phase_space_dim)

        self.dim = dim
        self.n_bodies = n_bodies
        self.gravity = gravity

        # Ensure masses is an array of shape (n_bodies,)
        if isinstance(masses, (int, float)):
            self.masses = jnp.full(self.n_bodies, masses)
        else:
            self.masses = masses

        # Precompute the outer product of masses for pairwise interactions
        self.mass_outer_product = jnp.outer(self.masses, self.masses)  # (n_bodies, n_bodies)

    def H(self, x, eps=0.1):
        """Hamiltonian (total energy) of the N-body system.

        Args:
            x (array): Phase space vector of shape (2 * n_bodies * dim,).
                First half: positions, second half: momenta.
            eps (float): Softening parameter to avoid singularities in the potential.

        Returns:
            float: The total energy (kinetic + potential) of the system.
        """
        if len(x) != self.pdim:
            raise ValueError(f"x must have length {self.pdim}. Got shape {x.shape} and value {x}")

        # Unpack phase space vector into positions and momenta
        positions, momenta = x.reshape(2, self.n_bodies, self.dim)

        # Kinetic energy: sum(p^2 / (2m)) for each body
        kinetic_energy = jnp.sum(jnp.sum(momenta**2, axis=1) / (2 * self.masses))

        # Compute pairwise displacement vectors between all bodies
        pairwise_displacements = positions.reshape(self.n_bodies, 1, self.dim) - positions.reshape(
            1, self.n_bodies, self.dim
        )  # (n_bodies, n_bodies, dim)
        # Squared distances between all pairs
        pairwise_squared_distances = jnp.sum(pairwise_displacements**2, axis=2)  # (n_bodies, n_bodies)

        # Potential energy: sum over unique pairs, softened by eps to avoid singularity
        # Use tril(..., -1) to sum only lower triangle (i < j), avoiding double-counting and self-interaction
        potential_energy = -self.gravity * jnp.sum(
            jnp.tril(
                self.mass_outer_product / jnp.sqrt(pairwise_squared_distances + (eps**2)),
                -1,
            )
        )

        hamiltonian = kinetic_energy + potential_energy
        return hamiltonian

    def plot_trajectory(
        self,
        trajectory,
        t_span,
        ax,
    ):
        """Plot the 2D trajectories of all bodies in the system.

        Args:
            trajectory (array): Array of shape (T, 2 * n_bodies), where T is the number of time steps.
                Each row contains the flattened x and y positions of all bodies at a given time.
            t_span (array): Array of time steps corresponding to the trajectory. Not used in this method,
                but included for interface consistency with related classes.
            ax (matplotlib.axes.Axes): Matplotlib Axes object to plot on.

        Raises:
            NotImplementedError: If the system dimension is not 2.
        """
        # Give an error message if the system is not 2D
        if self.dim != 2:
            raise NotImplementedError("N-body system plotting currently only supported for 2d systems")

        # Iterate over each body and plot its trajectory
        for object_i in range(self.n_bodies):
            # Extract x and y coordinates for the current object
            points_x = trajectory[:, object_i * 2]
            points_y = trajectory[:, object_i * 2 + 1]

            # Draw line
            line = ax.plot(points_x, points_y, linewidth=1, label=f"Body {object_i + 1}")

            # Draw points at end of line, using the color of the line
            ax.scatter(points_x[-1], points_y[-1], s=20, marker="o", color=line[0].get_color())

        # Set labels and legend
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend()
