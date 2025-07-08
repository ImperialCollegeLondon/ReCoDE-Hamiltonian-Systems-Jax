"""Implementation of simple harmonic oscillator."""

from hdynamics.hdynamics import Dynamics


class HarmonicOscillator(Dynamics):
    """Dynamics of a one-dimensional simple harmonic oscillator.

    Models a system with Hamiltonian
        H(q, p) = p^2/(2m) + 0.5 * k * q^2,
    where q is position and p is momentum.

    Attributes:
        m (float): Mass of the oscillator.
        k (float): Spring constant.
    """

    def __init__(self, m=1.0, k=1.0):
        """Initialise the harmonic oscillator.

        Args:
            m (float): Mass of the oscillator.
            k (float): Spring constant.
        """
        super().__init__(cdim=1)
        self.m = m
        self.k = k

    def H(self, x, eps=1.0):
        """Compute the Hamiltonian (total energy) for a given state.

        Args:
            x (array-like): State vector [q, p], where q is position and p is momentum.
            eps (float, optional): Unused, included for interface compatibility.

        Returns:
            float: The total energy of the system at state x.

        Raises:
            ValueError: If x does not have shape (2,).
        """
        if x.shape[0] != 2:
            raise ValueError(f"x must have shape (2,), got shape {x.shape} and value {x}")

        q, p = x
        pot_energy = 0.5 * self.k * (q**2)
        kin_energy = (p**2) / (2 * self.m)
        return pot_energy + kin_energy

    def plot_trajectory(
        self,
        trajectory,
        t_span,
        ax,
    ):
        """Plot the position of the oscillator as a function of time.

        Args:
            trajectory (array): Array of shape (T, 2), where T is the number of time steps.
                trajectory[:, 0] contains positions.
            t_span (array): Array of time points of length T.
            ax (matplotlib.axes.Axes): Matplotlib Axes object to plot on.

        The plot shows the time evolution of the position coordinate.
        """
        # Draw line
        line = ax.plot(t_span, trajectory[:, 0])

        # Draw point at end of line
        ax.scatter([t_span[-1]], [trajectory[-1, 0]], s=20, marker="o", color=line[0].get_color())
