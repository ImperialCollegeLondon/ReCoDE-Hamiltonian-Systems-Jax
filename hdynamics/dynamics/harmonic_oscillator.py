"""Implementation of simple harmonic oscillator."""

from hdynamics.hdynamics import Dynamics


class HarmonicOscillator(Dynamics):
    """Dynamics of a one-dimensional simple harmonic oscillator.

    Models a system with Hamiltonian H(q, p) = 0.5 * q^2 + 0.5 * omega^2 * p^2,
    where q is position and p is momentum.
    """

    def __init__(self, omega=1.0):
        """Initialise the harmonic oscillator.

        Args:
            omega (float): Angular frequency of the oscillator.
        """
        cdim = 1
        super().__init__(cdim)

        self.omega = omega

    def H(self, x, eps=1.0):
        """Compute the Hamiltonian (total energy) for a given state.

        Args:
            x (array-like): State vector [q, p], where q is position and p is momentum.
            eps (float, optional): Unused, included for interface compatibility.

        Returns:
            float: The total energy of the system at state x.
        """
        if x.shape[0] != 2:
            raise ValueError(f"x must have shape (2,), got shape {x.shape} and value {x}")

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
        """Plot the position of the oscillator as a function of time.

        Args:
            trajectory (array): Array of shape (T, 2), where T is the number of time steps.
                trajectory[:, 0] contains positions.
            t_span (array): Array of time points of length T.
            ax (matplotlib.axes.Axes): Matplotlib Axes object to plot on.

        The plot shows the time evolution of the position coordinate.
        """
        # Draw line
        ax.plot(t_span, trajectory[:, 0])

        # Draw point at end of line
        ax.scatter([t_span[-1]], [trajectory[-1, 0]], s=20, marker="o")
