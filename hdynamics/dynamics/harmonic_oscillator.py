"""Implementation of simple harmonic oscillator."""

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
