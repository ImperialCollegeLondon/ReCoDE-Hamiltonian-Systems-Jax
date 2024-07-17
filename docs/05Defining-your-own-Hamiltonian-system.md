# Defining your own Hamiltonian system

One of the goals of this ReCoDE is being able to implement a new Hamiltonian systems. In our framework, this can be done by creating a new
dynamical system class that inherits from the `Dynamics` class defined in `dynamics.py`. We will describe step-by-step how to define a new Hamiltonian system and will show how to implement the N-body system in the next chapter.


#### Step 1. Creating a new dynamical systems file

We start by creating a new file for our dynamical system `hdynamics/dynamics/[name].py`

#### Step 2. Inherit Dynamics class

We then create a new class by inheriting from `hdynamics.dynamics.Dynamics` class:

```
import jax.numpy as jnp

from hdynamics.dynamics import Dynamics

class MyHamiltonianSystem(Dynamics):
    """ My new class that implements a dynamical system. """
    def __init__(self, dim):
        cdim = dim
        super().__init__(cdim)

    def H(self, x):
        pass

    def plot_trajectory(self, trajectory, t_span, ax):
        """Plot trajectory within specified t_span range."""
        pass

    def plot_H(self, ax):
        """Plot the Hamiltonian."""
        pass

```

#### Step 3. Define the Hamiltonian of the system in `self.H(x)`

To implement the Hamiltonian, we simply define it as a function in `self.H(x)`.
This is the only necessary thing to define for our dynamical system to be valid.

#### Step 4 (optional). Define plot functions for trajectory and the Hamiltonian.

Optionally, we can also create code that plots trajectories `self.plot_trajectory(self, trajectory, t_span, ax)` and phase space `self.plot_H(self, ax)`. Here, `ax` is a Matplotlib <a href="https://matplotlib.org/stable/api/axes_api.html">Axes</a> object.

#### Done.

Now, we are done and we can perform simulations as described in pervious chapters simply by replacing the hamiltonian oscillator by our new class!