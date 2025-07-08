# Defining your own Hamiltonian system

One of the goals of this ReCoDE is being able to implement a new Hamiltonian systems. In our framework, this can be done by creating a new dynamical system class that inherits from the `Dynamics` class defined in `dynamics.py`. You're invited to follow along and implement your own Hamiltonian system as an exercise.


## 1. Creating a new dynamical systems file

We start by creating a new file for our dynamical system `hdynamics/dynamics/[name].py`

## 2. Inherit Dynamics class

We then create a new class by inheriting from `hdynamics.dynamics.Dynamics` class. The template below shows the class structure, but without the complete method implementations:

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
```

## 3. Define the Hamiltonian of the system in `self.H(x)`

To implement the Hamiltonian, we simply define it as a function in `self.H(x)`.
This is the only necessary thing to define for our dynamical system to be valid.

If you would like a suggestion of a simple Hamiltonian, you could use the example of a body free-falling in a gravitational field, which is defined as:

$
T = \frac{1}{2} m v^2\\
V = -gz\\
H = T + V
$

where $T$ is the kinetic energy, $V$ is the potential energy, $H$ is the Hamiltonian, $m$ is the mass of the body, $v$ is its vertical velocity, $g$ is the gravitational acceleration, and $z$ is the vertical position of the body.

## 4 (optional). Define plotting function

Optionally, we can also create code that plots trajectories `self.plot_trajectory(self, trajectory, t_span, ax)` and phase space `self.plot_H(self, ax)`. Here, `ax` is a Matplotlib <a href="https://matplotlib.org/stable/api/axes_api.html">Axes</a> object.

If you chose to implement the example of a body free-falling in the gravitational field, you could plot the vertical position of the body over time, or the speed of the bdy over time (remember speed can be defined in terms of momentum as $v = \frac{p}{m}$).

## 5. Run the Code
To run the code, you could import the class into a Jupyter notebook or Python script and create an instance of the class. Then, you can call the methods to perform the simulation and plot the result. For example:

```python
from hdynamics.dynamics import MyHamiltonianSystem

# Create an instance of the Hamiltonian system
system = MyHamiltonianSystem()

# Define initial conditions and time span
initial_conditions = jnp.array([1.0, 0.0])  # Example initial conditions
time_span = jnp.linspace(0, 10, 100)  # Example time span

# Simulate the system
trajectory, t_span = system.simulate(initial_conditions, time_span)

# Plot the trajectory
import matplotlib.pyplot as plt
fig, ax = plt.subplots()

system.plot_trajectory(trajectory, t_span, ax)
plt.show()
```