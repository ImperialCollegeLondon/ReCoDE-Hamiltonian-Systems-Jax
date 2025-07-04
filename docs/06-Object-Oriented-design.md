# Deep dive in the code

Before we start, we give an overview of the structure of the code.


## Object-oriented design

To achieve a code base that can easily be extended to also incorporate new dynamical systems, we define an abstract dynamical system in `hdynamics.py` as an abstract class named `Dynamics`. This acts as a blueprint for any dynamical system, without defining the specifics (such as the Hamiltonian) of any dynamical system in particular.

By defining the Hamiltonian method `H` as an abstract method using the ```@abstratcmethod``` decorator we force any class that inherits from `Dynamics` to implement this method before it can be instantiated. This is useful because a dynamic system must always define a Hamiltonian (as this defines the dynamics) so it can be solved, and this is how we enforce that. The child classes can optionally define the methods ```plot_trajectory``` and ```plot_H```. These are not abstract methods, but may be overridden in child classes.

| Method      | Description |
| ----------- | ----------- |
| `H()`      | <b>REQUIRED</b> This is the Hamiltonian of a dynamical system, and has to be specified for any dynamical system that is being implemented.   |
| `initial_phase()`   | <i>Already there.</i> This generates a random point in phase space and is already implemented.  |
| `plot_phase()`   | <i>OPTIONAL</i> This is a method to plot the phase space, and is optional.    |
| `plot_trajectory()`   | </i>OPTIONAL</i>  This is a method to plot the phase space, and is optional.      |

The Hamiltonian `H()` is the only method that is strictly required for any dynamical system. If we implement this method, we can simulate trajectories. We will see examples of this in the next 