# Structure of the code

## Project Structure

The structure of the code is as follows,

```log
.
├── hdynamics
│   ├── hdynamics.py
|   ├── dynamics
|          ├── nbody.py
|          ├── harmonic_oscillator.py
```

The `hdynamics.py` file contains the base class for dynamical systems, which defines the interface and some common functionality, including methods to solve the Hamiltonian equations of motion. The `dynamics` directory contains specific implementations of dynamical systems, such as the harmonic oscillator and N-body systems, which we will study as examples.

We will discuss the individual components in the sections to come.

## How these components work together?

By organising the code in this way, we can put many features that are common to all dynamical systems in the base class in `hdynamics.py`, allowing them to be reused in the specific implementations in the `dynamics` directory. This means we can easily add new dynamical systems by creating new files in the `dynamics` directory that inherit from the base class in `hdynamics.py`. Each specific dynamical system can then implement its own Hamiltonian and any other system-specific functionality while still benefiting from the common features provided by the base class.

## The `Dynamics` Class

The `Dynamics` class is defined in `hdynamics/hdynamics.py` and serves as the base class for all dynamical systems. It provides a common interface and some shared functionality for all dynamical systems, including methods to solve the Hamiltonian equations of motion. The key methods in the `Dynamics` class are:


* Constructor (`__init__`): Initializes the dynamics system with a specified number of dimensions. This is used to define the phase space of the system. Also creates a function `jac_h` which calculates the Jacobian of the Hamiltonian and stores it in the object. 
* `H`: This is the Hamiltonian of a dynamical system, and has to be specified for any dynamical system that is being implemented. This method is abstract, meaning it must be defined in any subclass that defines a dynamics system. 
* `symplectic_form`: Returns the symplectic form of the state of the system, which is used to compute the equations of motion.
*`rate_of_change`: Returns a function that computes the rate of change of the state of the system, given the Hamiltonian. This is in a format suitable for use in an ODE solver.
* `generate_trajectory`: Generates a trajectory of the system given an initial state, number of time steps and a length of timestep. This uses the function returned by the `rate_of_change` method to compute the evolution of the system over time.
* `plot_trajectory`: Plots the trajectory of the system. This may be overridden in subclasses to define a way of plotting the trajectory that makes sense for that particular system.