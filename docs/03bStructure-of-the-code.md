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