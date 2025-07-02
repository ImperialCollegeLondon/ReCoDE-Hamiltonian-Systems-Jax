# Structure of the code

## Project Structure

The structure of the code is as follows,

```log
.
├── hdynamics
│   ├── hdynamics.py
│   └── odeint.py
│   └── utils.py
|   ├── dynamics
|          ├── nbody.py
|          ├── harmonic_oscillator.py
```

with a brief description of different components summarized in the table below,

| Files    | Description |
| -------- | ------- |
| hdynamics.py  | Main code containing abstract class of dynamical system. |
| odeint.py | Code for integrating ordinary differential equations. |
| utils.py    | Utility functions. |
| data.py | Functions to generate trajectory data. |
| dynamics/  | Folder containing implemented dynamical systems. |

We will discuss the individual components in the sections to come.

## How these components work together?

The main purpose of the code base is to be able to implement different dynamical systems through their corresponding Hamiltonians, with a focus on being able to simulate and visualise the phase spaces and/or trajectories of these systems. Further, we intend to do this in an extendable manner, such that new dynamical systems can easily be added to the code base.