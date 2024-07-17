# Deep dive in the code

Before we start, we give an overview of the structure of the code.


## Object-oriented design

To achieve a code base that can easily be extended to also incorporate new dynamical systems, we define an abstract dynamical system in `hdynamics.py` which acts as a general blue print for any dynamical system, without defining the specifics (such as the Hamiltonian) of any dynamical system in particular:


```
"""Implementation of dynamics (e.g. specification of Hamiltonians)."""

from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp


class Dynamics(ABC):
    """Abstract class for a dynamical system."""

    def __init__(self, cdim):
        """Initialisation of dynamical system.

        cdim is the dimensionality of q and p,
        pdim=2*cdim is the dimensionality of the phase space.
        """
        self.cdim = cdim
        self.pdim = cdim * 2

    @abstractmethod
    def H(self, x):
        """Return scalar energy given phase space location x of shape (pdim,)."""
        pass

    def initial_phase(self, key, q_scale=1.0, p_scale=1.0, q_trans=0.0, p_trans=0.0):
        """Return initial location of shape (2, cdim)."""
        x_start = jax.random.normal(key, shape=(2, self.cdim))

        qp_scale = jnp.array([q_scale, p_scale]).reshape(2, 1)
        qp_trans = jnp.array([q_trans, p_trans]).reshape(2, 1)
        x_start = x_start * qp_scale + qp_trans

        return x_start

    def plot_trajectory(self, trajectory, t_span, ax):
        """Plot trajectory within specified t_span range."""
        raise NotImplementedError("No trajectory plotting available for selected dynamical system.")

    def plot_H(self, ax):
        """Plot the Hamiltonian."""
        raise NotImplementedError("No energy plotting available for selected dynamical system.")
```


Importantly, we must always define a Hamiltonian (as this defines the dynamics) and can optionally define plotting functions. 

| Function      | Description |
| ----------- | ----------- |
| `H()`      | <b>REQUIRED</b> This is the Hamiltonian of a function, and has to be specified for any dynamical system that is being implemented.   |
| `initial_phase()`   | <i>Already there.</i> This generates a random point in phase space and is already implemented.  |
| `plot_phase()`   | <i>OPTIONAL</i> This is a function to plot the phase space, and is optional.    |
| `plot_trajectory()`   | </i>OPTIONAL</i>  This is a function to plot the phase space, and is optional.      |

The Hamiltonian `H()` is the only function that is strictly required for any dynamical system. If we implement this function, we can simulate trajectories. We will see examples of this in the next sections.

## Use of JAX

JAX offers a general automatic differentiation system (autodiff), which we will use to obtain gradients in phase space given our hamiltonian definition. In JAX, we can define function and take gradients, Jacobians, Hessians, etc. to define new functions which can be evaluted. For the purposes of understanding our code, it should be enough to understand how to take gradients of a function,

| Math      | JAX |
| ----------- | ----------- |
| Function $H(x)$ | `H(x)`  |
| Gradient of a function $\nabla_x H $  | `jax.grad(H)`
| Evaluating gradient of a function $\nabla_x H(x) $  | `jax.grad(H)(x)`
| Passing evaluation through another function $J \nabla_x H(x) $  | `symplectic_form(jax.grad(H)(x))`

As can be seen `jax.grad` takes a function as input (here `H()`) and returns another function, namely the gradient of $H$. This function we can evaluate to find the gradient of $H$ evaluated at $x$. Lastly, we implement the symplectic form $J$ in `utils.py` which we can apply to the evaluated gradient. This will be useful for our purposes, because it gives us the <i>symplectic gradient</i>.