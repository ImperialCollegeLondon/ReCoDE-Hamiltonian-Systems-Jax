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

The Hamiltonian `H()` is the only method that is strictly required for any dynamical system. If we implement this method, we can simulate trajectories. We will see examples of this in the next sections.

## Use of JAX

JAX offers a general automatic differentiation system (autodiff), which we will use to obtain gradients in phase space given our hamiltonian definition. In JAX, we can define function and take gradients, Jacobians, Hessians, etc. to define new functions which can be evaluated. For the purposes of understanding our code, it should be enough to understand how to take gradients of a function,

| Math      | JAX |
| ----------- | ----------- |
| Function $H(x)$ | `H(x)`  |
| Gradient of a function $\nabla_x H $  | `jax.grad(H)`
| Evaluating gradient of a function $\nabla_x H(x) $  | `jax.grad(H)(x)`
| Passing evaluation through another function $J \nabla_x H(x) $  | `symplectic_form(jax.grad(H)(x))`

As can be seen `jax.grad` takes a function as input (here `H()`) and returns another function, namely the gradient of $H$. This function we can evaluate to find the gradient of $H$ evaluated at $x$. Lastly, we implement the symplectic form $J$ in `utils.py` which we can apply to the evaluated gradient. This will be useful for our purposes, because it gives us the <i>symplectic gradient</i>.  This is used in the `grad_x` function in `data.py` to calculate the gradient of the Hamiltonian, which is then used to integrate the equations of motion. 

By using JAX to take the gradient of the Hamiltonian, we can efficiently calculate the gradient of any dynamical system given only its Hamiltonian. This makes the implementation of new dynamical systems very easy, as we only need to implement the Hamiltonian and the rest of the code will work automatically and quickly, even for large systems.