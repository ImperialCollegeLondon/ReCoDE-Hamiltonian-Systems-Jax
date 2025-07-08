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