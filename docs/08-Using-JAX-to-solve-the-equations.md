# Use of JAX

JAX is a powerful library for numerical computing in Python, which allows us to efficiently and conveniently compute gradients and perform automatic differentiation. In the context of Hamiltonian dynamics, JAX can be used to compute the gradients of the Hamiltonian function, which is essential for integrating the equations of motion.

## The Implementation of JAX Functions

Unlike standard Python functions, JAX functions, especially those created using transformations like `jax.grad` (used in the `jac_h` attribute) and `jax.jit`, are executed in a special way. When you use these transformations, JAX traces your code to build a computation graph, which it can then optimize and compile for fast execution on CPUs, GPUs, or TPUs. The code called by these functions (such as within the method `H`) is not always run line-by-line in the normal Python interpreter. Instead, JAX focuses on pure, functional-style code that operates on arrays. This approach enables automatic differentiation and high performance.

JAX's just-in-time (JIT) compilation, enabled by wrapping functions with `jax.jit`, further accelerates code by compiling Python functions to efficient machine code the first time they are called. In this codebase, both the gradient of the Hamiltonian (`jac_h`) and the function that computes the rate of change of the system are JIT-compiled for efficiency.

In our case, the actual execution of the Hamiltonian method `H` is transformed into a JAX-compatible function that can be differentiated and optimized. The execution of this function is triggered when `jax.experimental.ode` is called in `generate_trajectory`. This means that the code inside `H` is not executed in the usual Python way, but rather as part of a JAX computation graph.

One side effect is that print statements may not behave as expected as the code is not executed by the normal Python interpreter. It also means that debugging and control flow can feel different from regular Python. For debugging inside JAX-transformed functions, you should use `jax.debug.print()` rather than the standard `print()` function.

Note that because JAX may trace and compile your functions, code inside these functions can be executed more than once (e.g., during tracing and then during actual computation), and JAX expects functions to be pure (no side effects). This is another reason why debugging output may not appear as expected.

## Attributes of `Dynamics` Class

In the `Dynamics` class, we define the interface of the Hamiltonian method `H`. When defined in a subclass, this will be a normal method that can be called to compute the Hamiltonian of a state of a dynamical system. 

In the constructor of the `Dynamics` class we define the attribute `jac_h`, which is a JAX function that computes the Jacobian of the Hamiltonian with respect to the phase space variables. This is done using `jax.grad(H)`, where `H` is the Hamiltonian method defined in the subclass. It can be called with a phase space vector `x` to compute the Jacobian of the Hamiltonian at that point in phase space, such as:

```python
grad_H = system.jac_h(phase_vector)
```

Although it is called in a way which looks like a method, it is actually not a method, but instead a JAX function stored as an attribute of the `Dynamics` class.

The method `get_rate_of_change` does not actually calculate a rate of change, but instead returns a method (referred to as `grad_x` within the method) that calculates the rate of change of the phase space variables given a phase space vector `x`. It does this using the static method `symplectic_form` and the JAX function `jac_h`.

`generate_trajectory` is a method that uses the JAX ODE solver to integrate the equations of motion for the dynamical system. It uses `get_rate_of_change` to obtain a function that computes the rate of change of the phase space variables, which is then passed to the JAX ODE solver `jax.experimental.odeint`. The ODE solver internally builds a computation graph that allows it to efficiently compute the trajectory of the system over time, and then executes this graph to obtain the trajectory.

## Performance Considerations

JAX is designed for high performance, especially when working with large arrays and complex computations. The use of JAX's automatic differentiation and just-in-time compilation allows for efficient execution of the Hamiltonian dynamics simulations. However, the tracing and compilation process introduces some extra overhead compared to other methods such as the `odeint` function from SciPy. This means that the first time you call a JAX function, it may take longer to execute due to the compilation step. Subsequent calls will be faster as the computation graph is reused.

As a result, this JAX implementation may be slow for small, simple, one-off computations (including many of the examples in this exemplar), but it may be much faster for larger, more complex simulations where the overhead of tracing and compiling is amortized over many computations.