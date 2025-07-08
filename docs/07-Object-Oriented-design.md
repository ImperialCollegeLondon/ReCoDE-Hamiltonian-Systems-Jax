# Object-Oriented Design

In this notebook, we'll explore the object-oriented design of the Hamiltonian systems code base, including explaining the design philosophy.

## Inheritance

The design of the code makes use of the fact that there are many different dynamical systems that can be described by Hamiltonian mechanics. By defining `Dynamics` as a base class, we can include components that are common to all dynamical systems in this class. This includes some more complex methods such as `generate_trajectory`, which indirectly uses the Hamiltonian method `H` to generate trajectories.

This makes reusing the entire code of this class easier, as we can simply inherit from this class when implementing a new dynamical system. The child class can then implement the Hamiltonian and any other system-specific functionality while still benefiting from the common features provided by the base class.

## Abstract Class

By making `Dynamics` an abstract class, we prevent it from being instantiated directly. This is useful because `Dynamics` is not a specific dynamical system, but rather a blueprint for any dynamical system. The abstract class enforces that any child class must implement the Hamiltonian method `H`, which is essential for defining the dynamics of the system. `H` is a good example of an abstract method as it is relied upon by other methods in the `Dynamics` class, such as `generate_trajectory`.

## Optional Methods

The child classes can optionally define the method `plot_trajectory`. This method is not abstract, meaning that it is not required for the child class to implement it. If the method is called in a child class that does not implement it, it will raise a `NotImplementedError`. This allows for flexibility in the design, as not all dynamical systems may require a specific plotting method.

## Caching

In the different classes, some calculations are performed in the constructor and stored in instance variables. This is done to avoid recalculating these values repeatedly as the simulation progresses. For example, the `cdim` variable is calculated in the constructor and stored as an instance variable, so it can be reused in other methods without recalculating it each time.

For example, in the `Dynamics` class, the `pdim` variable (representing the length of the phase vector) is calculated in the constructor and stored as an instance variable. The JAX function `jac_h` is also cached in the constructor to avoid recalculating it each time it is needed. This caching mechanism improves performance by reducing redundant calculations during the simulation. In the `NBody` class, the `mass_outer_product` variables is also calculated and cached in the constructor. This is possible because the mass of the bodies does not change during the simulation, so  the product of the masses can be calculated once in advance and reused throughout the simulation.