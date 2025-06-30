# Hamiltonian systems in Jax

In this ReCoDE exemplar, we introduce the basics of Hamiltonian dynamics and demonstrate how Python and JAX can be used to simulate and visualise systems. We go over the mathematical prerequisites and create a framework for solving Hamiltonian equations using the general dynamical systems solver in JAX. Finally, we demonstrate how we can implement simple harmonic oscillator and N-body systems, including visualisation code to plot various system simulated over time.

## Getting Started

To get started, proceed by going to the <a href="02Introduction-to-Hamiltonian-systems/">Introduction</a> page.

## Learning Outcomes

The main learning outcomes are to provide a:

- Basic understanding of conservative systems
- Learn how to simulate dynamics using Jax in Python
- Ability to visualise 2d and 3d dynamics

| Task       | Time    |
| ---------- | ------- |
| Reading    | 3 hours |
| Practising | 3 hours |

## Requirements

Undergraduate level calculus, linear algebra and a basic understanding of physics are required.

## System

The code requires Python and JAX (only CPU suffices, optionally with GPU support). 

If you are installing the code on a Windows machine, you will need to first install the Microsoft Visual C++ Build Tools. You can find the installer [here](https://visualstudio.microsoft.com/visual-cpp-build-tools/). If prompted, select "Desktop development with C++" and install the required components.

If you want to run the code in the project only, you can install the required packages from the `pyproject.toml` file using the following command:

```bash
pip install -e .
```

If you want to develop the code, you can install the required packages from the `pyproject.toml` file using the following command:

```bash
pip install -e .[dev]
```

## License

This project is licensed under the [BSD-3-Clause license](LICENSE.md)
