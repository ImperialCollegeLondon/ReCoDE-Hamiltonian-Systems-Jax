<!-- Your Project title, make it sound catchy! -->

# Hamiltonian systems in Jax

In this ReCoDE exemplar, we introduce the basics of Hamiltonian dynamics and demonstrate how Python and JAX can be used to simulate and visualise systems. We go over the mathematical prerequisits and create a framework for solving Hamiltonian equations using the general dynamical systems solver in JAX. Finally, we demonstrate how we can implement simple harmonic oscillator and N-body systems, including visualiation code to plot various system simulated over time.

## Getting Started

To get started, proceed by going to the <a href="01introduction/">Introduction</a> page.

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

Package requiements can be found in the `requirements.txt` file.

The code requires Python and JAX (only CPU suffices, optionally with GPU support). 

## Project Structure

```log
.
├── notebooks
|   ├── getting-started.ipynb
├── hdynamics
│   ├── hdynamics.py
│   └── odeint.py
│   └── utils.py
|   ├── dynamics
|          ├── nbody.py
|          ├── harmonic_oscillator.py
```

## License

This project is licensed under the [BSD-3-Clause license](LICENSE.md)
