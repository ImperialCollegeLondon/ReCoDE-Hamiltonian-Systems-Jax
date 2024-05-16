<!-- Your Project title, make it sound catchy! -->

# Hamiltonian systems in Jax

<!-- Provide a short description to your project -->

## Description

This ReCoDE is an introduction to Hamiltonian dynamics, and contains demonstrations of how Python and Jax can be used to simulate and visualise phase space and associated trajectories.

## Learning Outcomes

- Basic understanding of conservative systems
- Learn how to simulate dynamics using Jax in Python
- Ability to visualise 2d and 3d dynamics

| Task       | Time    |
| ---------- | ------- |
| Reading    | 3 hours |
| Practising | 3 hours |

## Requirements

Undergraduate level calculus, linear algebra and a basic understanding of physics are required.

Resources:

<!--
If your exemplar requires students to have a background knowledge of something
especially this is the place to mention that.

List any resources you would recommend to get the students started.

If there is an existing exemplar in the ReCoDE repositories link to that.
-->

### Academic

<!-- List the system requirements and how to obtain them, that can be as simple
as adding a hyperlink to as detailed as writting step-by-step instructions.
How detailed the instructions should be will vary on a case-by-case basis.

Here are some examples:

- 50 GB of disk space to hold Dataset X
- Anaconda
- Python 3.11 or newer
- Access to the HPC
- PETSc v3.16
- gfortran compiler
- Paraview
-->

### System

<!-- Instructions on how the student should start going through the exemplar.

Structure this section as you see fit but try to be clear, concise and accurate
when writing your instructions.

For example:
Start by watching the introduction video,
then study Jupyter notebooks 1-3 in the `intro` folder
and attempt to complete exercise 1a and 1b.

Once done, start going through through the PDF in the `main` folder.
By the end of it you should be able to solve exercises 2 to 4.

A final exercise can be found in the `final` folder.

Solutions to the above can be found in `solutions`.
-->

## Getting Started

Notebook:

``` notebooks/[plot] Example nbody trajectory plot.ipynb```

<!-- An overview of the files and folder in the exemplar.
Not all files and directories need to be listed, just the important
sections of your project, like the learning material, the code, the tests, etc.

A good starting point is using the command `tree` in a terminal(Unix),
copying its output and then removing the unimportant parts.

You can use ellipsis (...) to suggest that there are more files or folders
in a tree node.

-->

## Project Structure

```log
.
├── hdynamics
│   ├── hdynamics.py
│   └── odeint.py
│   └── utils.py
├── notebooks
|   ├── ...
├── ...
```

## License

This project is licensed under the [BSD-3-Clause license](LICENSE.md)
