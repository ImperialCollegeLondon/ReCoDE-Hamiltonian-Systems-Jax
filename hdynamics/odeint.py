"""ODE solver."""

import diffrax
import jax
import jax.numpy as jnp
import jax.experimental.ode


def ode_int(
    f,
    x,
    t_span,
    t_fun=None,
    rtol=1e-5,
    atol=1e-5,
):
    """Solve ODE equation f over time t_span.

    Input:
            f (func): differential equation
            x (vector): initial condition
            t_span (range): time points to integrate over
            backend (str): backend used for solving
            rtol (float): relative tolerance
            atol (float): absolute tolerance
            adjoint: type of adjoint method used
    """

    try:
        solution = jax.experimental.ode.odeint(f, x, t_span, rtol=rtol, atol=atol)
    except ValueError as e:
        print(f"Could not solve ODE: ", e)
    
    return solution

