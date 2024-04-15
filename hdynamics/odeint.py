import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint

import diffrax

def ode_int(f, x, t_span, t_fun=None, backend='diffrax_backjoint', steps=10, rtol=1e-5, atol=1e-5, adjoint=diffrax.RecursiveCheckpointAdjoint()):
    if backend == 'experimental':
        grad_x = lambda x, t: f(x, t)

        solution = jax.experimental.ode.odeint(grad_x, x, t_span)

        return solution

    elif 'diffrax' in backend:
        def grad_x(t, x, args):
            return f(x, t)

        solver = diffrax.Euler()
        term = diffrax.ODETerm(grad_x)

        if backend == 'diffrax_direct_tspan':
            n_solve_len = (len(t_span) - 1) * steps + 1

            solve_t_span = jnp.linspace(t_span[0], t_span[-1], n_solve_len)
            save_indices = jnp.arange(0, len(solve_t_span), steps)
            save_t_span = solve_t_span[save_indices]

            solution = diffrax.diffeqsolve(term,
                                           solver,
                                           t0=t_span.min(),
                                           t1=t_span.max(),
                                           dt0=None,
                                           saveat=diffrax.SaveAt(subs=diffrax.SubSaveAt(ts=save_t_span)),
                                           y0=x,
                                           #stepsize_controller=diffrax.ConstantStepSize(), #(t_span.max() - t_span.min())/3),
                                           stepsize_controller=diffrax.StepTo(ts=solve_t_span),
                                           max_steps=n_solve_len - 1,
                                           adjoint=adjoint,
                                           )
        elif backend == 'diffrax_direct':
            solution = diffrax.diffeqsolve(term,
                                           solver,
                                           t0=t_span.min(),
                                           t1=t_span.max(),
                                           saveat = diffrax.SaveAt(subs=diffrax.SubSaveAt(t1=True)),
                                           dt0=(t_span.max() - t_span.min())/steps,
                                           y0=x,
                                           stepsize_controller=diffrax.ConstantStepSize(), #(t_span.max() - t_span.min())/3),
                                           max_steps=steps,
                                           adjoint=adjoint,
                                           )
        elif backend == 'diffrax_backjoint':
            solution = diffrax.diffeqsolve(term,
                                           solver,
                                           t0=t_span.min(),
                                           t1=t_span.max(),
                                           saveat=diffrax.SaveAt(subs=diffrax.SubSaveAt(t1=True)),
                                           y0=x,
                                           #dt0=None,
                                           #stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
                                           dt0=(t_span.max() - t_span.min())/steps,
                                           stepsize_controller=diffrax.ConstantStepSize(), #(t_span.max() - t_span.min())/3),
                                           max_steps=steps,
                                           adjoint=diffrax.BacksolveAdjoint(),
                                           )

        else:
            print('Huh.')
            exit(1)

        return solution.ys
    else:
        raise NotImplementedError(f"Unknown ode solver backend: '{backend}'")
