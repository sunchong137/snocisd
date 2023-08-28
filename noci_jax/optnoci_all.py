import numpy as np
import jax, optax
import jax.numpy as jnp
from jax.config import config
# config.update("jax_debug_nans", True)
config.update("jax_enable_x64", True)
from noci_jax import reshf


def optimize_res(h1e, h2e, mo_coeff, nocc, nvecs=None, init_tvecs=None, 
                 MaxIter=5000, print_step=1000, lrate=1e-2, schedule=False):
    ''' 
    Given a set of Thouless rotations, optimize the parameters.
    Res HF approach, all parameters are optimized simultaneously.
    Args:
        
    Kwargs:
        init_vecs: 2D array of size (nvecs, -1)
    '''

    mo_coeff = jnp.array(mo_coeff)
    h1e = jnp.array(h1e)
    h2e = jnp.array(h2e)

    norb = h1e.shape[-1]
    nvir = norb - nocc
    lt = 2*nvir*nocc # 2 for spins

    if init_tvecs is None:
        init_tvecs = np.random.rand(nvecs, lt)
    init_tvecs = jnp.array(init_tvecs)
    if nvecs is None:
        nvecs = len(init_tvecs)
    
        
    init_tvecs = init_tvecs.flatten(order="C")

    # first construct the HF state
    rot0_u = jnp.zeros((nvir+nocc, nocc))
    rot0_u = rot0_u.at[:nocc, :nocc].set(jnp.eye(nocc))
    rot_hf = jnp.array([[rot0_u, rot0_u]]) # the HF state

    E0 = reshf.rbm_energy(rot_hf, mo_coeff, h1e, h2e, return_mats=False)
    
    def cost_func(t):
        tvecs = t.reshape(nvecs, -1)
        rmats = reshf.tvecs_to_rmats(tvecs, nvir, nocc)
        rmats = jnp.vstack([rot_hf, rmats])
        e = reshf.rbm_energy(rmats, mo_coeff, h1e, h2e, return_mats=False)
        return e

    def fit(params: optax.Params, Niter: int, lrate) -> optax.Params:

        optimizer = optax.adam(learning_rate=lrate)
        opt_state = optimizer.init(params)

        @jax.jit 
        def step(params, opt_state):
            loss_value, grads = jax.value_and_grad(cost_func)(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss_value

        for i in range(Niter):
            params, opt_state, loss_value = step(params, opt_state)

            if i % print_step == 0:
                print(f'step {i+1}, Energy: {loss_value};')

        return loss_value, params

    if schedule:
        # schedule
        niter1 = int(MaxIter / 1.5)
        niter2 = MaxIter - niter1
        lrate2 = lrate / 2.

        # optimizer = optax.adam(learning_rate=lrate)
        energy0, vecs = fit(init_tvecs, niter1, lrate)
        print(f"Energy lowered: {energy0 - E0}")
        energy, vecs = fit(vecs, niter2, lrate2)
        print(f"Energy lowered: {energy - energy0}")
    else:
        energy, vecs = fit(init_tvecs, MaxIter, lrate)
 

    return energy, vecs

