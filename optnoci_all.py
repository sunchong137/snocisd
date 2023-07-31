import jax, optax
import jax.numpy as jnp
from jax.config import config
# config.update("jax_debug_nans", True)
config.update("jax_enable_x64", True)
import rbm


def optimize_res(h1e, h2e, mo_coeff, nocc, nvecs=None, init_tvecs=None, 
                tol=1e-8, MaxIter=100):
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
        init_tvecs = jnp.random.rand(nvecs, lt)
    init_tvecs = init_tvecs.flatten(order="C")

    # first construct the HF state
    rot0_u = jnp.zeros((nvir+nocc, nocc))
    rot0_u = rot0_u.at[:nocc, :nocc].set(jnp.eye(nocc))
    rot_hf = jnp.array([[rot0_u, rot0_u]]) # the HF state

    E0 = rbm.rbm_energy(rot_hf, mo_coeff, h1e, h2e, return_mats=False)
    
    def cost_func(t):
        tvecs = t.reshape(nvecs, -1)
        rmats = rbm.tvecs_to_rmats(tvecs, nvir, nocc)
        rmats = jnp.vstack([rot_hf, rmats])
        e = rbm.rbm_energy(rmats, mo_coeff, h1e, h2e, return_mats=False)
        return e

    def fit(params: optax.Params, optimizer: optax.GradientTransformation) -> optax.Params:

        opt_state = optimizer.init(params)

        @jax.jit
        def step(params, opt_state):
            loss_value, grads = jax.value_and_grad(cost_func)(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss_value

        for i in range(MaxIter):
            params, opt_state, loss_value = step(params, opt_state)

            if i%500 == 0:
                print(f'step {i}, loss: {loss_value};')

        return loss_value, params

    optimizer = optax.adam(learning_rate=2e-2)
    energy, vecs = fit(init_tvecs, optimizer)
    print(f"Energy lowered: {energy - E0}")

    return energy, vecs

