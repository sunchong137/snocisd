import rbm
import optax
import jax
import jax.numpy as jnp
from jax.config import config
# config.update("jax_debug_nans", True)
config.update("jax_enable_x64", True)

def rbm_all(h1e, h2e, mo_coeff, nocc, nvecs, init_params=None, hiddens=[0,1],
            tol=1e-6, MaxIter=1000, **kwargs):
    '''
    Optimize the RBM parameters all together.
    Args:
        h1e: 2D array, one-body Hamiltonian
        h2e: 4D array, two-body Hamiltonian
        mo_coeff: array of size ()
        nocc: int, number of occupied orbitals
        nvecs: int, number of rbm_vectors
    kwargs:
        init_params: a list of vectors, initial guess of the RBM parameters.
        hiddens: hidden variables for RBM neural network.
    NOTE: hard to converge when optimizing all.
    '''

    mo_coeff = jnp.array(mo_coeff)
    h1e = jnp.array(h1e)
    h2e = jnp.array(h2e)

    norb = h1e.shape[-1]
    nvir = norb - nocc
    lt = 2*nvir*nocc # 2 for spins

    # get expansion coefficients
    coeff_hidden = rbm.hiddens_to_coeffs(hiddens, nvecs)
    coeff_hidden = jnp.array(coeff_hidden)

    if init_params is None:
        init_params = jnp.random.rand(nvecs, lt)

    init_params = init_params.flatten(order='C')

    # get combination coefficients


    def cost_func(w):
        w_n = w.reshape(nvecs, -1)
        rmats = rbm.params_to_rmats(w_n, nvir, nocc, coeff_hidden)
        e = rbm.rbm_energy(rmats, mo_coeff, h1e, h2e)
        return e

    def fit(params: optax.Params, optimizer: optax.GradientTransformation) -> optax.Params:

        opt_state = optimizer.init(params)

        @jax.jit
        def step(params, opt_state):
            loss_value, grads = jax.value_and_grad(cost_func)(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss_value

        loss_last = 0
        for i in range(MaxIter):
            params, opt_state, loss_value = step(params, opt_state)
            dloss = loss_value - loss_last

            # if i > 1000 and abs(dloss) < tol:
            #     print(f"Optimization converged after {i+1} steps.")
            #     break
            # else:
            #     loss_last = loss_value
            if i%10 == 0:
                print(f'step {i}, loss: {loss_value};')

        return loss_value, params

    # NOTE: schecule doesn't do too well
    # Schedule learning rate
    # schedule = optax.warmup_cosine_decay_schedule(
    # init_value=1e-2,
    # peak_value=1.0,
    # warmup_steps=20,
    # decay_steps=4000,
    # end_value=1e-3,
    # )

    # optimizer = optax.chain(
    # optax.clip(1.0),
    # optax.adamw(learning_rate=schedule),
    # )

    optimizer = optax.adam(learning_rate=2e-2)
    energy, vecs = fit(init_params, optimizer)


    # params = minimize(cost_func, init_params, method=method, tol=tol, options={"maxiter":MaxIter, "disp": disp}).x
    # final_energy = cost_func(params)

    return energy, vecs