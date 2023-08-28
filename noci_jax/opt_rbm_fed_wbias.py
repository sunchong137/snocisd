# hiddens are always [0, 1]
import numpy as np
from noci_jax import reshf
import optax
import jax
import jax.numpy as jnp
from jax.config import config
# config.update("jax_debug_nans", True)
config.update("jax_enable_x64", True)


def rbm_fed(h1e, h2e, mo_coeff, nocc, nvecs, init_params=None, bias=None, MaxIter=5000, print_step=1000):
    '''
    Optimize the RBM parameters one by one.
    '''
    mo_coeff = jnp.array(mo_coeff)
    h1e = jnp.array(h1e)
    h2e = jnp.array(h2e)

    norb = h1e.shape[-1]
    nvir = norb - nocc
    tshape = (nvir, nocc)

    if init_params is None:
        init_params = np.random.rand(nvecs, 2*nvir*nocc) # 2 for spins
    if bias is None:
        bias = np.random.rand(nvecs)

    init_params = jnp.array(init_params)
    bias = jnp.array(bias)

    rot0_u = jnp.zeros((nvir+nocc, nocc))
    rot0_u = rot0_u.at[:nocc, :nocc].set(jnp.eye(nocc))
    rot_hf = jnp.array([[rot0_u, rot0_u]]) # the HF state

    E0 = reshf.rbm_energy(rot_hf, mo_coeff, h1e, h2e)
    e_hf = E0

    opt_tvecs = jnp.array([np.zeros(2*nvir*nocc)]) # All Thouless vectors
    opt_lc = jnp.array([0]) # all optimized lc_coeffs (ln)

    rmats = reshf.tvecs_to_rmats(opt_tvecs, nvir, nocc)
    hmat, smat = reshf.rbm_energy(rmats, mo_coeff, h1e, h2e, return_mats=True)

    print("Start RBM FED...")
    for iter in range(nvecs):
        print(f"*****Optimizing Determinant {iter+1}*****")
        w0 = init_params[iter]
        b0 = bias[iter]
        e, v = opt_one_rbmvec(w0, b0, opt_tvecs, opt_lc, h1e, h2e, mo_coeff, tshape,
                              hmat=hmat, smat=smat,MaxIter=MaxIter, print_step=print_step)
        w = v[:-1]
        b = v[-1]
        init_params = init_params.at[iter].set(jnp.copy(w))
        bias = bias.at[iter].set(b)
        de = e - E0
        E0 = e
        print(f"##### Done optimizing determinant {iter+1}, energy lowered {de} #####")
        new_tvecs = reshf.add_vec(w, opt_tvecs) # new Thouless vectors from adding this RBM vector
        opt_tvecs = jnp.vstack([opt_tvecs, new_tvecs])
        opt_lc = jnp.concatenate([opt_lc, opt_lc + b])
        # update hmat and smat
        rmats_n = reshf.tvecs_to_rmats(new_tvecs, nvir, nocc)
        hmat, smat = reshf.expand_hs(hmat, smat, rmats_n, rmats, h1e, h2e, mo_coeff)
        rmats = jnp.vstack([rmats, rmats_n])


    print("Total energy lowered: {}".format(e - e_hf))
    return e, init_params, bias


def rbm_sweep(h1e, h2e, mo_coeff, nocc, init_params, bias, E0=None, hiddens=[0,1], 
              nsweep=1, MaxIter=5000, print_step=1000):

    nvecs = len(init_params)
    coeff_hidden = reshf.hiddens_to_coeffs(hiddens, nvecs-1)
    coeff_hidden = jnp.array(coeff_hidden)

    if nsweep < 1 or nvecs < 2:
        print("Number of sweeps needs to be > 1!")
        print("Number of new determinants needs to be > !")
        print("No sweep performed.")
        if E0 is None:
            coeff_hidden = reshf.hiddens_to_coeffs(hiddens, nvecs)
            coeff_hidden = jnp.array(coeff_hidden)
            rmats = reshf.params_to_rmats(init_params, nvir, nocc, coeff_hidden)
            E0 = reshf.rbm_energy(rmats, mo_coeff, h1e, h2e)
        return E0, init_params
    
    mo_coeff = jnp.array(mo_coeff)
    h1e = jnp.array(h1e)
    h2e = jnp.array(h2e)

    norb = h1e.shape[-1]
    nvir = norb - nocc
    tshape = (nvir, nocc)
    if E0 is None:
        rmats = reshf.params_to_rmats(init_params, nvir, nocc, coeff_hidden)
        E0 = reshf.rbm_energy(rmats, mo_coeff, h1e, h2e)
    
    print("Start sweeping...")
    for isw in range(nsweep):
        E_s = E0
        print("Sweep {}".format(isw+1))
        for iter in range(nvecs):
            # always pop the first vector and add the optimized to the end
            w0 = init_params[iter]
            b0 = bias[iter]
            params_fix = jnp.delete(init_params, iter, axis=0)
            bias_fix = jnp.delete(bias, iter, axis=0)
            fixed_vecs = reshf.expand_vecs(params_fix, coeff_hidden) 
            fixed_lc = reshf.expand_vecs(bias_fix, coeff_hidden) 
            E, v = opt_one_rbmvec(w0, b0, fixed_vecs, fixed_lc, h1e, h2e, mo_coeff, tshape,
                                hmat=None, smat=None, MaxIter=MaxIter, print_step=print_step)
            de = E - E0
            E0 = E
            print("Iter {}: energy lowered {}".format(iter+1, de))
            init_params = init_params.at[iter].set(jnp.copy(v[:-1]))
            bias = bias.at[iter].set(v[-1])
      
        de_s = E - E_s
        print("***Energy lowered after Sweep {}: {}".format(isw+1, de_s))
    
    return E, init_params, bias


def opt_one_rbmvec(vec0, bias0, tvecs, coeffs, h1e, h2e, mo_coeff, tshape, 
                   hmat=None, smat=None, MaxIter=5000, print_step=1000):
    '''
    Optimize one RBM vector with the other fixed.
    Args:
        vec0: 1D array, the RBM vector to be optimized.
        tvecs: a list of 1D arrays, previous Thouless vectors.

    Returns:
        float: energy
        1D array: optimized RBM vector.
    '''
    nvir, nocc = tshape
    rmats = reshf.tvecs_to_rmats(tvecs, nvir, nocc)

    if hmat is None: # assume smat is also None
        hmat, smat = reshf.rbm_energy(rmats, mo_coeff, h1e, h2e, return_mats=True)

    tvecs = jnp.array(tvecs)

    def cost_func(v):
        w = v[:-1]
        b = v[-1]
        tvecs_n = reshf.add_vec(w, tvecs) # newly added Thouless vectors
        lc_n = coeffs + b
        lc_coeffs = jnp.concatenate([coeffs, lc_n])
        lc_coeffs = jnp.exp(lc_coeffs)
        rmats_n = reshf.tvecs_to_rmats(tvecs_n, nvir, nocc)
        hm, sm = reshf.expand_hs(hmat, smat, rmats_n, rmats, h1e, h2e, mo_coeff)
        h = lc_coeffs.conj().T.dot(hm).dot(lc_coeffs)
        s = lc_coeffs.conj().T.dot(sm).dot(lc_coeffs)
        e = h/s
        return e

    init_params = jnp.concatenate([vec0, jnp.array([bias0])])

    def fit(params: optax.Params, optimizer: optax.GradientTransformation, MaxIter=MaxIter) -> optax.Params:

        opt_state = optimizer.init(params)

        @jax.jit
        def step(params, opt_state):
            loss_value, grads = jax.value_and_grad(cost_func)(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss_value

        # loss_last = 0
        for i in range(MaxIter):
            params, opt_state, loss_value = step(params, opt_state)

            # dloss = loss_value - loss_last
            # if i > 1000 and abs(dloss) < tol:
            #     a = 1
            #     # print(f"Optimization converged after {i+1} steps.")
            #     # break
            # else:
            #     loss_last = loss_value
            if (i+1) % print_step == 0:
                print(f'step {i+1}, loss: {loss_value};')

        return loss_value, params

    # TODO figure out learning rate
    optimizer = optax.adam(learning_rate=1e-2)
    energy, vec = fit(init_params, optimizer, MaxIter=int(MaxIter))

    del fit # release memory

    return energy, vec

