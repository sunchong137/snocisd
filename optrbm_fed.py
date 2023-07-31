import numpy as np
import slater, rbm
import optax
import jax
import jax.numpy as jnp
from jax.config import config
config.update("jax_debug_nans", True)


def rbm_fed(h1e, h2e, mo_coeff, nocc, nvecs,
            init_params=None, ao_ovlp=None, hiddens=[0,1],
            nsweep=3, tol=1e-7, MaxIter=100):
    '''
    Kwargs:
        nsweep: maximum number of sweeps
    Optimize the RBM parameters one by one.
    '''
    mo_coeff = jnp.array(mo_coeff)
    h1e = jnp.array(h1e)
    h2e = jnp.array(h2e)
    if ao_ovlp is not None:
        ao_ovlp = jnp.array(ao_ovlp)

    norb = h1e.shape[-1]
    nvir = norb - nocc
    tshape = (nvir, nocc)

    if init_params is None:
        init_params = jnp.array(np.random.rand(nvecs, 2*nvir*nocc)) # 2 for spins

    rot0_u = jnp.zeros((nvir+nocc, nocc))
    rot0_u = rot0_u.at[:nocc, :nocc].set(jnp.eye(nocc))
    rot_hf = jnp.array([[rot0_u, rot0_u]]) # the HF state
    E0 = rbm.rbm_energy(rot_hf, mo_coeff, h1e, h2e)
    e_hf = E0

    opt_rbms = [] # optimized RBM vectors
    opt_tvecs = jnp.array([np.zeros(2*nvir*nocc)]) # All Thouless vectors

    rmats = rbm.tvecs_to_rmats(opt_tvecs, nvir, nocc)
    hmat, smat = rbm.rbm_energy(rmats, mo_coeff, h1e, h2e, return_mats=True)

    # get expansion coefficients
    coeff_hidden = rbm.hiddens_to_coeffs(hiddens, nvecs)
    coeff_hidden = jnp.array(coeff_hidden)

    print("Start RBM FED...")
    for iter in range(nvecs):
        print(f"*****Optimizing Determinant {iter+1}*****")
        w0 = init_params[iter]
        e, w = opt_one_rbmvec(w0, opt_tvecs, h1e, h2e, mo_coeff, tshape,
                              hmat=hmat, smat=smat, tol=tol, MaxIter=MaxIter)

        opt_rbms.append(w)
        de = e - E0
        E0 = e
        print(f"##### Done optimizing determinant {iter+1}, energy lowered {de} #####")
        new_tvecs = rbm.add_vec(w, opt_tvecs) # new Thouless vectors from adding this RBM vector
        opt_tvecs = jnp.vstack([opt_tvecs, new_tvecs])
        # update hmat and smat
        lv = 2**iter
        h_n = jnp.zeros((lv*2, lv*2))
        s_n = jnp.zeros((lv*2, lv*2))
        h_n = h_n.at[:lv, :lv].set(hmat)
        s_n = s_n.at[:lv, :lv].set(smat)
        rmats_n = rbm.tvecs_to_rmats(new_tvecs, nvir, nocc)
        hmat, smat = _expand_hs(h_n, s_n, rmats_n, rmats, h1e, h2e, mo_coeff)
        rmats = jnp.vstack([rmats, rmats_n])

    if nsweep > 0:
        if nvecs < 2:
            print("WARNING: No sweeps needed for only one determinant!")
        else:
            print("Start sweeping...")

            for isw in range(nsweep):
                E_s = E0
                print("Sweep {}".format(isw+1))
                for iter in range(nvecs):
                    # always pop the first vector and add the optimized to the end
                    w0 = opt_rbms.pop(0)
                    fixed_vecs = rbm.expand_vecs(opt_rbms, coeff_hidden) # TODO not efficient
                    e, w = opt_one_rbmvec(w0, fixed_vecs, h1e, h2e, mo_coeff, tshape,
                                          hmat=None, smat=None, tol=tol, MaxIter=MaxIter)
                    de = e - E0
                    E0 = e
                    print("Iter {}: energy lowered {}".format(iter+1, de))
                    opt_rbms.append(w)
                de_s = e - E_s
                print("***Energy lowered after Sweep {}: {}".format(isw+1, de_s))

    print("Total energy lowered: {}".format(e - e_hf))
    return e, opt_rbms

def opt_one_rbmvec(vec0, tvecs, h1e, h2e, mo_coeff, tshape, 
                   hmat=None, smat=None, tol=1e-7, MaxIter=100):
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
    nvecs = len(tvecs)
    rmats = rbm.tvecs_to_rmats(tvecs, nvir, nocc)

    if hmat is None: # assume smat is also None
        hmat, smat = rbm.rbm_energy(rmats, mo_coeff, h1e, h2e, return_mats=True)

    h_n = jnp.zeros((nvecs*2, nvecs*2))
    s_n = jnp.zeros((nvecs*2, nvecs*2))
    h_n = h_n.at[:nvecs, :nvecs].set(jnp.copy(hmat))
    s_n = s_n.at[:nvecs, :nvecs].set(jnp.copy(smat))

    tvecs = jnp.array(tvecs)
    def cost_func(w):
        tvecs_n = rbm.add_vec(w, tvecs) # newly added Thouless vectors
        rmats_n = rbm.tvecs_to_rmats(tvecs_n, nvir, nocc)
        hm, sm = _expand_hs(h_n, s_n, rmats_n, rmats, h1e, h2e, mo_coeff)
        e = rbm.solve_lc_coeffs(hm, sm)
        return e

    init_params = jnp.array(vec0)

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
            if i%500 == 0:
                print(f'step {i}, loss: {loss_value};')

        return loss_value, params

    # TODO figure out learning rate
    optimizer = optax.adam(learning_rate=1e-2)
    energy, vec = fit(init_params, optimizer, MaxIter=int(MaxIter))

    del fit # release memory

    return energy, vec

def _expand_hs(h_n, s_n, rmats_n, rmats_fix, h1e, h2e, mo_coeff):
    '''
    Expand the H matrix and S matrix
    | (fix, fix)   (fix, n)|
    | (n, fix)     (n, n)  |
    (fix, fix) is given by h_n and s_n
    we evaluate (n, fix) and (n, n)  
    '''
    nvecs = len(rmats_n)
    hm = jnp.copy(h_n)
    sm = jnp.copy(s_n)

    # generate hmat and smat for the lower left block and upper right block
    h_new, s_new = rbm.gen_hmat(rmats_n, rmats_fix, mo_coeff, h1e, h2e)
    hm = hm.at[nvecs:, :nvecs].set(h_new)
    hm = hm.at[:nvecs, nvecs:].set(h_new.T.conj())
    sm = sm.at[nvecs:, :nvecs].set(s_new)
    sm = sm.at[:nvecs, nvecs:].set(s_new.T.conj())

    # generate hmat and smat for the lower diagonal block
    h_new, s_new = rbm.rbm_energy(rmats_n, mo_coeff, h1e, h2e, return_mats=True)
    hm = hm.at[nvecs:, nvecs:].set(h_new)
    sm = sm.at[nvecs:, nvecs:].set(s_new)

    return hm, sm