import numpy as np
import jax, optax
import jax.numpy as jnp
from jax.config import config
config.update("jax_debug_nans", True)
config.update("jax_enable_x64", True)
import rbm

def optimize_fed(h1e, h2e, mo_coeff, nocc, nvecs=None, init_tvecs=None, MaxIter=100, nsweep=0):
    '''
    Given a set of Thouless rotations, optimize the parameters.
    Using FED (few-determinant) approach.
    Args:
        tmats0: a list of Thouless rotations as the initial guess.
        mo_coeff: a list of two 2D arrays
        h1e: 2D array, one-body Hamiltonian 
        h2e: 4D array, two-body Hamiltonian
    Kwargs:
        tol: threshold to terminate minimization
        MaxIter: maximum number of iterations
        nsweep: number of sweeps

    Returns:
        float, final energy
        a list of arrays, the optimized Thouless parameters.
    '''

    mo_coeff = jnp.array(mo_coeff)
    h1e = jnp.array(h1e)
    h2e = jnp.array(h2e)

    norb = h1e.shape[-1]
    nvir = norb - nocc
    tshape = (nvir, nocc)
    lt = 2*nvir*nocc # 2 for spins

    if init_tvecs is None:
        init_tvecs = np.random.rand(nvecs, lt)
    init_tvecs = jnp.array(init_tvecs)

    rot0_u = jnp.zeros((nvir+nocc, nocc))
    rot0_u = rot0_u.at[:nocc, :nocc].set(jnp.eye(nocc))
    rmats_new = jnp.array([[rot0_u, rot0_u]]) # the HF state

    hmat, smat = rbm.rbm_energy(rmats_new, mo_coeff, h1e, h2e, return_mats=True)
    e_hf = rbm.solve_lc_coeffs(hmat, smat)
    E0 = e_hf 
    if nvecs is None:
        nvecs = len(init_tvecs)

    # Start optimization
    print("Starting NOCI optimization with FED approach...")

    for iter in range(nvecs):
        print(f"*****Optimizing Determinant {iter+1}*****")
        t0 = init_tvecs[iter]
        smat0 = jnp.copy(smat)
        hmat0 = jnp.copy(hmat)
        E, t = opt_one_thouless(t0, rmats_new, mo_coeff, h1e, h2e, tshape, 
                                hmat=hmat0, smat=smat0, MaxIter=MaxIter)
        de = E - E0
        print("Iter {}: energy lowered {}".format(iter+1, de))
        E0 = E
        init_tvecs = init_tvecs.at[iter].set(jnp.copy(t))
        r = rbm.tvecs_to_rmats(jnp.array([t]), nvir, nocc) # TODO implement one vector case
        h_n = jnp.zeros((iter+2, iter+2))
        s_n = jnp.zeros((iter+2, iter+2))
        h_n = h_n.at[:iter+1, :iter+1].set(hmat0)
        s_n = s_n.at[:iter+1, :iter+1].set(smat0)
        hmat, smat = _expand_hs(h_n, s_n, r, rmats_new, h1e, h2e, mo_coeff)
        rmats_new = jnp.vstack([rmats_new, r])

    de_fed = E - e_hf  
    print("***Energy lowered after FED: {}".format(de_fed))

    # # Start sweeping
    # if nsweep > 0:
    #     print("Start sweeping...")
    #     sdets = slater.gen_determinants(mo_coeff, rmats_new)
    #     hmat = noci.full_hamilt_w_sdets(sdets, h1e, h2e, ao_ovlp=ao_ovlp)
    #     smat = noci.full_ovlp_w_rotmat(rmats_new) 
 
    #     for isw in range(nsweep):
    #         print("Sweep {}".format(isw+1))
    #         E_s = E0
    #         for iter in range(num_t):
    #             t0 = tmats0[iter]
    #             rmats_new.pop(1)
    #             hmat0 = np.delete(hmat, 1, axis=0)
    #             hmat0 = np.delete(hmat0, 1, axis=1)
    #             smat0 = np.delete(smat, 1, axis=0)
    #             smat0 = np.delete(smat0, 1, axis=1)
    #             E, t, hmat, smat = opt_one_thouless(t0, rmats_new, mo_coeff, h1e, h2e, hmat=hmat0, smat=smat0, MaxIter=MaxIter)
    #             de = E - E0
    #             print("Iter {}: energy lowered {}".format(iter+1, de))
    #             E0 = E
    #             tmats0[iter] = np.copy(t)
    #             r = slater.thouless_to_rotation(t, normalize=True)
    #             rmats_new.append(r)
    #         de_s = E - E_s 
    #         print("***Energy lowered after Sweep {}: {}".format(isw+1, de_s))

    # len_new_rots = len(rmats_new)
    # if len_new_rots < num_t:
    #     print("WARNING: only {} vectors are successfully optimized!".format(len_new_rots))

    de_tot = E - e_hf 
    print("SUMMARY: Total energy lowered {}".format(de_tot))
    return E, init_tvecs


def opt_one_thouless(tvec0, rmats, mo_coeff, h1e, h2e, tshape, hmat=None, smat=None, MaxIter=100):


    nvecs = len(rmats) + 1
    nvir, nocc = tshape

    if hmat is None:
        hmat, smat = rbm.rbm_energy(rmats, mo_coeff, h1e, h2e, return_mats=True)

    h_n = jnp.zeros((nvecs, nvecs))
    s_n = jnp.zeros((nvecs, nvecs))
    h_n = h_n.at[:-1, :-1].set(jnp.copy(hmat))
    s_n = s_n.at[:-1, :-1].set(jnp.copy(smat))

    def cost_func(t): 
        _t = jnp.array([t])
        # thouless to rotation
        r_n = rbm.tvecs_to_rmats(_t, nvir, nocc)
        hm, sm = _expand_hs(h_n, s_n, r_n, rmats, h1e, h2e, mo_coeff)
        energy = rbm.solve_lc_coeffs(hm, sm)
        return energy  
          
    init_params = jnp.array(tvec0)

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
    nvecs = len(rmats_fix)
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