import numpy as np
import slater, noci, rbm
import optax
import jax
import jax.numpy as jnp
from jax.config import config
config.update("jax_debug_nans", True)


def rbm_fed(h1e, h2e, mo_coeff, nocc, nvecs,
            init_rbms=None, ao_ovlp=None, hiddens=[0,1],
            nsweep=3, tol=1e-7, MaxIter=100, disp=False, method="BFGS"):
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

    if init_rbms is None:
        init_rbms = jnp.random.rand(nvecs, 2*nvir*nocc) # 2 for spins

    rot0_u = np.zeros((nvir+nocc, nocc))
    rot0_u[:nocc, :nocc] = np.eye(nocc)
    rot_hf = np.array([rot0_u, rot0_u]) # the HF state
    E0 = noci.noci_energy([rot_hf], mo_coeff, h1e, h2e, ao_ovlp=ao_ovlp, include_hf=True)
    e_hf = E0

    opt_rbms = [] # optimized RBM vectors
    opt_tvecs = jnp.array([np.zeros(2*nvir*nocc)]) # All Thouless vectors

    rmats = rbm.tvecs_to_rotations(opt_tvecs, tshape, normalize=True)
    sdets = slater.gen_determinants(mo_coeff, rmats)

    hmat = noci.full_hamilt_w_sdets(sdets, h1e, h2e, ao_ovlp=ao_ovlp)
    smat = noci.full_ovlp_w_rotmat(rmats)


    print("Start RBM FED...")
    for iter in range(nvecs):
        print(f"*****Optimizing Determinant {iter+1}*****")
        w0 = init_rbms[iter]
        e, w = opt_one_rbmvec(w0, opt_tvecs, h1e, h2e, mo_coeff, tshape,
                              ao_ovlp=ao_ovlp, hmat=None, smat=None,
                              tol=tol, MaxIter=MaxIter, disp=disp, method=method)
        #TODO keep on debugging

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
        rmats_n = rbm.tvecs_to_rotations(new_tvecs, tshape, normalize=True)
        sdets_n = slater.gen_determinants(mo_coeff, rmats_n)
        hmat, smat = _expand_hs(h_n, s_n, rmats_n, sdets_n, rmats, sdets, h1e, h2e, ao_ovlp=ao_ovlp)
        rmats = jnp.vstack([rmats, rmats_n])
        sdets = jnp.vstack([sdets, sdets_n])

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
                    opt_vecs = rbm.expand_vecs(opt_rbms) # TODO not efficient
                    e, w = opt_one_rbmvec(w0, opt_vecs, h1e, h2e, mo_coeff, tshape,
                                        ao_ovlp=ao_ovlp, hmat=None, smat=None, tol=tol, MaxIter=MaxIter)
                    de = e - E0
                    E0 = e
                    print("Iter {}: energy lowered {}".format(iter+1, de))
                    opt_rbms.append(w)
                de_s = e - E_s
                print("***Energy lowered after Sweep {}: {}".format(isw+1, de_s))

    print("Total energy lowered: {}".format(e - e_hf))
    return e, opt_rbms

def opt_one_rbmvec(vec0, tvecs, h1e, h2e, mo_coeff, tshape, ao_ovlp=None,
                   hmat=None, smat=None, tol=1e-7, MaxIter=100, disp=False, method="BFGS"):
    '''
    Optimize one RBM vector with the other fixed.
    Args:
        vec0: 1D array, the RBM vector to be optimized.
        tvecs: a list of 1D arrays, previous Thouless vectors.

    Returns:
        float: energy
        1D array: optimized RBM vector.
    '''

    nvecs = len(tvecs)
    rmats = rbm.tvecs_to_rotations(tvecs, tshape, normalize=True)
    sdets = slater.gen_determinants(mo_coeff, rmats)

    if hmat is None: # construct previous Hamiltonian matrix
        hmat = noci.full_hamilt_w_sdets(sdets, h1e, h2e, ao_ovlp=ao_ovlp)
    if smat is None: # construct previous overlap matrix
        smat = noci.full_ovlp_w_rotmat(rmats)

    h_n = jnp.zeros((nvecs*2, nvecs*2))
    s_n = jnp.zeros((nvecs*2, nvecs*2))
    h_n = h_n.at[:nvecs, :nvecs].set(jnp.copy(hmat))
    s_n = s_n.at[:nvecs, :nvecs].set(jnp.copy(smat))

    tvecs = jnp.array(tvecs)
    def cost_func(w):
        tvecs_n = rbm.add_vec(w, tvecs) # newly added Thouless vectors
        rmats_n = rbm.tvecs_to_rotations(tvecs_n, tshape, normalize=True)
        sdets_n = slater.gen_determinants(mo_coeff, rmats_n)
        hm, sm = _expand_hs(h_n, s_n, rmats_n, sdets_n, rmats, sdets, h1e, h2e, ao_ovlp=ao_ovlp)
        e = noci.solve_lc_coeffs(hm, sm)

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

        loss_last = 0
        for i in range(MaxIter):
            params, opt_state, loss_value = step(params, opt_state)
            dloss = loss_value - loss_last

            if i > 1000 and abs(dloss) < tol:
                a = 1
                # print(f"Optimization converged after {i+1} steps.")
                # break
            else:
                loss_last = loss_value
            if i%500 == 0:
                print(f'step {i}, loss: {loss_value};')

        return loss_value, params

    # # NOTE: schecule doesn't do too well
    # # Schedule learning rate
    # schedule = optax.warmup_cosine_decay_schedule(
    # init_value=0,
    # peak_value=1.0,
    # warmup_steps=100,
    # decay_steps=1_000,
    # end_value=1e-2,
    # )

    # optimizer = optax.chain(
    # #optax.clip(1.0),
    # optax.adamw(learning_rate=schedule),
    # )

    # TODO figure out learning rate
    optimizer = optax.adam(learning_rate=1e-2)
    energy, vec = fit(init_params, optimizer, MaxIter=int(MaxIter))

    # optimizer2 = optax.adam(learning_rate=5e-3)
    # energy, vec = fit(vec, optimizer2, MaxIter=int(MaxIter/3))

    # optimizer2 = optax.adam(learning_rate=5e-4)
    # energy, vec = fit(vec, optimizer2, MaxIter=int(MaxIter/3))

    #v = minimize(cost_func, vec0, method=method, tol=tol, options={"maxiter":MaxIter, "disp": disp}).x
    #energy = cost_func(v)

    return energy, vec

def _expand_hs(h_n, s_n, rmats_n, sdets_n, rmats, sdets, h1e, h2e, ao_ovlp=None):
    '''
    Expand the
    '''
    nvecs = len(rmats_n)
    hm = jnp.copy(h_n)
    sm = jnp.copy(s_n)
    # TODO avoid the following
    hm = hm.at[nvecs:, nvecs:].set(noci.full_hamilt_w_sdets(sdets_n, h1e, h2e, ao_ovlp=ao_ovlp))
    sm = sm.at[nvecs:, nvecs:].set(noci.full_ovlp_w_rotmat(rmats_n))

    # crossing terms
    for i in range(nvecs): # old vectors
        for j in range(nvecs): # new vectors
            _s = slater.ovlp_rotmat(rmats[i], rmats_n[j])
            _h = slater.trans_hamilt(sdets[i], sdets_n[j], h1e, h2e, ao_ovlp=ao_ovlp)
            sm = sm.at[i, nvecs+j].set(_s)
            sm = sm.at[nvecs+j, i].set(_s)
            hm = hm.at[i, nvecs+j].set(_h)
            hm = hm.at[nvecs+j, i].set(_h)

    return hm, sm