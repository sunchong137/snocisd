import numpy as np
import jax, optax
import jax.numpy as jnp
from jax.config import config
config.update("jax_debug_nans", True)
config.update("jax_enable_x64", True)
import rbm

def optimize_fed(h1e, h2e, mo_coeff, nocc, nvecs=None, init_tvecs=None, MaxIter=100):
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

    if init_tvecs is None:
        init_tvecs = np.random.rand(nvecs, -1)
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
        hmat, smat = rbm._expand_hs(hmat0, smat0, r, rmats_new, h1e, h2e, mo_coeff)
        rmats_new = jnp.vstack([rmats_new, r])

    de_fed = E - e_hf  
    print("###SUMMARY: Energy lowered after FED: {}".format(de_fed))

    return E, init_tvecs.reshape(nvecs, -1)


def optimize_sweep(h1e, h2e, mo_coeff, nocc, init_tvecs, MaxIter=100, nsweep=1, E0=None):

    nvecs = len(init_tvecs)
    if nsweep < 1 or nvecs < 2:
        print("Number of sweeps needs to be > 1!")
        print("Number of new determinants needs to be > !")
        print("No sweep performed.")
        if E0 is None:
            rot0_u = jnp.zeros((nvir+nocc, nocc))
            rot0_u = rot0_u.at[:nocc, :nocc].set(jnp.eye(nocc))
            rmats_new = jnp.array([[rot0_u, rot0_u]]) # the HF state
            rmats_n = rbm.tvecs_to_rmats(init_tvecs, nvir, nocc)
            rmats_new = jnp.vstack([rmats_new, rmats_n])
        
            E0 = rbm.rbm_energy(rmats_new, mo_coeff, h1e, h2e)
        return E0, init_tvecs

    mo_coeff = jnp.array(mo_coeff)
    h1e = jnp.array(h1e)
    h2e = jnp.array(h2e)

    norb = h1e.shape[-1]
    nvir = norb - nocc
    tshape = (nvir, nocc)


    rot0_u = jnp.zeros((nvir+nocc, nocc))
    rot0_u = rot0_u.at[:nocc, :nocc].set(jnp.eye(nocc))
    rmats_new = jnp.array([[rot0_u, rot0_u]]) # the HF state
    rmats_n = rbm.tvecs_to_rmats(init_tvecs, nvir, nocc)
    rmats_new = jnp.vstack([rmats_new, rmats_n])
    # Start sweeping
    if E0 is None:
        E0 = rbm.rbm_energy(rmats_new, mo_coeff, h1e, h2e)

    e_hf = E0

    print("Start sweeping...")
    for isw in range(nsweep):
        print("Sweep {}".format(isw+1))
        E_s = E0
        for iter in range(nvecs):
            t0 = init_tvecs[iter]
            rmats_new = jnp.delete(rmats_new, 1, axis=0)
            hmat0, smat0= rbm.rbm_energy(rmats_new, mo_coeff, h1e, h2e, return_mats=True)
            E, t, = opt_one_thouless(t0, rmats_new, mo_coeff, h1e, h2e, tshape, 
                                    hmat=hmat0, smat=smat0, MaxIter=MaxIter)
            de = E - E0
            print("Iter {}: energy lowered {}".format(iter+1, de))
            E0 = E
            init_tvecs = init_tvecs.at[iter].set(jnp.copy(t))
            r = rbm.tvecs_to_rmats(jnp.array([t]), nvir, nocc)
            rmats_new = jnp.vstack([rmats_new, r])
        de_s = E - E_s 
        print("***Energy lowered after Sweep {}: {}".format(isw+1, de_s))


    de_tot = E - e_hf 
    print("###SUMMARY: Total energy lowered after sweeping {}".format(de_tot))

    return E, init_tvecs.reshape(nvecs, -1)

def opt_one_thouless(tvec0, rmats, mo_coeff, h1e, h2e, tshape, hmat=None, smat=None, MaxIter=100):


    # nvecs = len(rmats) + 1
    nvir, nocc = tshape

    if hmat is None:
        hmat, smat = rbm.rbm_energy(rmats, mo_coeff, h1e, h2e, return_mats=True)

    def cost_func(t): 
        _t = jnp.array([t])
        # thouless to rotation
        r_n = rbm.tvecs_to_rmats(_t, nvir, nocc)
        hm, sm = rbm._expand_hs(hmat, smat, r_n, rmats, h1e, h2e, mo_coeff)
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

            if i%500 == 0:
                print(f'step {i}, loss: {loss_value};')

        return loss_value, params

    # TODO figure out learning rate
    optimizer = optax.adam(learning_rate=1e-2)
    energy, vec = fit(init_params, optimizer, MaxIter=int(MaxIter))

    del fit # release memory

    return energy, vec
