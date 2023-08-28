# Copyright 2023 NOCI_Jax developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from noci_jax import reshf
import optax
import jax
import jax.numpy as jnp
from jax.config import config
# config.update("jax_debug_nans", True)
config.update("jax_enable_x64", True)


def rbm_fed(h1e, h2e, mo_coeff, nocc, nvecs, init_params=None, 
            MaxIter=100, print_step=1000, lrate=1e-2, schedule=False):
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
    init_params = jnp.array(init_params)

    rot0_u = jnp.zeros((nvir+nocc, nocc))
    rot0_u = rot0_u.at[:nocc, :nocc].set(jnp.eye(nocc))
    rot_hf = jnp.array([[rot0_u, rot0_u]]) # the HF state

    E0 = reshf.rbm_energy(rot_hf, mo_coeff, h1e, h2e)
    e_hf = E0

    opt_tvecs = jnp.array([np.zeros(2*nvir*nocc)]) # All Thouless vectors

    rmats = reshf.tvecs_to_rmats(opt_tvecs, nvir, nocc)
    hmat, smat = reshf.rbm_energy(rmats, mo_coeff, h1e, h2e, return_mats=True)

    print("Start RBM FED...")
    for iter in range(nvecs):
        print(f"*****Optimizing Determinant {iter+1}*****")
        w0 = init_params[iter]
        e, w = opt_one_rbmvec(w0, opt_tvecs, h1e, h2e, mo_coeff, tshape,
                              hmat=hmat, smat=smat,MaxIter=MaxIter, 
                              print_step=print_step, lrate=lrate, schedule=schedule)
        
        init_params = init_params.at[iter].set(jnp.copy(w))
        de = e - E0
        E0 = e
        print(f"##### Done optimizing determinant {iter+1}, energy lowered {de} #####")
        new_tvecs = reshf.add_vec(w, opt_tvecs) # new Thouless vectors from adding this RBM vector
        opt_tvecs = jnp.vstack([opt_tvecs, new_tvecs])
        # update hmat and smat
        rmats_n = reshf.tvecs_to_rmats(new_tvecs, nvir, nocc)
        hmat, smat = reshf.expand_hs(hmat, smat, rmats_n, rmats, h1e, h2e, mo_coeff)
        rmats = jnp.vstack([rmats, rmats_n])

    print("Total energy lowered: {}".format(e - e_hf))
    return e, init_params


def rbm_sweep(h1e, h2e, mo_coeff, nocc, init_params, E0=None, hiddens=[0,1], 
              nsweep=1, MaxIter=100, print_step=1000, lrate=1e-2, schedule=False):

    nvecs = len(init_params)
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
    
    coeff_hidden = reshf.hiddens_to_coeffs(hiddens, nvecs-1)
    coeff_hidden = jnp.array(coeff_hidden)

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
            new_params = np.delete(init_params, iter, axis=0)
            fixed_vecs = reshf.expand_vecs(new_params, coeff_hidden) 
            E, w = opt_one_rbmvec(w0, fixed_vecs, h1e, h2e, mo_coeff, tshape,
                                hmat=None, smat=None, MaxIter=MaxIter, 
                                print_step=print_step, lrate=lrate, schedule=schedule)
            de = E - E0
            E0 = E
            print("Iter {}: energy lowered {}".format(iter+1, de))
            init_params = init_params.at[iter].set(jnp.copy(w))
      
        de_s = E - E_s
        print("***Energy lowered after Sweep {}: {}".format(isw+1, de_s))
    
    return E, init_params


def opt_one_rbmvec(vec0, tvecs, h1e, h2e, mo_coeff, tshape, 
                   hmat=None, smat=None, MaxIter=100, print_step=1000, 
                   lrate=1e-2, schedule=False):
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
    def cost_func(w):
        tvecs_n = reshf.add_vec(w, tvecs) # newly added Thouless vectors
        rmats_n = reshf.tvecs_to_rmats(tvecs_n, nvir, nocc)
        hm, sm = reshf.expand_hs(hmat, smat, rmats_n, rmats, h1e, h2e, mo_coeff)
        e = reshf.solve_lc_coeffs(hm, sm)
        return e

    init_params = jnp.array(vec0)

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
        energy0, vec = fit(init_params, niter1, lrate)
        print("Reducing Learning rate.")
        energy, vec = fit(vec, niter2, lrate2)
    else:
        energy, vec = fit(init_params, MaxIter, lrate)


    del fit # release memory

    return energy, vec

