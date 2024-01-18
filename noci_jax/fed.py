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
import jax, optax
import jax.numpy as jnp
from jax.config import config
config.update("jax_debug_nans", True)
config.update("jax_enable_x64", True)
from noci_jax import slater_jax

def optimize_fed(h1e, h2e, mo_coeff, nocc, nvecs=None, init_tvecs=None, 
                 MaxIter=100, print_step=1000, lrate=1e-2, schedule=False,
                 e_nuc=0.0):
    '''
    Given a set of Thouless rotations, optimize the parameters.
    Using FED (few-determinant) approach.
    Args:
        h1e: 2D array, one-body Hamiltonian 
        h2e: 4D array, two-body Hamiltonian
        mo_coeff: a list of two 2D arrays
        nocc: number of occupied orbitals
    Kwargs:
        nvecs: number of determinants
        init_vecs: 2D array of size (nvecs, -1)
        MaxIter: maximum number of iterations
        print_step: when to print the progress
        lrate: learning rate
        schedule: whether to change learning rate
        e_nuc: the nuclear energy
    Returns:
        energy
        optimized Thouless
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

    rmats_new = slater_jax.gen_rmat_hf(nvir, nocc)
    hmat, smat = slater_jax.noci_energy(rmats_new, mo_coeff, h1e, h2e, return_mats=True)
    e_hf = slater_jax.solve_lc_coeffs(hmat, smat)
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
                                hmat=hmat0, smat=smat0, MaxIter=MaxIter, 
                                print_step=print_step, lrate=lrate, schedule=schedule)
        de = E - E0
        print("Iter {}: energy lowered {}".format(iter+1, de))
        E0 = E
        init_tvecs = init_tvecs.at[iter].set(jnp.copy(t))
        r = slater_jax.tvecs_to_rmats(t, nvir, nocc) 
        hmat, smat = slater_jax.expand_hs(hmat0, smat0, r, rmats_new, h1e, h2e, mo_coeff)
        rmats_new = jnp.vstack([rmats_new, r])

    de_fed = E - e_hf  
    print("###SUMMARY: Energy lowered after FED: {}".format(de_fed))

    return E + e_nuc, init_tvecs.reshape(nvecs, -1)


def optimize_sweep(h1e, h2e, mo_coeff, nocc, init_tvecs, MaxIter=100, nsweep=1, E0=None, 
                   print_step=1000, lrate=1e-2, schedule=False):
    '''
    Sweep from left to right to further optimize the parameters.
    '''
    nvecs = len(init_tvecs)
    if nsweep < 1 or nvecs < 2:
        print("Number of sweeps needs to be > 1!")
        print("Number of new determinants needs to be > !")
        print("No sweep performed.")
        if E0 is None:
            rmats_new = slater_jax.gen_rmat_hf(nvir, nocc)
            rmats_n = slater_jax.tvecs_to_rmats(init_tvecs, nvir, nocc)
            rmats_new = jnp.vstack([rmats_new, rmats_n])
        
            E0 = slater_jax.noci_energy(rmats_new, mo_coeff, h1e, h2e)
        return E0, init_tvecs

    mo_coeff = jnp.array(mo_coeff)
    h1e = jnp.array(h1e)
    h2e = jnp.array(h2e)

    norb = h1e.shape[-1]
    nvir = norb - nocc
    tshape = (nvir, nocc)

    rmats_new = slater_jax.gen_rmat_hf(nvir, nocc)
    rmats_n = slater_jax.tvecs_to_rmats(init_tvecs, nvir, nocc)
    rmats_new = jnp.vstack([rmats_new, rmats_n])
    # Start sweeping
    if E0 is None:
        E0 = slater_jax.noci_energy(rmats_new, mo_coeff, h1e, h2e)

    e_hf = E0

    print("Start sweeping...")
    for isw in range(nsweep):
        print("Sweep {}".format(isw+1))
        E_s = E0
        for iter in range(nvecs):
            t0 = init_tvecs[iter]
            rmats_new = jnp.delete(rmats_new, 1, axis=0)
            hmat0, smat0= slater_jax.noci_energy(rmats_new, mo_coeff, h1e, h2e, return_mats=True)
            E, t, = opt_one_thouless(t0, rmats_new, mo_coeff, h1e, h2e, tshape, 
                                    hmat=hmat0, smat=smat0, MaxIter=MaxIter, 
                                    print_step=print_step, lrate=lrate, schedule=schedule)
            de = E - E0
            print("Iter {}: energy lowered {}".format(iter+1, de))
            E0 = E
            init_tvecs = init_tvecs.at[iter].set(jnp.copy(t))
            r = slater_jax.tvecs_to_rmats(jnp.array([t]), nvir, nocc)
            rmats_new = jnp.vstack([rmats_new, r])
        de_s = E - E_s 
        print("***Energy lowered after Sweep {}: {}".format(isw+1, de_s))


    de_tot = E - e_hf 
    print("###SUMMARY: Total energy lowered after sweeping {}".format(de_tot))

    return E, init_tvecs.reshape(nvecs, -1)

def opt_one_thouless(tvec0, rmats, mo_coeff, h1e, h2e, tshape, 
                     hmat=None, smat=None, MaxIter=100, print_step=1000, 
                     lrate=1e-2, schedule=False):

    '''
    Optimize one Thouless matrices while fixing the rest.
    '''

    # nvecs = len(rmats) + 1
    nvir, nocc = tshape

    if hmat is None:
        hmat, smat = slater_jax.noci_energy(rmats, mo_coeff, h1e, h2e, return_mats=True)
    def cost_func(t): 
        # thouless to rotation
        r_n = slater_jax.tvecs_to_rmats(t, nvir, nocc)
        hm, sm = slater_jax.expand_hs(hmat, smat, r_n, rmats, h1e, h2e, mo_coeff)
        energy = slater_jax.solve_lc_coeffs(hm, sm)
        return energy  
    init_params = jnp.array(tvec0)

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
        energy0, vecs = fit(init_params, niter1, lrate)
        # print(f"Energy lowered: {energy0 - E0}")
        print("Reducing the learning rate.")
        energy, vecs = fit(vecs, niter2, lrate2)
        # print(f"Energy lowered: {energy - energy0}")
    else:
        energy, vecs = fit(init_params, MaxIter, lrate)
    del fit # release memory
    return energy, vecs
