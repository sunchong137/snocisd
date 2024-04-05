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
from jax import config
# config.update("jax_debug_nans", True)
config.update("jax_enable_x64", True)
from noci_jax import slater_jax


def optimize_res(h1e, h2e, mo_coeff, nocc, nvecs=None, init_tvecs=None, 
                 MaxIter=5000, print_step=1000, lrate=1e-2, schedule=False, 
                 e_nuc=0.0):
    ''' 
    Given a set of Thouless rotations, optimize the parameters.
    Res HF approach, all parameters are optimized simultaneously.
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
        optimized Thouless matrices
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

    # print information 
    print("#"*40)
    print("# Resonate Hartree-Fock Optimization")
    print("# Number of Determinants to optimize: {}".format(nvecs))
    
        
    init_tvecs = init_tvecs.flatten(order="C")

    # first construct the HF state
    rot_hf = slater_jax.gen_rmat_hf(nvir, nocc)
    E0 = slater_jax.noci_energy(rot_hf, mo_coeff, h1e, h2e, return_mats=False)

    def cost_func(t):
        tvecs = t.reshape(nvecs, -1)
        rmats = slater_jax.tvecs_to_rmats(tvecs, nvir, nocc)
        rmats = jnp.vstack([rot_hf, rmats])
        e = slater_jax.noci_energy(rmats, mo_coeff, h1e, h2e, return_mats=False)
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
    
    energy += e_nuc 
    print("########### End optimization ###########")
    print("# Final energy: {:1.8f}".format(energy))
    print("#"*40)

    return energy, vecs.reshape(nvecs, 2, nvir, nocc)

