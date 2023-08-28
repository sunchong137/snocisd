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

import optax
import jax
import jax.numpy as jnp
from noci_jax import reshf
from jax.config import config
# config.update("jax_debug_nans", True)
config.update("jax_enable_x64", True)

def rbm_all(h1e, h2e, mo_coeff, nocc, nvecs, init_params=None, bias=None, hiddens=[0,1],
            MaxIter=1000, print_step=1000, lrate=1e-2, truncate=None, schedule=False):
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
    coeff_hidden = reshf.hiddens_to_coeffs(hiddens, nvecs, order=truncate)
    coeff_hidden = jnp.array(coeff_hidden)

    if init_params is None:
        init_params = jnp.random.rand(nvecs, lt)

    init_params = init_params.flatten(order='C')
    len_params = len(init_params)

    def cost_func_no_bias(w):
        w_n = w.reshape(nvecs, -1)
        rmats = reshf.params_to_rmats(w_n, nvir, nocc, coeff_hidden)
        e = reshf.rbm_energy(rmats, mo_coeff, h1e, h2e)
        return e
    
    def cost_func_bias(v):
        w_n = jnp.copy(v[:len_params]).reshape(nvecs, -1)
        b_n = jnp.copy(v[len_params:])
        rmats = reshf.params_to_rmats(w_n, nvir, nocc, coeff_hidden)
        lc_coeffs = jnp.exp(coeff_hidden.dot(b_n)) 
        e = reshf.rbm_energy(rmats, mo_coeff, h1e, h2e, lc_coeffs=lc_coeffs)
        return e
    
    if bias is None:
        cost_func = cost_func_no_bias 
        params0 = init_params
    else:
        bias = jnp.array(bias)
        cost_func = cost_func_bias
        params0 = jnp.concatenate([init_params, bias])

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

    return energy, vecs