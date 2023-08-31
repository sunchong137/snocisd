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

'''
Compute order parameters.
'''

import numpy as np
import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)

def corr_spin_state(rmats, mo_coeff, lc_coeff):
    '''
    Evaluate spin correlation:
    C_ij = <n_iup n_jdn> - <n_iup><n_jdn>
    '''
    # first calculate metric and thus overlap
    metrics_all = jnp.einsum('nsji, msjk -> nmsik', rmats.conj(), rmats)
    smat = jnp.prod(jnp.linalg.det(metrics_all), axis=-1)
    inv_metrics = jnp.linalg.inv(metrics_all)
    sdets = jnp.einsum("sij, nsjk -> nsik", mo_coeff, rmats)

    # evaluate <n_iup> and <n_jdn>
    tdm1s_diag = jnp.einsum("msij, nmsjk, nsik -> nmsi", sdets, inv_metrics, sdets.conj())
    dm1_diag = jnp.einsum("nmsi, nm -> nmsi", tdm1s_diag, smat)
    dm1_diag = jnp.einsum("n, m, nmsi -> si", lc_coeff.conj(), lc_coeff, dm1_diag) 

    # evaluate <n_iup n_jdn>
    tdm_diag_u = tdm1s_diag[:, :, 0]
    tdm_diag_d = tdm1s_diag[:, :, 1]
    dm2_ud_diag = jnp.einsum("nmi, nmj -> nmij", tdm_diag_u, tdm_diag_d)
    dm2_ud_diag = jnp.einsum("n, m, nmij -> ij", lc_coeff.conj(), lc_coeff, dm2_ud_diag) 
    phi_norm = jnp.einsum("n, m, nm ->", lc_coeff.conj(), lc_coeff, smat)

    dm1_diag = dm1_diag / phi_norm 
    dm2_ud_diag = dm2_ud_diag / phi_norm

    # evaluate correlation
    c_spin = dm2_ud_diag - jnp.einsum("i, j -> ij", dm1_diag[0], dm1_diag[1])
    return c_spin


def corr_spin_dms(dm1s, dm2_ud):
    
    n_u = jnp.diag(dm1s[0])
    n_d = jnp.diag(dm1s[1])
    nn_ud = jnp.einsum("iijj->ij", dm2_ud)

    return nn_ud - jnp.einsum("i, j -> ij", n_u, n_d)