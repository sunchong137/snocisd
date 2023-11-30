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
import copy
from pyscf import ci
from noci_jax import slater, select_ci, nocisd

def two_layers(mf, nocc, nroots=4, dt=0.1, m_tol=1e-6, cprs_tol=1e-5):
    '''
    Choose important doubles as reference, then do CISD on each.
    Args:
        mf: pyscf scf object.
        nocc: int, number of occupied orbitals.
        nroots: int, number of roots to choose from doubles.
        m_tol: threshold for metric-based selection.
        cprs_tol: compression threshold.
    Returns:
        r_all: rotation matrices for the states.
    '''
    mo_coeff = mf.mo_coeff
    norb = mo_coeff.shape[-1]
    nvir = norb - nocc
    my_ci = ci.UCISD(mf)
    t_hf = np.zeros((1, 2, nvir, nocc))
    r_all = slater.tvecs_to_rmats(t_hf, nvir, nocc)
    # first generate important double excitations as references
    c2 = nocisd.ucisd_amplitudes_doubles(my_ci)
    t2 = nocisd.c2t_doubles_truncate(c2, num_roots=nroots, dt=dt, nvir=nvir, nocc=nocc)
    r2 = slater.tvecs_to_rmats(t2, nvir, nocc)
    r_all, idx1 = select_ci.select_rmats_ovlp(r_all, r2, m_tol=m_tol, return_indices=True)
    t2 = t2[idx1]

    # Then do multi-ref
    t_ref = np.vstack([t_hf, t2])
    r_ref = slater.tvecs_to_rmats(t_ref, nvir, nocc)
    r_cisd = nocisd.gen_nocisd_multiref(t_ref, mf, nvir, nocc, dt, tol2=cprs_tol)
    
    r_all = select_ci.select_rmats_ovlp(r_ref, r_cisd, m_tol=m_tol, return_indices=False)

    return r_all


def gen_nocid_two_layers(mf, nocc, nroots1=4, nroots2=2, dt=0.1):
    '''
    Generate NOCI as following:
    |HF> -> CISD -> choose |mu> with largest coeff -> CISD on |mu> -> choose ...
    '''
    # Generate the first layer
    mo_coeff = mf.mo_coeff
    norb = mo_coeff.shape[-1]
    nvir = norb - nocc
    my_ci = ci.UCISD(mf)

    # first layer
    c2 = nocisd.ucisd_amplitudes_doubles(my_ci)
    t2 = nocisd.c2t_doubles_truncate(c2, num_roots=nroots1, dt=dt, nvir=nvir, nocc=nocc)
    num_layer1 = len(t2)
    t_hf = np.zeros((1, 2, nvir, nocc))
    r_all = slater.tvecs_to_rmats(np.vstack([t_hf, t2]), nvir, nocc) # store all the rmats
    
    U2 = slater.orthonormal_mos(t2)
    mo_2 = np.einsum("sij, nsjk -> nsik", mo_coeff, U2)

    my_mf = copy.copy(mf)
    # do cisd on the first layer
    for i in range(num_layer1):
        my_mf.mo_coeff = mo_2[i]
        my_ci = ci.UCISD(my_mf)
        c2_n = nocisd.ucisd_amplitudes_doubles(my_ci)
        t2_n = nocisd.c2t_doubles_truncate(c2_n, num_roots=nroots2, dt=dt, nvir=nvir, nocc=nocc)
        r = slater.tvecs_to_rmats(t2_n, nvir, nocc)
        r = slater.rotate_rmats(r, U2[i])
        r_all = np.vstack([r_all, r])

    return r_all


def gen_two_layers_w_selection(mf, nocc, nroots1=4, nroots2=2, dt=0.1, m_tol=1e-6):
    '''
    Generate NOCI as following:
    |HF> -> CISD -> choose |mu> with largest coeff -> CISD on |mu> -> choose ...
    '''
    # Generate the first layer
    mo_coeff = mf.mo_coeff
    norb = mo_coeff.shape[-1]
    nvir = norb - nocc
    my_ci = ci.UCISD(mf)
    t_hf = np.zeros((1, 2, nvir, nocc))
    r_all = slater.tvecs_to_rmats(t_hf, nvir, nocc)

    # first layer
    c2 = nocisd.ucisd_amplitudes_doubles(my_ci)
    t2 = nocisd.c2t_doubles_truncate(c2, num_roots=nroots1, dt=dt, nvir=nvir, nocc=nocc)
    r2 = slater.tvecs_to_rmats(t2, nvir, nocc)
    r_all, idx1 = select_ci.select_rmats_ovlp(r_all, r2, m_tol=m_tol, return_indices=True)
    t2 = t2[idx1]
    num_layer1 = len(t2)    
    
    U2 = slater.orthonormal_mos(t2)
    mo_2 = np.einsum("sij, nsjk -> nsik", mo_coeff, U2)

    my_mf = copy.copy(mf)
    # do cisd on the first layer
    for i in range(num_layer1):
        my_mf.mo_coeff = mo_2[i]
        my_ci = ci.UCISD(my_mf)
        c2_n = nocisd.ucisd_amplitudes_doubles(my_ci)
        t2_n = nocisd.c2t_doubles_truncate(c2_n, num_roots=nroots2, dt=dt, nvir=nvir, nocc=nocc)
        r2_n = slater.tvecs_to_rmats(t2_n, nvir, nocc)
        r2_n = slater.rotate_rmats(r2_n, U2[i])
        r_all, idx2 = select_ci.select_rmats_ovlp(r_all, r2_n, m_tol=m_tol, return_indices=True)

    return r_all
