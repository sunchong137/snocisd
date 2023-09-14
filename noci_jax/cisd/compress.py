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
Compress the linear combinations of CISD with NOCI.
'''
import numpy as np
from pyscf import ci

def get_cisd_coeffs_uhf(mf):
    '''
    Return the CISD coefficients.
    Returns:
        c1: 3D array
        c2: 
    '''
    myci = ci.UCISD(mf)                                                                    
    _, civec = myci.kernel()
    _, c1, c2 = myci.cisdvec_to_amplitudes(civec)
    nocc, nvir = c1[0].shape

    c1_n = np.transpose(np.array(c1), (0, 2, 1))
    # transpose c2
    c2_n = np.transpose(np.array(c2), (0, 3, 1, 4, 2))
    c2_n = c2_n.reshape(3, nvir*nocc, nvir*nocc)

    return c1_n, c2_n

def cis_to_thou(c1, dt=0.1):
    '''
    Given the CIS coefficients, generate Thouless rotation paramters.
    Only for UHF.
    Args:
        c1: 3D array, size (2, nocc, nvir) amplitudes for singly excited states.
        t: float, a small number for the NOSD expansion approximation
    Returns:
        arrays of Thouless for NOSDs
        coeffs for the NOSDs.
    '''
    t1_p = c1 * dt / 2.
    t1_m = -c1 * dt / 2.

    coeffs = np.array([1./dt, -1./dt])

    return np.array([t1_p, t1_m]), coeffs
    