# Copyright 2023 NOCI_JAX developers. All Rights Reserved.
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
from scipy import linalg as sla
from pyscf import lo


def get_C_ortho(mol, method='lowdin'):
    '''
    Evaluate the transformation matrix.
    Args:
        mol: PySCF gto.Mole() object.
    Kwargs:
        method: string, lowdin or schmidt.
    Returns:
        2D array, transformation matrix from AO to OAO (orthogonal AO)
    '''
    ao_ovlp = mol.intor_symmetric('int1e_ovlp')
    if method == 'lowdin':
        return C_ortho_lowdin(ao_ovlp) 
    elif method == 'schmidt':
        return C_ortho_schmidt(ao_ovlp)
    else:
        print("Warning: method can only be 'lowdin' or 'schmidt'! Using Lowdin method.")
        return C_ortho_lowdin(ao_ovlp) 

def C_ortho_lowdin(ao_ovlp):
    '''
    Evaluate the transform matrix via directly multiply S^{-1/2}.

    '''
    e, v = np.linalg.eigh(ao_ovlp)
    idx = e > 1e-15 # avoid singularity
    return np.dot(v[:, idx]/np.sqrt(e[idx]), v[:, idx].conj().T)

def C_ortho_schmidt(ao_ovlp):
    '''
    Schmidt orthogonalization.
    '''
    c = np.linalg.cholesky(ao_ovlp)
    return sla.solve_triangular(c, np.eye(c.shape[1]), lower=True, overwrite_b=False).conj().T