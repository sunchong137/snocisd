# Copyright 2023 by NOCI_Jax developers. All Rights Reserved.
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

import jax.numpy as jnp
import numpy as np
from scipy import linalg as sla
from noci_jax import reshf


def test_solve_lc():
    n = 10
    hmat = np.random.rand(n, n)
    hmat = hmat + hmat.T 
    smat = np.random.rand(n, n) * 0.1 
    smat = smat + smat.T 
    smat += np.eye(n)

    # use scipy
    e0, v0 = sla.eigh(hmat, b=smat)
    esla = e0[0] 
    vsla = v0[:, 0]

    hmat = jnp.array(hmat)
    smat = jnp.array(smat)

    e, v = reshf.solve_lc_coeffs(hmat, smat, return_vec=True)
    v = np.array(v)

    h = v.conj().T.dot(hmat).dot(v)
    s = v.conj().T.dot(smat).dot(v)
    e2 = h/s
    assert np.allclose(e, esla)
    assert np.allclose(e, e2)
    assert np.allclose(v, vsla)


