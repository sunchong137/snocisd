# Copyright 2023-2024 NOCI_Jax developers. All Rights Reserved.
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
from noci_jax.jw import hartreefock 


def test_jw_hams():
    norb = 6
    Sz = 0
    nelec = int(norb/2 + Sz + 1e-10)
    
    # create a random determinant 
    frand = np.random.rand(norb, norb)
    frand += frand.T 
    M = frand 
    V = frand 
    w = np.random.rand(norb)
    h = hartreefock.jw_ham(M, V)
    print(h)
test_jw_hams(); exit()
def test_energy():
    norb = 6
    Sz = 0
    nelec = int(norb/2 + Sz + 1e-10)
    

    # create a random determinant 
    frand = np.random.rand(norb, norb)
    frand += frand.T 
    _e, _v = np.linalg.eigh(frand)
    det = _v[:, :nelec]
    M = frand 
    V = frand 

    e = hartreefock.eval_energy(det, M, V)

test_energy()