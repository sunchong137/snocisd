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
'''
Evaluate the convergence of DMRG with reprect to the bond dimension.
'''
import numpy as np
from noci_jax.misc import hubbard_dmrg, hubbard_ba
import os

res_dir = "./results"
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

def scan_bd_hub(U, e_tol=1e-5, pbc=True):
    '''
    Scan through number of sites and bond dimension.
    '''
    nsites = np.arange(6, 22, 2, dtype=int)
    min_bdim = 40
    max_bdim = 600
    bdims = np.arange(min_bdim, max_bdim, 80, dtype=int)
    if pbc is True:
        fname = res_dir + "/hubbard1d_scan_U{}_pbc.txt".format(U)
    else:
        fname = res_dir + "/hubbard1d_scan_U{}_obc.txt".format(U)
    with open(fname, "w") as fin:
        fin.write("# DMRG for 1D Hubbard model, U = {}, PBC = {}\n".format(U, pbc))
        fin.write("# Rows: number of sites; Columns: maximum bond dimension\n")
        # fin.write("0        BA")
        for bd in bdims:
            fin.write("         {}".format(bd))
        fin.write("\n")
        for n in nsites:
            ne_a = n//2
            E0 = 100
            fin.write("{:2d}".format(n))
            # E_ba = hubbard_ba.lieb_wu(n, ne_a, ne_a, U)
            # fin.write("  {:1.6f}".format(E_ba))
            for bd in bdims:
                E = hubbard_dmrg.hubbard1d_dmrg(n, U, (ne_a, ne_a), pbc=pbc, init_bdim=20, 
                                                max_bdim=bd, nsweeps=15)
                E /= n
                if abs(E - E0) < e_tol:
                    fin.write("   {:1.6f}".format(E))
                    break 
                else:
                    fin.write("   {:1.6f}".format(E))
                    E0 = E 
            
            fin.write("\n")


scan_bd_hub(2)