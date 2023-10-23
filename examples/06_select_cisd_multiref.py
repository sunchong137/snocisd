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
from jax import numpy as jnp
from pyscf import gto, scf, ci
from noci_jax import nocisd, slater, pyscf_helper, thouless, opt_res
import copy

from jax.config import config
config.update("jax_enable_x64", True)

# Step 1: Set up the system
nH = 4
bl = 1.5
geom = []
for i in range(nH):
    geom.append(['H', 0.0, 0.0, i*bl])

mol = gto.Mole()
mol.atom = geom
mol.unit='angstrom'
mol.basis = "sto3g"
mol.build()

# Step 2: Run the UHF calculation
# mf = pyscf_helper.uhf_with_ortho_ao(mol) # return mf with orthogonal AO basis
mf = scf.UHF(mol)
mf.kernel()
mo1 = mf.stability()[0]                                                             
init = mf.make_rdm1(mo1, mf.mo_occ)                                                 
mf.kernel(init) 

# Step 3: Get attributes needed for the NOCI calculations
h1e, h2e, e_nuc = pyscf_helper.get_integrals(mf, ortho_ao=False)
norb, nocc, nvir, mo_coeff = pyscf_helper.get_mos(mf)
e_hf = mf.energy_tot()
nelec = mol.nelectron


# Step 4: NOCI with res HF
ndets = 1
save_file = "data/h{}_R{}_{}_ndet{}.npy".format(nH, bl, mol.basis, ndets)
try:
    # raise ValueError
    tnew = np.load(save_file) # shape of (ndets, 2, nvir, nocc)
except:
    niter = 5000
    print_step = 1000
    t0 = thouless.gen_init_singles(nocc, nvir, max_nt=ndets, zmax=2, zmin=0.1)

    t0 = t0.reshape(ndets, -1)
    E, tnew = opt_res.optimize_res(h1e, h2e, mo_coeff, nocc, nvecs=ndets, 
                                init_tvecs=t0, MaxIter=niter, print_step=print_step, e_nuc=e_nuc)
    np.save(save_file, tnew)

t_all = slater.add_tvec_hf(tnew)
r_all = slater.tvecs_to_rmats(t_all, nvir, nocc)
r_new = nocisd.gen_nocisd_multiref(t_all, mf, nvir, nocc, dt=0.1, tol2=1e-5)



