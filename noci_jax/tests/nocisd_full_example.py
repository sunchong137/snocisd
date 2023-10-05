import numpy as np
from jax import numpy as jnp
from pyscf import gto, scf, ci
from noci_jax import nocisd, slater, pyscf_helper, thouless, opt_res

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
mol.basis = "631g"
mol.build()

# Step 2: Run the UHF calculation
mf = pyscf_helper.uhf_with_ortho_ao(mol) # return mf with orthogonal AO basis
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
    tnew = np.load(save_file) # shape of (ndets, 2, nvir, nocc)
except:
    niter = 8000
    print_step = 1000
    t0 = thouless.gen_init_singles(nocc, nvir, max_nt=ndets, zmax=2, zmin=0.1)

    t0 = t0.reshape(ndets, -1)
    E, tnew = opt_res.optimize_res(h1e, h2e, mo_coeff, nocc, nvecs=ndets, 
                                init_tvecs=t0, MaxIter=niter, print_step=print_step)
    np.save(save_file, tnew)

# Step 5: Get the unitary for each NOCI
U_new = slater.orthonormal_mos(tnew)
U_hf = np.eye(norb)
U_hf = np.array([U_hf, U_hf])
U_all = np.vstack([U_hf[None,:], U_new])
mo_all = jnp.einsum("sij, nsjk -> nsik", mo_coeff, U_all)



# Step 6: Run CISD and get the NOCI expansion


# Step 7: Evalaute the energy.

# myci = ci.UCISD(mf)
# e_corr, civec = myci.kernel()
# e_cisd = e_hf + e_corr 
