'''
Optmizing the rotation matrices.
'''

import numpy as np
from pyscf import gto, scf, fci, cc
import sys
sys.path.append("../")
import helpers, molecules, rbm, noci

nH = 6
a = 1.5

# construct molecule
mol = gto.Mole()
mol.atom = molecules.gen_geom_hchain(nH, a)
mol.unit='angstrom'
mol.basis = "sto3g"
mol.build()

mf = scf.UHF(mol)
norb = mol.nao 

ao_ovlp = mol.intor_symmetric ('int1e_ovlp')

h1e = mf.get_hcore()
h2e = mol.intor('int2e')


# Hartree-Fock
init_guess = mf.get_init_guess()
helpers.make_init_guess(init_guess)
mf.init_guess = init_guess
mf.kernel()
occ = mf.get_occ()
nocc = int(np.sum(occ[0]))
nvir = norb - nocc

# energy
elec_energy = mf.energy_elec()[0]
e_nuc = mf.energy_nuc()
e_hf = mf.energy_tot()
mo_coeff = np.asarray(mf.mo_coeff)

# check UHF
dm = mf.make_rdm1()
# print(dm[0] - dm[1])
diff = np.linalg.norm(dm[0] - dm[1])
if diff < 1e-5:
    print("WARNING: converged to RHF solution.")

# CISE
mycc = cc.UCCSD(mf)
mycc.run()
de = mycc.e_corr
e_cc = e_hf + de

# FCI
myci = fci.FCI(mf)
e_fci, c = myci.kernel()

rot0_u = np.zeros((norb, nocc))
rot0_u[:nocc, :nocc] = np.eye(nocc)
r0 = np.zeros((nvir, nocc))
# add new rotation matrices
n_rot = None
r_singles = noci.gen_thouless_singles(nocc, nvir, max_nt=n_rot, zmax=5, zmin=0.1)
nr = 3
rmats0 = np.random.rand(nr, 2, nvir, nocc)
E, rn = noci.optimize_fed(rmats0, mo_coeff, h1e, h2e, ao_ovlp=ao_ovlp, tol=1e-6, MaxIter=80)
e_noci = E + e_nuc
print(e_hf, e_noci, e_cc, e_fci)
