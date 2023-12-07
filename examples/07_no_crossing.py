import numpy as np
from pyscf import gto, scf
from noci_jax.misc import pyscf_helper

# set up the system with pyscf
bls = np.linspace(1, 4, 40)
res = []
for b in bls:
    mol = gto.Mole()
    mol.atom = '''
    Li   0   0   0
    F    0   0   {}
    '''.format(b)
    mol.unit = "angstrom"
    mol.basis = "631g**"
    mol.spin = 0
    mol.build()
    # Mean-field calculation
    mf = scf.UHF(mol)
    mf.kernel()
    # pyscf_helper.run_stab_scf(mf)
    e_sing = mf.energy_tot()

    mol2 = gto.Mole()
    mol2.atom = '''
    Li   0   0   0
    F    0   0   {}
    '''.format(b)
    mol2.unit = "angstrom"
    mol2.basis = "631g**"
    mol2.spin = 2
    mol2.build()
    # Mean-field calculation
    mf2 = scf.UHF(mol2)
    mf2.kernel()
    # pyscf_helper.run_stab_scf(mf2)
    e_trip = mf2.energy_tot()
    res.append([b, e_sing, e_trip])

res = np.asarray(res)
np.savetxt("data/no_cross_lif.txt", res)


