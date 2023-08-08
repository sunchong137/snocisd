'''
Optmizing the rotation matrices.
'''

import numpy as np
from pyscf import gto, scf
import sys
sys.path.append("../../")
import rbm
import optnoci_all as optdets


def gen_geom_hchain(n, bond=0.8):
    # generate geometry for hydrogen chain
    # H2 bond length = 0.74 Angstrom
    geom = []
    for i in range(n):
        geom.append(['H', .0, .0, i*bond])
    return geom


nH = 10
bl = 2.0

# construct molecule
mol = gto.Mole()
mol.atom = gen_geom_hchain(nH, bl)
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
init_guess[0][0,0] = 10
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


# generate initial guess for thouless rotations
n_dets = 4
niter = 10000
nsave = 1
print_step = 20
t0 = rbm.gen_thouless_singles(nocc, nvir, max_nt=n_dets, zmax=10, zmin=0.1)[:n_dets]

t0 = t0.reshape(n_dets, -1)
# save_file = f"results/hchain{nH}_{bl}_{mol.basis}_tvecs_ndets{n_dets}.npy"

for i in range(nsave): 
    E, t_new = optdets.optimize_res(h1e, h2e, mo_coeff, nocc, nvecs=n_dets,
                            init_tvecs=t0, MaxIter=niter, print_step=print_step)
    # tsave.append(t_new.reshape(n_dets, 2, nvir, nocc))
    t0 = np.copy(t_new.reshape(n_dets, -1))

# np.save(save_file, tsave)

