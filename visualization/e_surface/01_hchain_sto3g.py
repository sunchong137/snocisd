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
n_dets = 2
niter = 1000
print_step = 1000
t0 = rbm.gen_thouless_singles(nocc, nvir, max_nt=n_dets, zmax=10, zmin=0.1)[:n_dets]
t0 = t0.reshape(n_dets, -1)
save_file = f"hchain{nH}_{bl}_{mol.basis}_tvecs_ndets{n_dets}.npy"

try:
    t_new = np.load(save_file)
except:
    E, t_new = optdets.optimize_res(h1e, h2e, mo_coeff, nocc, nvecs=n_dets,
                                init_tvecs=t0, MaxIter=niter, print_step=print_step)

    t_new = t_new.reshape(n_dets, -1)
    np.save(save_file, t_new)


def cost_func(tvecs):
    rmats = rbm.tvecs_to_rmats(tvecs, nvir, nocc)
    e = rbm.rbm_energy(rmats, mo_coeff, h1e, h2e, return_mats=False)
    return e

# change two attributs to plot the energy surface
xmin1 = -50
xmax1 = 50
xmin2 = -50
xmax2 = 50
npoint1 = 25 #(xmax1-xmin1)*2
npoint2 = 25 #(xmax2-xmin2)*2
d1s = np.linspace(xmin1, xmax1, npoint1)
d2s = np.linspace(xmin2, xmax2, npoint2)

# print(t_new)
# exit()
# STO3g:

# two minima  0: 4
#             1: 1, 2, 5, 8, 9
# good combos: 0:2 + 0:6,7,11,12

res = []
a = 0
i = 9
b = 1
j = 0
for d1 in d1s:
    for d2 in d2s:
        tvec = np.copy(t_new)
        tvec[a, i] += d1
        tvec[b, j] += d2
        E = cost_func(tvec)
       # print(d1, d2, E)
        res.append([d1, d2, E])
res = np.asarray(res)
np.save(f"results/hchain{nH}_{bl}_{mol.basis}_esurf.npy", res)


