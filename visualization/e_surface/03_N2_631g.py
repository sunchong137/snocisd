'''
Example from Carlos' paper J. Chem. Phys. 139, 204102 (2013)
RHF: -108.9547
CCSD: -109.2740
CCSD(T): -109.2863
'''

from pyscf import gto, scf, cc, ci
import numpy as np
import sys 
sys.path.append("../..")
import rbm
import optnoci_all as optdets

# set up the system with pyscf
bl = 1.5
mol = gto.Mole()
mol.atom = '''
N   0   0   0
N   0   0   {}
'''.format(bl)
mol.unit = "angstrom"
mol.basis = "sto3g"
mol.cart = True
mol.build()

# Mean-field calculation
mf = scf.UHF(mol)
mf.conv_tol = 1e-10
# break symmetry
init_guess = mf.get_init_guess()
init_guess[0][0, 0] = 2
mf.kernel(init_guess)
e_hf = mf.energy_tot()
print("UHF: ", e_hf)


# # CISD                                                                                
# myci = ci.CISD(mf)                                                                    
# myci.run()                                                                            
# e_ci = myci.e_tot                                                                     
# print("CISD: ", e_ci) 
# exit()

# CCSD 
# mycc = cc.CCSD(mf).run()  
# e_ccsd = mycc.e_tot
# print("CCSD: ", e_ccsd)
# exit()
# # CCSD(T)
# et = mycc.ccsd_t()
# e_ccsdt = e_ccsd + et
# print("CCSD(T): ", e_ccsdt)

# NOCI res HF

# First get values that will be used for NOCI
norb = mol.nao # number of orbitals
occ = mf.get_occ()
nocc = int(np.sum(occ[0])) # number of occupied orbitals for spin up
nvir = norb - nocc
# ao_ovlp = mol.intor_symmetric ('int1e_ovlp') # overlap matrix of AO orbitals
h1e = mf.get_hcore()
h2e = mol.intor('int2e')
mo_coeff = np.asarray(mf.mo_coeff)
e_nuc = mf.energy_nuc()
# nvir = 3
# nocc = 7


# generate initial guess for thouless rotations
n_dets = 2 # 10 is better than CISD
niter = 5000
print_step = 1000
t0 = rbm.gen_thouless_singles(nocc, nvir, max_nt=n_dets, zmax=10, zmin=0.1)[:n_dets]
t0 = t0.reshape(n_dets, -1)
# RES HF
save_file = f"data/N2{bl}_{mol.basis}_tvecs_ndets{n_dets}.npy"

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
npoint1 = 80 #(xmax1-xmin1)*2
npoint2 = 80 #(xmax2-xmin2)*2
d1s = np.linspace(xmin1, xmax1, npoint1)
d2s = np.linspace(xmin2, xmax2, npoint2)

# sto3g 
# 2 minima: 0: 6, 20
# 
# print(t_new)
# exit()
res = []
a = 0
i = 6
b = 0
j = 20
for d1 in d1s:
    for d2 in d2s:
        tvec = np.copy(t_new)
        tvec[a, i] += d1
        tvec[b, j] += d2
        E = cost_func(tvec)
        # print(d1, d2, E)
        res.append([d1, d2, E])
res = np.asarray(res)
np.save(f"results/N2{bl}_{mol.basis}_esurf.npy", res)




