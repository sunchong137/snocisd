import numpy as np
from pyscf import gto, scf, ci, fci
from pyscf.lib import numpy_helper
import scipy
np.set_printoptions(edgeitems=30, linewidth=100000, precision=5)
from noci_jax import nocisd
from noci_jax import slater, pyscf_helper
import logging

# System set up
nH = 4
bl = 1.5
geom = []
for i in range(nH):
    geom.append(['H', 0.0, 0.0, i*bl])

# construct molecule
mol = gto.Mole()
mol.atom = geom
mol.unit='angstrom'
mol.basis = "631g"
mol.build()

mf = scf.UHF(mol)
mf.kernel()
mo1 = mf.stability()[0]                                                             
init = mf.make_rdm1(mo1, mf.mo_occ)                                                 
mf.kernel(init) 

h1e, h2e, e_nuc = pyscf_helper.get_integrals(mf, ortho_ao=False)
norb, nocc, nvir, mo_coeff = pyscf_helper.get_mos(mf)
e_hf = mf.energy_tot()
nelec = mol.nelectron
myci = ci.UCISD(mf) 


def test_get_ci_coeff():

    _, _, c2 = nocisd.ucisd_amplitudes(myci, flatten_c2=True)
    for i in [0,2]:
        assert np.linalg.norm(c2[i]-c2[i].T) < 1e-10 
    # C_aabb is not symmetric

def test_singles_c2t():
    _, c1, _ = nocisd.ucisd_amplitudes(myci)
    nvir, nocc = c1[0].shape
    dt = 0.1
    tmats = nocisd.c2t_singles(c1, dt)
    t0 = np.zeros((2, nvir, nocc))
    t_all = np.vstack(t0 + tmats)
    rmats = slater.tvecs_to_rmats(t_all, nvir, nocc)
    # ovlp = slater.metric_rmats(rmats[0], rmats[1])
    
    E = slater.noci_energy(rmats, mo_coeff, h1e, h2e, return_mats=False, lc_coeffs=None, e_nuc=e_nuc)
    assert np.allclose(E, e_hf)

def test_doubles_c2t():

    # occ_ref = np.random.rand(nocc, nocc)
    # occ_ref += occ_ref.T
    occ_ref = None
    _, c1, c2 = nocisd.ucisd_amplitudes(myci)
    dt = 0.1
    t1= nocisd.c2t_singles(c1, dt)
    _t2, lams = nocisd.c2t_doubles(c2, dt=dt, tol=8e-2)
    t2 = np.vstack(_t2)
    t_hf = np.zeros((1, 2, nvir, nocc))
    tmats = t2
    # tmats = np.vstack([t1, t2])
    tmats = np.vstack([t_hf, tmats])

    rmats = slater.tvecs_to_rmats(tmats, nvir, nocc, occ_mat=occ_ref)
    # ovlp = slater.metric_rmats(rmats[0], rmats[1])


    h, s = slater.noci_energy(rmats, mo_coeff, h1e, h2e, return_mats=True, lc_coeffs=None, e_nuc=e_nuc)
    print(np.linalg.det(s))
    E = slater.noci_energy(rmats, mo_coeff, h1e, h2e, return_mats=False, lc_coeffs=None, e_nuc=e_nuc)
    print(E)


def test_compress():
    dt = 0.2
    tmats, coeffs = nocisd.compress(myci, dt1=dt, dt2=dt, tol2=1e-5)
    nvir, nocc = tmats.shape[2:]
    rmats = slater.tvecs_to_rmats(tmats, nvir, nocc)
    E = slater.noci_energy(rmats, mo_coeff, h1e, h2e, return_mats=False, lc_coeffs=coeffs, e_nuc=e_nuc)
    print("compress: ", E)



# def test_c2_symm():
#     norm = np.linalg.norm
#     # myci = ci.UCISD(mf)
#     # _, civec = myci.kernel()
#     c0, c1, c2 = nocisd.ucisd_amplitudes(mf, flatten_c2=False)
#     print(c2[0][0,1,0,1])
#     # print(np.diag(c2[1]))
#     # print(norm(c2[0]-c2[0].T.conj()))
#     # print(norm(c2[1]-c2[1].T.conj()))
#     # print(norm(c2[2]-c2[2].T.conj()))

def test_cisd_vecs():
    myci = ci.UCISD(mf)  
    e_corr, civec = myci.kernel()
    c0, c1, c2 = myci.cisdvec_to_amplitudes(civec)
    ov = nocc * nvir
    l0 = 1
    l1 =  ov * 2
    l2 = ov**2 + ov*(nocc-1)*(nvir-1)//2
    # print(civec)
    print(len(civec))

# def ci_singles():
#     dt = 0.00001
#     vec = np.copy(civec)
#     vec[loc[2]:] = 0
#     vec /= np.linalg.norm(vec)
#     e_ci = pyscf_helper.cisd_energy_from_vec(vec, mf)
#     print(e_ci)
#     c0, c1, c2 = nocisd.ucisd_amplitudes(mf, civec=vec)
#     t1 = nocisd.c2t_singles(c1, dt=dt)
#     coeffs = np.array([c0] + [1/dt, -1/dt]*2)
#     coeffs /= np.linalg.norm(coeffs)
#     t = slater.add_tvec_hf(t1)
#     rmats = slater.tvecs_to_rmats(t, nvir, nocc)
#     E = slater.noci_energy(rmats, mo_coeff, h1e, h2e, return_mats=False, lc_coeffs=coeffs, e_nuc=e_nuc)
#     print(E)

# def ci_doubles_aa():
#     dt = 0.001
#     vec = np.copy(civec)
#     vec[loc[0]:loc[3]] = 0
#     # vec[loc[4]:] = 0
#     # print(vec)
#     vec /= np.linalg.norm(vec)
#     e_ci = pyscf_helper.cisd_energy_from_vec(vec, mf)
#     print("CISD energy:", e_ci)
#     c0, c1, c2 = nocisd.ucisd_amplitudes(mf, civec=vec)
#     t2, lam2s = nocisd.c2t_doubles(c2, dt=dt)
#     t2_aa = t2[0]
#     t2_bb = t2[2]

#     c2 = np.concatenate([lam2s[0],]*2 + [lam2s[2],]*2)/(dt**2)
    
#     coeffs = np.concatenate([[c0], c2])
#     t2_all = np.vstack([t2_aa, t2_bb])
#     t_all = slater.add_tvec_hf(t2_all)
#     rmats = slater.tvecs_to_rmats(t_all, nvir, nocc)
#     E = slater.noci_energy(rmats, mo_coeff, h1e, h2e, return_mats=False, lc_coeffs=coeffs, e_nuc=e_nuc)
#     print("CISD energy:", e_ci)
#     print("compressed NOCI",E)


# def ci_doubles_ab():
#     # TODO not figured out
#     dt = 1
#     vec = np.copy(civec)
#     vec[loc[0]:loc[2]] = 0
#     vec[loc[3]:] = 0
#     vec /= np.linalg.norm(vec)
#     e_ci = pyscf_helper.cisd_energy_from_vec(vec, mf)

#     c0, c1, c2 = nocisd.ucisd_amplitudes(mf, civec=vec)
#     t2, lam2s = nocisd.c2t_doubles(c2, dt=dt)
#     tab = np.asarray(t2[1])
#     # print(tab)
#     cab = np.concatenate([lam2s[1],]*2 + [-lam2s[1],]*2)/(dt**2)
#     t = slater.add_tvec_hf(tab)
#     # t = tab
#     rmats = slater.tvecs_to_rmats(t, nvir, nocc)
#     coeffs = np.concatenate([[c0], cab])
#     # coeffs = cab
#     coeffs /= np.linalg.norm(coeffs)
#     # H, S = slater.noci_energy(rmats, mo_coeff, h1e, h2e, return_mats=True, lc_coeffs=coeffs, e_nuc=e_nuc)
#     # print(np.diag(H)+e_nuc)
#     # print(S)
#     # exit()
#     E = slater.noci_energy(rmats, mo_coeff, h1e, h2e, return_mats=False, lc_coeffs=coeffs, e_nuc=e_nuc)
#     print(e_ci)
#     print(E)


