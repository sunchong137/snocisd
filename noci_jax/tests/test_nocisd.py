import numpy as np
from pyscf import gto, scf, ci, cc
np.set_printoptions(edgeitems=30, linewidth=100000, precision=5)
from noci_jax import nocisd
from noci_jax import slater, select_ci, slater_jax
from noci_jax.misc import pyscf_helper



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
e_corr, civec = myci.kernel()
e_cisd = e_hf + e_corr 


# ccsd
mycc = cc.UCCSD(mf)
mycc.run()
de = mycc.e_corr
e_cc = e_hf + de



def test_get_ci_coeff():

    _, _, c2 = nocisd.ucisd_amplitudes(myci, flatten_c2=True)
    for i in [0,2]:
        assert np.linalg.norm(c2[i]-c2[i].T) < 1e-10 
    # C_aabb is not symmetric


def test_compress():
    dt = 0.2
    tmats, coeffs = nocisd.compress(myci, civec=civec, dt1=dt, dt2=dt, tol2=1e-5)
    nvir, nocc = tmats.shape[2:]
    rmats = slater.tvecs_to_rmats(tmats, nvir, nocc)
    E = slater.noci_energy(rmats, mo_coeff, h1e, h2e, return_mats=False, lc_coeffs=coeffs, e_nuc=e_nuc)
    # print("compress: ", E)
    assert np.allclose(E, e_cisd)

def test_given_mo():
    mymf = scf.UHF(mol)
    mymf.kernel()
    mo1 = mymf.stability()[0]                                                             
    init = mymf.make_rdm1(mo1, mymf.mo_occ)                                                 
    mymf.kernel(init) 
    mo_coeff = mymf.mo_coeff 
    h = np.random.rand(norb, norb)
    h += h.T
    _, u = np.linalg.eigh(h)
    h2 = np.random.rand(norb, norb)
    h2 += h2.T
    _, u2 = np.linalg.eigh(h2)
    mo_coeff[0] = mo_coeff[0]@u
    mo_coeff[1] = mo_coeff[1]@u2
    mymf.mo_coeff = mo_coeff

    e_hf = mymf.energy_tot()
 
    myci_n = ci.UCISD(mymf)
    my_corr, civec = myci_n.kernel()
    my_e = e_hf + my_corr 
    assert my_e >= e_cisd


def test_c2t_doubles_truncate():
    t2 = nocisd.gen_nocid_truncate(mf, nocc, nroots=2, dt=0.1)

def test_gen_nocisd_multiref_hsp():

    m_tol = 1e-5
    r_new, r_fix = nocisd.gen_nocisd_multiref_hsp(mf, nvir, nocc)
    e_hsp = slater_jax.noci_energy_jit(r_fix, mo_coeff, h1e, h2e, e_nuc=e_nuc)
    r_select = select_ci.select_rmats_ovlp(r_fix, r_new, m_tol=m_tol, max_ndets=1000)
    e_snoci = slater_jax.noci_energy_jit(r_select, mo_coeff, h1e, h2e, e_nuc=e_nuc)
    print(e_hsp, e_snoci)
    
test_gen_nocisd_multiref_hsp()