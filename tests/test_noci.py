import numpy as np
import time
import sys
from pyscf import gto, scf, fci
sys.path.append("../")
import noci, slater, molecules

def test_gen_s():
    nvir = 3
    nocc = 2
    max_nt= 4
    singles = noci.gen_thouless_singles(nocc, nvir, max_nt=max_nt)
    l = len(singles)
    s0_ref = np.array([[[ 0.1, 10. ],[ 0.1, 0.1],[ 0.1, 0.1]],
                       [[ 0., 0. ], [ 0., 0. ], [ 0.,  0. ]]])
    s1_ref = np.array([[[ 0., 0. ], [ 0., 0. ], [ 0.,  0. ]], 
                       [[ 0.1, 10. ],[ 0.1, 0.1],[ 0.1, 0.1]]])
    assert l == max_nt*2
    assert np.allclose(singles[0], s0_ref)
    assert np.allclose(singles[1], s1_ref)

def test_gen_d():
    nvir = 3
    nocc = 2
    max_nt = None
    doubles = noci.gen_thouless_doubles(nocc, nvir, max_nt=max_nt)
    l = len(doubles)
    assert l == 2*(int(nvir*(nvir-1)/2) * int(nocc*(nocc-1)/2))
    d0_ref = np.array([[[ 10, 0.1 ], [ 0.1, 10], [ 0.1, 0.1]],
                       [[ 0., 0. ], [ 0., 0. ], [ 0.,  0. ]]])
    d1_ref = np.array([[[ 0., 0. ], [ 0., 0. ], [ 0.,  0. ]], 
                       [[ 10, 0.1 ], [ 0.1, 10], [ 0.1, 0.1]]])
    assert np.allclose(doubles[0], d0_ref)
    assert np.allclose(doubles[1], d1_ref)


def test_solve_lc():

    hmat = np.array([[-3.3930846 ,-0.66344294,-0.66001328,-0.66057444],
                     [-0.66344294,-3.1668585 , 0.01292748,-0.27121202],
                     [-0.66001328, 0.01292748,-3.13750223,-0.04905015],
                     [-0.66057444,-0.27121202,-0.04905015,-2.88950005]])
    smat = np.array([[1.        , 0.19510533, 0.19510533, 0.19510533],
                     [0.19510533, 1.        , 0.03806609, 0.06775383],
                     [0.19510533, 0.03806609, 1.        , 0.03806609],
                     [0.19510533, 0.06775383, 0.03806609, 1.        ]])
 
    e = noci.solve_lc_coeffs(hmat, smat)
    assert np.allclose(e, -3.393145112610386)

def test_gradient():
    nH = 4
    a = 1.5

    # construct molecule
    mol = gto.Mole()
    mol.atom = molecules.gen_geom_hchain(nH, a)
    mol.unit='angstrom'
    mol.basis = "sto3g"
    mol.build()

    mf = scf.UHF(mol)
    norb = mol.nao 
    nocc = nH // 2
    nvir = norb - nocc
    ao_ovlp = mol.intor_symmetric ('int1e_ovlp')

    h1e = mf.get_hcore()
    h2e = mol.intor('int2e')

    # Hartree-Fock
    init_guess = mf.get_init_guess()
    init_guess[0][0,0]=10
    init_guess[1][0,0]=0
    mf.init_guess = init_guess
    mf.kernel()

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


    # FCI
    myci = fci.FCI(mf)
    e_fci, c = myci.kernel()


    # add new rotation matrices
    n_rot = 3
    t_singles = noci.gen_thouless_singles(nocc, nvir, max_nt=n_rot, zmax=5, zmin=0.1)
    tmats = t_singles[:n_rot]

    # add hf 
    t_hf = np.zeros_like(tmats[0])
    tmats = [t_hf] + list(tmats)
    
    idx = 1
    # test numerical gradient
    grad_num = noci.num_grad_two_points(tmats, idx, h1e, h2e, mo_coeff, ao_ovlp=ao_ovlp, delt=0.005, include_hf=True)
    print("numerical: \n", grad_num)
    # analytical gradient
    grad_ana = noci.noci_gradient_one_det(tmats, idx, h1e, h2e, mo_coeff, ao_ovlp=ao_ovlp, include_hf=True)
    print("analytical: \n", grad_ana)

test_gradient()