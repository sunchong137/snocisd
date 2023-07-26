'''
Optmizing the rotation matrices.
'''

import numpy as np
from pyscf import gto, scf, fci, ci
import sys
sys.path.append("../")
import  noci, molecules, optdets

def test_hchain_fed():
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
    e_nuc = mf.energy_nuc()
    e_hf = mf.energy_tot()
    mo_coeff = np.asarray(mf.mo_coeff)


    # FCI
    myci = fci.FCI(mf)
    e_fci, c = myci.kernel()

    rot0_u = np.zeros((norb, nocc))
    rot0_u[:nocc, :nocc] = np.eye(nocc)
    # add new rotation matrices
    n_rot = None
    r_singles = noci.gen_thouless_singles(nocc, nvir, max_nt=n_rot, zmax=5, zmin=0.1)
    rmats = r_singles[:4]
    #rmats.append([r_singles[2], rot0_u])
    #rmats.append([rot0_u, r_singles[2]])

    E, rn = optdets.optimize_fed(rmats, mo_coeff, h1e, h2e, ao_ovlp=ao_ovlp, tol=5e-4, MaxIter=100)
    e_noci = E + e_nuc
    print("Energy: ", e_noci)
    assert e_noci <= e_hf
    assert e_noci >= e_fci

test_hchain_fed()

def test_hchain_res():
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
    n_rot = 4
    r_singles = noci.gen_thouless_singles(nocc, nvir, max_nt=n_rot, zmax=5, zmin=0.1)
    rmats = r_singles[:4]

    E, rn = optdets.optimize_res(rmats, mo_coeff, h1e, h2e, ao_ovlp=ao_ovlp, tol=1e-5, MaxIter=5)

    e_noci = E + e_nuc
    print("Energy: ", e_noci)
    assert e_noci <= e_hf
    assert e_noci >= e_fci


def test_hchain_res_pyscf():
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
    n_rot = 4
    t_singles = noci.gen_thouless_singles(nocc, nvir, max_nt=n_rot, zmax=5, zmin=0.1)
    tmats = t_singles[:n_rot]

    E, rn = optdets.optimize_res(tmats, mf=mf, tol=1e-4, MaxIter=1)

    e_noci = E + e_nuc
    print("Energy: ", e_noci)
    assert e_noci <= e_hf
    assert e_noci >= e_fci

