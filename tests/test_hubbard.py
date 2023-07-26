import numpy as np
import sys
sys.path.append("../")
import molecules
from pyscf import gto, scf, ao2mo, fci

def test_mf_rhf():
    norb = 6
    U = 4
    spin = 0
    mymf = molecules.hubbard_mf(norb, U, spin=spin)
    e = mymf.energy_elec()[0]
    mo = mymf.mo_coeff 

    # pyscf example
    mymol = gto.M() 
    n = 6
    mymol.nelectron = n

    mf = scf.RHF(mymol)
    h1 = np.zeros((n,n))
    for i in range(n-1):
        h1[i,i+1] = h1[i+1,i] = -1.0
    h1[n-1,0] = h1[0,n-1] = -1.0  # PBC
    eri = np.zeros((n,n,n,n))
    for i in range(n):
        eri[i,i,i,i] = 4.0

    mf.get_hcore = lambda *args: h1
    mf.get_ovlp = lambda *args: np.eye(n)
    # ao2mo.restore(8, eri, n) to get 8-fold permutation symmetry of the integrals
    # ._eri only supports the two-electron integrals in 4-fold or 8-fold symmetry.
    mf._eri = ao2mo.restore(8, eri, n)

    mf.kernel()

    e_hf = mf.energy_elec()[0]
    mo_coeff = mf.mo_coeff

    assert np.allclose(e, e_hf)
    assert np.allclose(mo, mo_coeff)
    
def test_uhf():
    norb = 6
    U = 6
    spin = 1
    mymf = molecules.hubbard_mf(norb, U, spin=spin)
    e = mymf.energy_elec()[0]
    mo = mymf.mo_coeff 

    # pyscf example
    mymol = gto.M() 
    n = 6
    mymol.nelectron = n
    nelec = n

    mf = scf.UHF(mymol)
    h1 = np.zeros((n,n))
    for i in range(n-1):
        h1[i,i+1] = h1[i+1,i] = -1.0
    h1[n-1,0] = h1[0,n-1] = -1.0  # PBC
    eri = np.zeros((n,n,n,n))
    for i in range(n):
        eri[i,i,i,i] = U

    mf.get_hcore = lambda *args: h1
    mf.get_ovlp = lambda *args: np.eye(n)
    # ao2mo.restore(8, eri, n) to get 8-fold permutation symmetry of the integrals
    # ._eri only supports the two-electron integrals in 4-fold or 8-fold symmetry.
    mf._eri = ao2mo.restore(8, eri, n)

    # break symmetry
    init_guess = mf.get_init_guess()
    np.fill_diagonal(init_guess[0], 1)
    np.fill_diagonal(init_guess[1], 0)

    mf.init_guess = init_guess
    mf.kernel()

    e_hf = mf.energy_elec()[0]
    mo_coeff = mf.mo_coeff

    print(mo[0] - mo[1])
    print(mo_coeff[0] - mo_coeff[1])


    assert np.allclose(e, e_hf)
    #assert np.allclose(mo, mo_coeff)
    
test_uhf()
def test_fci():
    norb = 6
    U = 4
    spin = 0
    mymf = molecules.hubbard_mf(norb, U, spin=spin)
    e_fci, ci = molecules.hubbard_fci(mymf)
    
    # compare to pyscf 
    n = 6
    h1 = np.zeros((n,n))
    for i in range(n-1):
        h1[i,i+1] = h1[i+1,i] = -1.0
    h1[n-1,0] = h1[0,n-1] = -1.0  # PBC
    eri = np.zeros((n,n,n,n))
    for i in range(n):
        eri[i,i,i,i] = 4.0
    myci = fci.direct_spin0
    e, c = myci.kernel(h1, eri, n, n)
    # c and ci are not the same because they are based on different 
    assert np.allclose(e, e_fci)
