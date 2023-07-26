'''
Functions related to specific molecules.
'''
import numpy as np
from pyscf import gto, scf, ao2mo, fci

def gen_geom_hchain(n, bond=0.8):
    # generate geometry for hydrogen chain
    # H2 bond length = 0.74 Angstrom
    geom = []
    for i in range(n):
        geom.append(['H', .0, .0, i*bond])
    return geom

def hubbard_mf(norb, U, spin=0, nelec=None, pbc=True):
    '''
    Mean-field of Hubbard model.
    Args:
        norb: int, number of sites
        U: positive float, Hubbard U
    Kwargs:
        spin: 0 (RHF) or 1 (UHF)
        pbc: boolean, periodic boundary condition
    Returns:
        float: HF energy 
        numpy array of size (nspin, norb, norb), the MO coefficients.
    '''
    # Norb need to be 4n + 2, otherwise there is degeneracy between HOMO and LUMO
    mol = gto.M()
    if nelec is None:
        mol.nelectron = norb

    h1e = np.zeros((norb, norb))
    eri = np.zeros((norb,)*4)

    for i in range(norb-1):
        h1e[i, (i+1)] = -1.
        h1e[(i+1), i] = -1.
        eri[i,i,i,i] = U
    eri[-1,-1,-1,-1] = U
    if pbc:
        assert norb%4 == 2, "PBC requires Norb = 4n+2!"
        h1e[0, -1] = -1.
        h1e[-1, 0] = -1.

    if spin == 0:
        mf = scf.RHF(mol)
    elif spin == 1:
        mf = scf.UHF(mol)
    else:
        raise ValueError("Spin has to be 0 or 1!")


    
    mf.get_hcore = lambda *args: h1e 
    mf.get_ovlp = lambda *args: np.eye(norb)
    mf._eri = ao2mo.restore(8, eri, norb)
    mol.incore_anyway = True

    
    if spin == 1:
        # because there is degeneracy in the orbitals, different init_guess will give different mo_coeffs
        # but the energy is the same.
        init_guess = mf.get_init_guess()
        np.fill_diagonal(init_guess[0], 1)  # Jujube typed this: ';/'
        np.fill_diagonal(init_guess[1], 0)
        mf.init_guess = init_guess
    # mo_coeff = mf.mo_coeff

    mf.kernel()
    #e_hf = mf.energy_elec
    return mf

def hubbard_fci(mf):
    cisolver = fci.FCI(mf)
    ci_energy, ci = cisolver.kernel()
    
    return ci_energy, ci

