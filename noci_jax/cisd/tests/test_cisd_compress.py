import numpy as np
from pyscf import gto, scf, ci, fci
from pyscf.lib import numpy_helper
import scipy
np.set_printoptions(edgeitems=30, linewidth=100000, precision=3)
from noci_jax.cisd import compress


def test_get_ci_coeff():
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
    mol.basis = "sto3g"
    mol.build()

    mf = scf.UHF(mol)
    mf.kernel()
    mo1 = mf.stability()[0]                                                             
    init = mf.make_rdm1(mo1, mf.mo_occ)                                                 
    mf.kernel(init) 
    c1, c2 = compress.get_cisd_coeffs_uhf(mf)
    for i in range(3):
        assert np.linalg.norm(c2[i]-c2[i].T) < 1e-6  # change to a larger number if basis larger
    
test_get_ci_coeff()