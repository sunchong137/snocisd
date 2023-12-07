import numpy as np
from noci_jax import hamiltonians 
from noci_jax.misc import pyscf_helper
from pyscf import cc

def test_hubbard_hf():
    L = 6
    U = 1
    mf = hamiltonians.gen_scf_hubbard1D(L, U, nelec=L, pbc=False)
    pyscf_helper.run_stab_scf_breaksymm(mf)
    e_hf = mf.energy_tot()
    mycc = cc.UCCSD(mf)
    e_corr, t1, t2 = mycc.kernel()
    de = mycc.e_corr
    e_cc = e_hf + de
    print(e_cc)

    # mf.kernel()
    # mo1 = mf.stability()[0]                                                             
    # init = mf.make_rdm1(mo1, mf.mo_occ)  
    # init[0][0, 0] = 1
    # init[1][0, 0] = 0                                               
    # mf.kernel(init)

def ccsd_w_init_guess():
    L = 6
    mf = hamiltonians.gen_scf_hubbard1D(L, 5, nelec=L, pbc=False)
    pyscf_helper.run_stab_scf_breaksymm(mf)
    e_hf = mf.energy_tot()
    mycc = cc.CCSD(mf)
    e_corr, t1, t2 = mycc.kernel()
    de = mycc.e_corr
    e_cc = e_hf + de

    mf = hamiltonians.gen_scf_hubbard1D(L, 6, nelec=L, pbc=False)
    pyscf_helper.run_stab_scf_breaksymm(mf)
    e_hf = mf.energy_tot()
    mycc = cc.CCSD(mf)
    mycc.level_shift = 0.6
    e_corr, t1, t2 = mycc.ccsd(t1=t1, t2=t2)
    de = mycc.e_corr
    e_cc = e_hf + de

ccsd_w_init_guess()
