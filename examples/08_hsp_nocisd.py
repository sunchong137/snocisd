import numpy as np
from pyscf import gto, scf, fci, cc, ci
from noci_jax import nocisd, slater, thouless, select_ci
from noci_jax import slater_jax, hamiltonians, reshf
from noci_jax.misc import pyscf_helper
import os 

save_dir = "./results/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

U = 8
pbc = False
ndets = 3
if pbc:
    save_name = f"hsp_noci_ndets{ndets}_hub1d_U{U}_pbc.txt"
else:
    save_name = f"hsp_noci_ndets{ndets}_hub1d_U{U}_obc.txt"

with open(save_dir + save_name, "w") as infile:
    infile.write(f"# 1D Hubbard model U = {U}, PBC = {pbc}\n")
    infile.write(f"# nsite      resHF         resHF+CISD\n")
    num_sites = [4, 6, 8, 10, 12, 14, 16, 18, 20]
    num_sites = [6]
    for nsite in num_sites:
        nelec = nsite
        mf = hamiltonians.gen_scf_hubbard1D(nsite, U, nelec=nelec, pbc=pbc, spin=1)
        pyscf_helper.run_stab_scf_breaksymm(mf)

        h1e, h2e, e_nuc = pyscf_helper.get_integrals(mf, ortho_ao=False)
        norb, nocc, nvir, mo_coeff = pyscf_helper.get_mos(mf)
        e_hf = mf.energy_tot()
        nelec = mf.mol.nelectron


        m_tol = 1e-5
        r_new, r_fix = nocisd.gen_nocisd_multiref_hsp(mf, nvir, nocc)
        e_hsp = slater_jax.noci_energy_jit(r_fix, mo_coeff, h1e, h2e, e_nuc=e_nuc)
        r_select = select_ci.select_rmats_ovlp(r_fix, r_new, m_tol=m_tol, max_ndets=1000)
        e_snoci = slater_jax.noci_energy_jit(r_select, mo_coeff, h1e, h2e, e_nuc=e_nuc)
 
        print("e_snoci: {}".format(e_snoci/nsite))
        # infile.write("{:2d}     {:.8f}      {:.8f}\n".format(nsite, e_reshf/nsite, e_snoci/nsite))


