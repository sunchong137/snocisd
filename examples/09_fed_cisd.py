'''
Evaluate the spin correlation of H chain.
'''
import numpy as np
from noci_jax import nocisd, slater, thouless, select_ci, slater_jax, hamiltonians, fed 
from noci_jax.misc import pyscf_helper
from noci_jax import slater_jax
import os, sys


chkdir = "./chkfiles/" # save tmp files and restart files
if not os.path.exists(chkdir):
    os.makedirs(chkdir)

orig_stdout = sys.stdout
nH = 6
bl = 1.2
ndets = 4

mol = hamiltonians.gen_mol_hchain(nH, bl, "sto6g")
mf = pyscf_helper.mf_with_ortho_ao(mol)
pyscf_helper.run_stab_scf_breaksymm(mf)
h1e, h2e, e_nuc = pyscf_helper.get_integrals(mf, ortho_ao=False)
norb, nocc, nvir, mo_coeff = pyscf_helper.get_mos(mf)
e_hf = mf.energy_tot()
nelec = mol.nelectron

niter = 4000
print_step = 1000
try:
    t_all = np.load(chkdir + f"/tvecs_hchain{nH}_{bl}A_ndets{ndets}_fed.npy")
    print("# Loading saved tvecs!")
except:
    print("# Re-evaluating reference SDs with FED!")
    if ndets == 1: # first determinant
        t0 = thouless.gen_init_singles(nocc, nvir, max_nt=1, zmax=2, zmin=0.1).reshape(1, -1)
        e_fed, t_new = fed.optimize_fed(h1e, h2e, mo_coeff, nocc, nvecs=ndets, 
                                init_tvecs=t0, MaxIter=niter, print_step=print_step, e_nuc=e_nuc)
        print("Energy fed: ", e_fed + e_nuc)
        t_new = t_new.reshape(1, 2, nvir, nocc)
        t_all = slater.add_tvec_hf(t_new)
        np.save(chkdir + f"/tvecs_hchain{nH}_{bl}A_ndets{ndets}_fed.npy", t_all)
    else:
        try:
            t_fix = np.load(chkdir + f"/tvecs_hchain{nH}_{bl}A_ndets{ndets-1}_fed.npy")
        except:
            raise NameError(f"Need to run FED with {ndets-1} dets first!")
        t_fix = t_fix.reshape(ndets, 2, nvir, nocc) # include hf
        rmats = slater.tvecs_to_rmats(t_fix, nvir, nocc)
        t_init = thouless.gen_init_singles_onedet(nocc, nvir, ndets-1, zmax=2, zmin=0.1,)
        e_fed, t_new = fed.opt_one_thouless(t_init, rmats, mo_coeff, h1e, h2e, tshape=(nvir, nocc), 
                        hmat=None, smat=None, MaxIter=niter, print_step=print_step, 
                        lrate=1e-2, schedule=False)
        print("Energy fed: ", e_fed + e_nuc)
        t_all = np.vstack([t_fix, t_new])
        np.save(chkdir + f"/tvecs_hchain{nH}_{bl}A_ndets{ndets}_fed.npy", t_all)

dt = 0.1
m_tol = 1e-5

if ndets == 1:
    r_fix = slater.tvecs_to_rmats(t_all, nvir, nocc)
    r_new = nocisd.gen_nocisd_multiref(t_all, mf, nvir, nocc, dt=dt, tol2=1e-5)
    r_select = select_ci.select_rmats_ovlp(r_fix, r_new, m_tol=m_tol, max_ndets=1000)
    np.save(chkdir + f"rselect_hchain{nH}_{bl}A_ndets{ndets}_fed.npy", r_select)
else:
    try:
        r_fix = np.load(chkdir + f"rselect_hchain{nH}_{bl}A_ndets{ndets-1}_fed.npy")
        print(f"# Loading selected NOCISD from first {ndets-1} dets!")
    except:
        raise NameError(f"Need to run NOCISD with {ndets-1} dets first!")
    r_new = nocisd.gen_nocisd_onevec(t_all[-1], mf, nvir, nocc, dt=dt, tol2=1e-5)
    r_select = select_ci.select_rmats_ovlp(r_fix, r_new, m_tol=m_tol, max_ndets=1000)
    np.save(chkdir + f"rselect_hchain{nH}_{bl}A_ndets{ndets}_fed.npy", r_select)


e_snoci, c = slater_jax.noci_energy_vec_jit(r_select, mo_coeff, h1e, h2e, e_nuc=e_nuc)
print("NOCISD Energy: {:0.10f}".format(e_snoci))
