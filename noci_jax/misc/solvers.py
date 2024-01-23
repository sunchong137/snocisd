# Copyright 2023 NOCI_Jax developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
'''
Interface for solvers.
'''
from pyscf import scf, fci, ao2mo 
try:
    from pyblock2._pyscf.ao2mo import integrals as itg
    from pyblock2.driver.core import DMRGDriver, SymmetryTypes
except:
    print("Warning: Block2 is not implemented!")

def run_shci(mol, max_cycle=100, tol=1e-8):
    '''
    Interface to the selected CI in PySCF.
    We use the RHF instance because at this accuracy,
    RHF or UHF will not be very different, and RHF
    is cheaper.

    Args:
        mol: the PySCF datatype that stores the 
        information of the molecule.
    '''

    mf = scf.RHF(mol) 
    mf.kernel()
    norb = mol.nao
    nelec = mol.nelectron
    mo_coeff = mf.mo_coeff
    h1e = mf.get_hcore()
    eri = mf._eri
    h1e_mo = mo_coeff.T @ h1e @ mo_coeff
    eri_mo = ao2mo.kernel(eri, mo_coeff, compact=False).reshape(norb, norb, norb, norb)
    e_nuc = mf.energy_nuc()
    scisolver = fci.SCI()
    scisolver.max_cycle = max_cycle
    scisolver.conv_tol = tol
    e, civec = scisolver.kernel(h1e_mo, eri_mo, norb, nelec)

    return e + e_nuc, civec


def run_block2(mf, spin_symm=True, init_bdim=20, max_bdim=100, nsweeps=10, max_noise=1e-4,
               scratch_dir='./tmp', thr=1e-8, return_pdms=False, clean_scratch=True):
    '''
    Run DMRG calculation with block2.
    Args:
        mf: pyscf RHF object, converged mf;
    Kwargs:
    '''
    if spin_symm:
        ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = itg.get_rhf_integrals(mf,
                                        ncore=0, ncas=None, g2e_symm=8)
        symm = SymmetryTypes.SU2
    else:
        ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = itg.get_uhf_integrals(mf,
                                        ncore=0, ncas=None, g2e_symm=8)
        symm = SymmetryTypes.SZ
        
    driver = DMRGDriver(scratch=scratch_dir, symm_type=symm, n_threads=4)
    driver.initialize_system(n_sites=ncas, n_elec=n_elec, spin=spin, orb_sym=orb_sym)

    mpo = driver.get_qc_mpo(h1e=h1e, g2e=g2e, ecore=ecore, iprint=1)
    ket = driver.get_random_mps(tag="GS", bond_dim=init_bdim, nroots=1)

    # schedule 
    hsweep = nsweeps // 2
    bond_dims = [init_bdim] * (hsweep//2) + [max_bdim] * (hsweep//2)
    noises = [max_noise] * (hsweep//2) + [max_noise*1e-1] * (hsweep//2) + [0]
    thrds = [thr] * hsweep

    energy = driver.dmrg(mpo, ket, n_sweeps=nsweeps, bond_dims=bond_dims, noises=noises,
        thrds=thrds, iprint=1)
    print('DMRG energy = %20.15f' % energy)

    
    if return_pdms:
        pdm1 = driver.get_1pdm(ket)
        pdm2 = driver.get_2pdm(ket)#.transpose(0, 3, 1, 2)
        if spin_symm:
            pdm2 = pdm2.transpose(0, 3, 1, 2)
        else:
            for i in range(len(pdm2)):
                pdm2[i] = pdm2[i].transpose(0, 3, 1, 2)
        if clean_scratch:
            import shutil
            shutil.rmtree(scratch_dir)   
        return energy, ket, pdm1, pdm2 
    else:
        if clean_scratch:
            import shutil
            shutil.rmtree(scratch_dir)   
        return energy, ket
