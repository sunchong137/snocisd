# Copyright 2023 NOCI_JAX developers. All Rights Reserved.
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
Includes:
1. Bethe Ansatz for 1D Hubbard model.
2. DMRG for 1D and 2D Hubbard model.

'''
import numpy as np
from scipy.optimize import root as find_root
import numpy as np
from pyblock2.driver.core import DMRGDriver, SymmetryTypes

Pi = np.pi

def lieb_wu(L, Nup, Ndn, U, MaxIter=200, MaxIterL=100, MaxIterK=100, 
            e_tol=1e-10, L_tol=1e-8, k_tol=1e-8):
    '''
    Bethe Ansatz Kernel.
    Adapted from the Mathematical code by Carlos Jimenez-Hoyos.
    Following Lieb&Wu's paper: https://arxiv.org/abs/cond-mat/0207529   
    '''
    Iarr, Jarr, karr, Larr = gen_ba_grids(L, Nup, Ndn)
    ThF = lambda x: -2 * np.arctan(2*x / U)
    Ne = Nup + Ndn
    E0 = 0
    for it in range(MaxIter):
        Lsav = np.zeros(Nup)
        ksav = np.zeros(Ne)
        # Loop for Larr
        for itL in range(MaxIterL):
            for k in range(Nup):
                x0 = Larr[k]
                def eq_to_solve(x):
                    left = -np.sum(ThF(2*x-2*np.sin(karr)))
                    right = -np.sum(ThF(x-Larr)) + ThF(x-Larr[k]) + 2 * Pi * Jarr[k]
                    return left - right
                Lsav[k] = find_root(eq_to_solve, x0=x0).x[0]
          
            err = np.linalg.norm(Lsav - Larr)
            if err < L_tol:
                break 
            Larr = Lsav.copy()

        # iteration over karr
        for itK in range(MaxIterK):
            for k in range(Ne):
                x0 = karr[k] 
                def eq_to_solve(x):
                    left = L * x
                    right = 2 * Pi * Iarr[k] + np.sum(ThF(2*np.sin(x) - 2*Larr[:Nup]))
                    return left - right
                ksav[k] = find_root(eq_to_solve, x0=x0).x[0]
            err = np.linalg.norm(ksav - karr)
            if err < k_tol:
                break 
            karr = ksav

        # evaluate energy
        E = -2 * np.sum(np.cos(karr))
        if abs(E - E0) < e_tol:
            print("Bethe Ansatz converged in {} iterations.".format(it+1))
            break
        E0 = E
    return E

# Construct arrays
def gen_ba_grids(L, Nup, Ndn):
    '''
    Construct grid points needed in the integration.
    '''
    Ne = Nup + Ndn
    assert Ne <= L * 2
    # parity
    p_up = Nup % 2
    p_dn = Ndn % 2
    p_ne = Ne % 2
    # Iarr
    if (p_up + p_ne) % 2 == 0: # same parity
        Iarr = np.arange(-Ne/2, Ne/2, 1)
    else:
        Iarr = np.arange(-(Ne-1)/2, (Ne+1)/2, 1)

    # Jarr 
    if (p_up + p_dn) % 2 == 0:
        Jarr = np.arange(-(Nup-1)/2, (Nup+1)/2, 1)
    else:
        Jarr = np.arange(-Nup/2-1, Nup/2+1, 1)
    
    # karr
    coeff = 2 * Pi / L
    kup = np.arange(1, Nup+1, 1)
    kdn = np.arange(1, Ndn+1, 1)
    kup = (kup // 2) * (-1)**kup * coeff
    kdn = (kdn // 2) * (-1)**kdn * coeff 
    karr = np.concatenate([kup, kdn])
    karr.sort()

    # init guess of Larr TODO maybe not in this function
    Larr = (2*Pi/Nup) * np.arange(-(Nup-1)/2, (Nup+1)/2, 1)
    return Iarr, Jarr, karr, Larr 

def hubbard1d_dmrg(nsite, U, nelec=None, pbc=False, return_mps=False, filling=1.0, init_bdim=50, 
                   max_bdim=200, nsweeps=8, cutoff=1e-8, max_noise=1e-5):
    # set system
    if nelec is None:
        nelec = int(nsite * filling + 1e-10)
        if abs(nelec/nsite - filling) > 1e-5:
            print("WARNING: The filling is changed to {:1.2f}".format(nelec/nsite))
        if nelec % 2 == 0:
            spin = 0
        else:
            spin = 1
    else:
        try:
            neleca, nelecb = nelec
            spin = abs(neleca - nelecb)
            nelec = neleca + nelecb 
        except:
            spin = 0  
    
    
    driver = DMRGDriver(scratch="./tmp", symm_type=SymmetryTypes.SZ, n_threads=4)
    driver.initialize_system(n_sites=nsite, n_elec=nelec, spin=spin)

    # build Hamiltonian
    # c - creation spin up, d - annihilation spin up
    # C - creation spin dn, D - annihilation spin dn
    ham_str = driver.expr_builder() 
    ham_str.add_term("cd", np.array([[[i, i + 1], [i + 1, i]] for i in range(nsite - 1)]).flatten(), -1)
    ham_str.add_term("CD", np.array([[[i, i + 1], [i + 1, i]] for i in range(nsite - 1)]).flatten(), -1)
    if pbc:
        ham_str.add_term("cd", np.array([[nsite-1, 0], [0, nsite-1]]).flatten(), -1)
        ham_str.add_term("CD", np.array([[nsite-1, 0], [0, nsite-1]]).flatten(), -1)

    ham_str.add_term("cdCD", np.array([[i, ] * 4 for i in range(nsite)]).flatten(), U)
    ham_mpo = driver.get_mpo(ham_str.finalize(), iprint=0)

    # Schedule, using the linearly growing bond dimonsion
    bdims = list(np.linspace(init_bdim, max_bdim, nsweeps//2, endpoint=True, dtype=int))
    if max_noise < 1e-16:
        noises = [0.0]
    else:
        noises = list(np.logspace(np.log(max_noise), -16, nsweeps//2, endpoint=True)) + [0.0]

    ket = driver.get_random_mps(tag="KET", bond_dim=init_bdim, nroots=1)
    thrds = [1e-10] * nsweeps
    energy = driver.dmrg(ham_mpo, ket, n_sweeps=nsweeps, bond_dims=bdims, noises=noises,
             thrds=thrds, cutoff=cutoff, iprint=0)

    print('DMRG total energy = {:2.6f}, energy per site = {:2.6f}'.format(energy, energy/nsite))
    if return_mps:
        return energy, ket, ham_mpo
    else:
        return energy
    

def hubbard2d_dmrg(nx, ny, U, nelec=None, pbc=False, filling=1.0, init_bdim=50, 
                   max_bdim=600, nsweeps=10, cutoff=1e-8, max_noise=1e-5, return_pdms=False):
    
    nsite = nx * ny
    # set system
    if nelec is None:
        nelec = int(nsite * filling + 1e-10)
        if abs(nelec/nsite - filling) > 1e-5:
            print("WARNING: The filling is changed to {:1.2f}".format(nelec/nsite))
        if nelec % 2 == 0:
            spin = 0
        else:
            spin = 1
    else:
        try:
            neleca, nelecb = nelec
            spin = abs(neleca - nelecb)
            nelec = neleca + nelecb 
        except:
            spin = 0  
    
    driver = DMRGDriver(scratch="./chkdir", symm_type=SymmetryTypes.SZ, n_threads=4)
    driver.initialize_system(n_sites=nsite, n_elec=nelec, spin=spin)

    # build Hamiltonian
    # c - creation spin up, d - annihilation spin up
    # C - creation spin dn, D - annihilation spin dn
    ham_str = driver.expr_builder() 
    connected_pairs = []
    for i in range(ny-1):
        for j in range(nx-1):
            idx = i*nx + j 
            idx_r = i*nx + j + 1
            idx_d = (i+1)*nx + j
            connected_pairs += [idx, idx_r, idx_r, idx, idx, idx_d, idx_d, idx]
        connected_pairs += [(i+1)*nx-1, (i+2)*nx-1, (i+2)*nx-1, (i+1)*nx-1]

    dn = (ny-1) * nx
    for j in range(nx-1):
        connected_pairs += [dn+j, dn+j+1, dn+j+1, dn+j]

    if pbc:
        # down-up
        for i in range(nx):
            connected_pairs += [i, dn+i, dn+i, i]
        # right-left
        for j in range(ny):
            connected_pairs += [nx*j, nx*(j+1)-1, nx*(j+1)-1, nx*j]
            
    connected_pairs = np.asarray(connected_pairs)
    ham_str.add_term("cd", connected_pairs, -1)
    ham_str.add_term("CD", connected_pairs, -1)

    ham_str.add_term("cdCD", np.array([[i, ] * 4 for i in range(nsite)]).flatten(), U)
    ham_mpo = driver.get_mpo(ham_str.finalize(), iprint=0)

    # Schedule, using the linearly growing bond dimonsion
    bdims = list(np.linspace(init_bdim, max_bdim, nsweeps//2, endpoint=True, dtype=int))
    if max_noise < 1e-16:
        noises = [0.0]
    else:
        noises = list(np.logspace(np.log(max_noise), -16, nsweeps//2, endpoint=True)) + [0.0]

    ket = driver.get_random_mps(tag="KET", bond_dim=init_bdim, nroots=1)
    thrds = [1e-10] * nsweeps
    energy = driver.dmrg(ham_mpo, ket, n_sweeps=nsweeps, bond_dims=bdims, noises=noises,
             thrds=thrds, cutoff=cutoff, iprint=0)
    print('DMRG total energy = {:2.6f}, energy per site = {:2.6f}'.format(energy, energy/nsite))
    if return_pdms:
        pdm1 = driver.get_1pdm(ket)
        pdm2 = driver.get_2pdm(ket)#.transpose(0, 3, 1, 2)
        for i in range(len(pdm2)):
            pdm2[i] = pdm2[i].transpose(0, 3, 1, 2)
        return energy, ket, pdm1, pdm2
    return energy, ket 
