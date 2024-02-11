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
Hamiltonians used in this project.
'''
from pyscf import gto, scf, ao2mo 
import numpy as np
import logging

def gen_mol_n2(bl=1.2, basis="sto3g", cartesian=False):
    '''
    Nitrogen molecule.
    '''
    mol = gto.Mole()
    mol.atom = '''
    N   0   0   0
    N   0   0   {}
    '''.format(bl)
    mol.unit = "angstrom"
    mol.basis = basis
    mol.cart = cartesian
    mol.build()
    return mol


def gen_mol_LiF(bl=1.5, basis="sto3g", cartesian=False):
    '''
    Lithium Fluoride.
    '''
    mol = gto.Mole()
    mol.atom = '''
    Li   0   0   0
    F   0   0   {}
    '''.format(bl)
    mol.unit = "angstrom"
    mol.basis = basis
    mol.cart = cartesian
    mol.build()
    return mol   


def gen_mol_hchain(n_atom, bl=1.1, basis="sto3g", cartesian=False):
    '''
    1D Hydrogen chain.
    '''
    print("########## Hydrogen chain ##########")
    print(f"# NH = {n_atom}; bond = {bl}A; basis={basis}")
    print("#----------------------------------#")
    geom = []
    for i in range(n_atom):
        geom.append(['H', 0.0, 0.0, i*bl])
    mol = gto.Mole()
    mol.atom = geom
    mol.unit ='angstrom'
    mol.basis = basis
    mol.cart = cartesian
    mol.build()
    return mol


def gen_mol_hlatt(nx, ny, bl=1.1, basis="sto3g", cartesian=False):
    '''
    2D Hydrogen Lattice.
    '''
    geom = []
    for i in range(nx):
        for j in range(ny):
            geom.append(['H', i*bl, j*bl, 0.0])
    mol = gto.Mole()
    mol.atom = geom
    mol.unit ='angstrom'
    mol.basis = basis
    mol.cart = cartesian
    mol.build()
    return mol


def gen_scf_hubbard1D(nsite, U, nelec=None, pbc=True, filling=None, spin=1):
    '''
    1D Hubbard model.
    Returns:
        PySCF scf object for Hubbard model.
    '''

    mol = gto.M()
    if nelec is None:
        nelec = int(nsite * filling + 1e-10)
        if abs(nelec - nsite * filling) > 1e-2:
            logging.warning("Changing filling from {:0.2f} to {:0.2f} to keep integer number of electrons!".format(filling, nelec/nsite))
    # print info
    print("#####################################")
    print("### One-dimensional Hubbard model ###")
    print(f"### Number of sites: {nsite}")
    print(f"### Number of electrons: {nelec}")
    print(f"### U = {U}, PBC = {pbc}")
    print("#####################################")

    mol.nelectron = nelec
    mol.nao = nsite
    
    h1e = np.zeros((nsite, nsite))
    for i in range(nsite-1):
        h1e[i, i+1] = -1
        h1e[i+1, i] = -1

    eri = np.zeros((nsite, )*4)
    for i in range(nsite):
        eri[i,i,i,i] = U
    if pbc:
        h1e[0, -1] = h1e[-1, 0] = -1

    if spin == 1:
        mf = scf.UHF(mol)
    elif spin == 0:
        mf = scf.RHF(mol)
    else:
        raise ValueError("Spin can only be 0 or 1!")
    
    mf.get_hcore = lambda *args: h1e 
    mf.get_ovlp = lambda *args: np.eye(nsite)
    mf._eri = ao2mo.restore(8, eri, nsite)
    mol.incore_anyway = True
    return mf
    

def gen_scf_hubbard2D(nx, ny, U, nelec=None, pbc=True, filling=None, spin=1):
    '''
    2D Hubbard model.
      nx
     _ _ _
    |_|_|_|
    |_|_|_| ny
    |_|_|_|
    site index: top to bottom then left to right. 
    Returns:
        PySCF scf object for Hubbard model.
    '''
    mol = gto.M()
    nsite = nx * ny
    if nelec is None:
        nelec = int(nsite * filling + 1e-10)
        if abs(nelec - nsite * filling) > 1e-2:
            logging.warning("Changing filling from {:0.2f} to {:0.2f} to keep integer number of electrons!".format(filling, nelec/nsite))

    # print info
    print("#####################################")
    print("### Two-dimensional Hubbard model ###")
    print(f"### Number of sites: {nx} x {ny}")
    print(f"### Number of electrons: {nelec}")
    print(f"### U = {U}, PBC = {pbc}")
    print("#####################################")
    mol.nelectron = nelec
    mol.nao = nsite
    mol.incore_anyway = True
    # define Hamiltonians
    if nx == 1:
        return gen_scf_hubbard1D(ny, U, nelec=nelec, pbc=pbc, filling=filling)
    if ny == 1:
        return gen_scf_hubbard1D(nx, U, nelec=nelec, pbc=pbc, filling=filling)

    nsite = nx * ny
    h1e = np.zeros((nsite, nsite))
    for i in range(ny-1):
        for j in range(nx-1):
            idx = i*nx + j 
            idx_r = i*nx + j + 1
            idx_d = (i+1)*nx + j
            h1e[idx, idx_r] = h1e[idx_r, idx] = -1
            h1e[idx, idx_d] = h1e[idx_d, idx] = -1
        h1e[(i+1)*nx-1, (i+2)*nx-1] = h1e[(i+2)*nx-1, (i+1)*nx-1] = -1 # last column

    dn = (ny-1) * nx
    for j in range(nx-1):
        h1e[dn+j, dn+j+1] = h1e[dn+j+1, dn+j] = -1

    if pbc:
        # down-up
        for i in range(nx):
            h1e[i, dn+i] = h1e[dn+i, i] = -1
        # right-left
        for j in range(ny):
            h1e[nx*j, nx*(j+1)-1] = h1e[nx*(j+1)-1, nx*j] = -1
            
    eri = np.zeros((nsite, )*4)
    for i in range(nsite):
        eri[i,i,i,i] = U

    if spin == 1:
        mf = scf.UHF(mol)
    elif spin == 0:
        mf = scf.RHF(mol)
    else:
        raise ValueError("Spin can only be 0 or 1!")
    
    mf.get_hcore = lambda *args: h1e 
    mf.get_ovlp = lambda *args: np.eye(nsite)
    mf._eri = ao2mo.restore(8, eri, nsite)
    
    return mf
