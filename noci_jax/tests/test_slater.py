# Copyright 2023 by NOCI_Jax developers. All Rights Reserved.
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

import numpy as np
from scipy import linalg as sla
from noci_jax import slater, nocisd 
from noci_jax.misc import pyscf_helper, basis_transform
from pyscf import gto, ci, ao2mo

mol = gto.Mole()
mol.atom = "H 0 0 0; H 0 0 1.2; H 0 0 2.4; H 0 0 3.6"
mol.unit='angstrom'
mol.basis = "sto6g"
mol.build()
mf = pyscf_helper.mf_with_ortho_ao(mol, spin_symm=False)
mf.kernel(verbose=0, tol=1e-10)

def test_tvecs_to_rmats():
    nocc = 2
    nvir = 3
    occ_mat = np.eye(2)/2
    occ_mat[0,1] = -0.1
    occ_mat += occ_mat.T
    tvecs = np.arange(nocc*nvir*2)
    rmats = slater.tvecs_to_rmats(tvecs, nvir, nocc, occ_mat=occ_mat)
    ref = np.array([[[[ 1.,  -0.1], [-0.1,  1. ], [ 0.,1. ],[ 2., 3. ], [ 4.,5. ]],
                     [[ 1.,-0.1],[-0.1,  1. ],[ 6., 7. ],[ 8., 9. ],[10.,  11. ]]]])
    assert np.allclose(rmats, ref)

def test_solve_lc():
    n = 10
    hmat = np.random.rand(n, n)
    hmat = hmat + hmat.T 
    smat = np.random.rand(n, n) * 0.1 
    smat = smat + smat.T 
    smat += np.eye(n)
    # use scipy
    e0, v0 = sla.eigh(hmat, b=smat)
    esla = e0[0] 
    vsla = v0[:, 0]
    hmat = np.array(hmat)
    smat = np.array(smat)
    e, v = slater.solve_lc_coeffs(hmat, smat, return_vec=True)
    v = np.array(v)
    h = v.conj().T.dot(hmat).dot(v)
    s = v.conj().T.dot(smat).dot(v)
    e2 = h/s
    assert np.allclose(e, esla)
    assert np.allclose(e, e2)
    assert np.allclose(v, vsla)


def test_rdm12_compare_mf():
    h1e, h2e, _ = pyscf_helper.get_integrals(mf)
    norb, nocc, nvir, mo_coeff = pyscf_helper.get_mos(mf)
    Eelec_hf = mf.energy_elec()[0]
    # first compare to HF
    tvecs_hf = np.zeros((1, 2*nvir*nocc))
    rmats = slater.tvecs_to_rmats(tvecs_hf, nvir, nocc)
    c = np.array([1.])
    rdm1s, rdm2s = slater.make_rdm12(rmats, mo_coeff, c)
    dm1_hf = np.asarray(mf.make_rdm1())
    dm2_hf = np.asarray(mf.make_rdm2())
    assert np.allclose(rdm1s, dm1_hf) 
    assert np.allclose(dm2_hf, rdm2s)
    # test energy
    E1 = np.einsum("ij, sji ->", h1e, rdm1s)
    E2 = np.einsum("ijkl, sjilk ->", h2e, rdm2s)
    E2 += np.einsum("ijkl, jilk ->", h2e, rdm2s[1].transpose(2,3,0,1))
    E = E1 + 0.5*E2
    assert np.allclose(E, Eelec_hf)

    # test electron number
    tvecs_n = np.random.rand(1, 2*nvir*nocc)
    tvecs_all = np.vstack([tvecs_hf, tvecs_n])
    rmats = slater.tvecs_to_rmats(tvecs_all, nvir, nocc)
    hmat, smat = slater.noci_energy(rmats, mo_coeff, h1e, h2e, return_mats=True)
    energy, c = slater.solve_lc_coeffs(hmat, smat, return_vec=True)
    rdm1 = slater.make_rdm1(rmats, mo_coeff, c)
    ne_a = np.sum(np.diag(rdm1[0]))
    ne_b = np.sum(np.diag(rdm1[1]))
    assert np.allclose(ne_a, 2)
    assert np.allclose(ne_b, 2)


def test_rdm12s_compare_cisd():
    '''
    Compare to the CISD 
    '''
    # mol = gto.Mole()
    # mol.atom = "H 0 0 0; H 0 0 1.2; H 0 0 2.4; H 0 0 3.6"
    # mol.unit='angstrom'
    # mol.basis = "sto6g"
    # mol.build()
    # mf = pyscf_helper.mf_with_ortho_ao(mol, spin_symm=False)
    # mf.kernel(verbose=0, tol=1e-10)
    h1e, h2e, _ = pyscf_helper.get_integrals(mf)
    norb, nocc, nvir, mo_coeff = pyscf_helper.get_mos(mf)
    c_mo2ao = np.linalg.inv(mo_coeff)
    aa = (c_mo2ao[0], c_mo2ao[0], c_mo2ao[0], c_mo2ao[0])
    ab = (c_mo2ao[0], c_mo2ao[0], c_mo2ao[1], c_mo2ao[1])
    bb = (c_mo2ao[1], c_mo2ao[1], c_mo2ao[1], c_mo2ao[1])

    myci = ci.UCISD(mf)
    e_corr, civec = myci.kernel()
    dt = 0.1
    tmats, lc_coeff = nocisd.compress(myci, civec=civec, dt1=dt, dt2=dt, tol2=1e-5)
    rmats = slater.tvecs_to_rmats(tmats, nvir, nocc)
    rdm1, rdm2 = slater.make_rdm12(rmats, mo_coeff, lc_coeff)
    # check rdm1
    rdm1_ci = np.array(myci.make_rdm1())
    rdm1_ci[0] = basis_transform.basis_trans_mat(rdm1_ci[0], c_mo2ao[0])
    rdm1_ci[1] = basis_transform.basis_trans_mat(rdm1_ci[1], c_mo2ao[1])
    assert np.allclose(rdm1, rdm1_ci)
    # check rdm2
    rdm2_ci = np.array(myci.make_rdm2())
    rdm2_ci[0] = ao2mo.incore.general(rdm2_ci[0], aa, compact=False).reshape(norb, norb, norb, norb)
    rdm2_ci[1] = ao2mo.incore.general(rdm2_ci[1], ab, compact=False).reshape(norb, norb, norb, norb)
    rdm2_ci[2] = ao2mo.incore.general(rdm2_ci[2], bb, compact=False).reshape(norb, norb, norb, norb)
    assert np.allclose(rdm2_ci, rdm2)

    # test diagonal
    dm1_diag, dm2_ud_diag = slater.make_rdm12_diag(rmats, mo_coeff, lc_coeff) 
    rdm2_ci_diag = np.zeros((norb, norb))
    for i in range(norb):
        for j in range(norb):
            rdm2_ci_diag[i, j] = rdm2_ci[1][i,i,j,j]
    assert np.allclose(dm1_diag[0], np.diag(rdm1[0]))
    assert np.allclose(dm1_diag[1], np.diag(rdm1[1]))
    assert np.allclose(dm2_ud_diag, rdm2_ci_diag)

def test_orthonormal():

    nvir = 5
    nocc = 3
    norb = nvir + nocc
    tmats = np.random.rand(nvir, nocc)
    r = slater.orthonormal_mos(tmats)
    I = np.eye(norb)
    assert np.allclose(np.dot(r.T, r), I)
    nt = 3
    spin = 2
    tmats = np.random.rand(nt, spin, nvir, nocc)
    rotm = slater.orthonormal_mos(tmats)
    x = np.moveaxis(rotm, -2, -1).conj() @ rotm
    I_all = np.tile(I, nt*spin).T.reshape(nt, spin, norb, norb)
    assert np.allclose(x, I_all)


def test_spin_flip():
    # mol = gto.Mole()
    # mol.atom = "H 0 0 0; H 0 0 1.5; H 0 0 3.0; H 0 0 4.5"
    # mol.unit='angstrom'
    # mol.basis = "sto6g"

    # # mol.symmetry = True
    # mol.build()
    # norb = mol.nao
    # nocc = 2
    # mf = scf.UHF(mol)
    # pyscf_helper.run_stab_scf(mf)
    pyscf_helper.run_stab_scf_breaksymm(mf)
    h1e, h2e, _ = pyscf_helper.get_integrals(mf)
    norb, nocc, nvir, mo_coeff = pyscf_helper.get_mos(mf)
    Cmo = np.asarray(mo_coeff)
    Ca = Cmo[0]
    Cb = Cmo[1]
    U = np.linalg.inv(Ca) @ Cb
    assert np.allclose(U.T.conj()@U, np.eye(norb))

    rmats = np.random.rand(3, 2, norb, nocc)
    rmats_n = slater.half_spin(rmats, U=U)

    dets = np.einsum("sij, nsjk -> nsik", Cmo, rmats)
    dets_n = np.einsum("sij, nsjk -> nsik", Cmo, rmats_n)
    assert np.allclose(dets[:, 0], dets_n[:, 1])
    assert np.allclose(dets[:, 1], dets_n[:, 0])

    rmats = np.zeros((1, 2, norb, nocc))
    rmats[0, 0, :nocc] = np.eye(nocc)
    rmats[0, 1, :nocc] = np.eye(nocc)

    rmats_n = slater.half_spin(rmats, U=U)
    rmats_all = np.vstack([rmats, rmats_n])
    h1e, h2e, e_nuc = pyscf_helper.get_integrals(mf, ortho_ao=False)
    e = slater.noci_energy(rmats_all, Cmo, h1e, h2e, e_nuc=e_nuc)
    print(e)

