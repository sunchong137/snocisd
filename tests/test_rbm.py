import numpy as np
from pyscf import gto, scf, fci
import sys
sys.path.append('../')
import helpers, molecules, noci
# import rbm_slow as rbm
import rbm

def test_expand_hiddens():
    h = [0, 1]
    rep = 3 
    coeff = rbm.expand_hiddens(h, rep)
    print(coeff)
test_expand_hiddens()

def test_params_to_rotations():
    nocc = 2
    nvir = 3
    norb = nocc + nvir
    rshape = (nvir, nocc)
    nvecs = 2
    rbm_vecs = np.random.rand(nvecs, 2*nocc*nvir)
    rmats = rbm.params_to_rotations(rbm_vecs, rshape, normalize=False)
    
    # test v0
    v0 = rmats[0].flatten(order='C')
    v0ref = np.zeros(2*nocc*norb)
    v0ref[0] = 1
    v0ref[3] = 1
    v0ref[nocc*norb] = 1
    v0ref[nocc*norb+3] = 1
    assert np.allclose(v0, v0ref)
    
    # test v1 
    v1 = rmats[1].flatten(order='C')
    v1ref = np.copy(v0ref)
    v1ref[nocc**2:(nocc*norb)] = rbm_vecs[-1][:nocc*nvir]
    v1ref[nocc**2+(nocc*norb):] = rbm_vecs[-1][nocc*nvir:]
    r1ref = v1ref.reshape(2, norb, nocc)
    assert np.allclose(rmats[1], r1ref)
    assert np.allclose(v1, v1ref)


def test_thouless():
    nocc = 2
    nvir = 3
    rshape = (nvir, nocc)
    nvecs = 2
    rbm_vecs = np.random.rand(nvecs, 2*nocc*nvir)
    rmats = rbm.params_to_rotations(rbm_vecs, rshape, normalize=False)
    assert len(rmats) == 4


def test_expand_vectors():

    # test empty vecs
    rbm_all = rbm.expand_vecs([])
    assert np.allclose(rbm_all, [])

    vec = np.random.rand(4)
    rbm_all = rbm.expand_vecs([vec])
    assert np.allclose(rbm_all[0], vec*0)
    
def test_add_vec():
    vec = np.random.rand(4) 
    vec_list = [np.zeros(4)]
    new_vec = rbm.add_vec(vec, vec_list)
    assert np.allclose(new_vec[0], vec)

def test_rbm_all():
    nH = 4
    a = 1.5
    # construct molecule
    mol = gto.Mole()
    mol.atom = molecules.gen_geom_hchain(nH, a)
    mol.basis = "sto3g"
    mol.build()
    norb = mol.nao
    ao_ovlp = mol.intor_symmetric ('int1e_ovlp')
    
    # UHF
    mf = scf.UHF(mol)
    h1e = mf.get_hcore()
    h2e = mol.intor('int2e')

    # Hartree-Fock
    init_guess = mf.get_init_guess()
    helpers.make_init_guess(init_guess)
    mf.init_guess = init_guess
    mf.kernel()
    occ = mf.get_occ()
    nocc = int(np.sum(occ[0]))
    nvir = norb - nocc

    # energy
    e_nuc = mf.energy_nuc()
    e_hf = mf.energy_tot()
    mo_coeff = np.asarray(mf.mo_coeff)

    # FCI
    myci = fci.FCI(mf)
    e_fci, c = myci.kernel()

    rot0_u = np.zeros((norb, nocc))
    rot0_u[:nocc, :nocc] = np.eye(nocc)
    r0 = np.zeros((nvir, nocc))

    nvecs = 2
    rmats = np.random.rand(nvecs, 2*nvir*nocc)

    rs = noci.gen_thouless_singles(nocc, nvir, zmax=2, zmin=0.01)
    tmats = rs[:nvecs]
    noise = np.random.rand(nvecs, 2*nvir*nocc) * 5e-2
    tmats = noise + tmats.reshape(nvecs, -1)
    er2, vecs2 = rbm.rbm_all(h1e, h2e, mo_coeff, nocc, nvecs, init_rbms=tmats, ao_ovlp=ao_ovlp, tol=1e-5, MaxIter=10)
    e_rbm2 = er2 + e_nuc

    assert e_rbm2 >= e_fci
    assert e_rbm2 <= e_hf
   

def test_rbm_fed():

    nH = 4
    a = 1.5
    # construct molecule
    mol = gto.Mole()
    mol.atom = molecules.gen_geom_hchain(nH, a)
    mol.basis = "sto3g"
    mol.build()
    norb = mol.nao
    ao_ovlp = mol.intor_symmetric ('int1e_ovlp')
    
    # UHF
    mf = scf.UHF(mol)
    h1e = mf.get_hcore()
    h2e = mol.intor('int2e')

    # Hartree-Fock
    init_guess = mf.get_init_guess()
    helpers.make_init_guess(init_guess)
    mf.init_guess = init_guess
    mf.kernel()
    occ = mf.get_occ()
    nocc = int(np.sum(occ[0]))
    nvir = norb - nocc

    # energy
    e_nuc = mf.energy_nuc()
    e_hf = mf.energy_tot()
    mo_coeff = np.asarray(mf.mo_coeff)

    # FCI
    myci = fci.FCI(mf)
    e_fci, c = myci.kernel()

    rot0_u = np.zeros((norb, nocc))
    rot0_u[:nocc, :nocc] = np.eye(nocc)
    r0 = np.zeros((nvir, nocc))

    nvecs = 2
    tmats = np.random.rand(nvecs, 2*nvir*nocc)

    rs = noci.gen_thouless_singles(nocc, nvir, zmax=2, zmin=0.01)
    tmats = rs[:nvecs]
    noise = np.random.rand(nvecs, 2*nvir*nocc) * 5e-2
    tmats = noise + tmats.reshape(nvecs, -1)
    er, vecs = rbm.rbm_fed(h1e, h2e, mo_coeff, nocc, nvecs, init_rbms=tmats, ao_ovlp=ao_ovlp, nsweep=0, tol=1e-5, MaxIter=10)
    e_rbm = er + e_nuc
    assert e_rbm >= e_fci
    assert e_rbm <= e_hf
