# Restricted Boltzmann machine for NOCI
# Only works for hidden variables = 0, 1

import numpy as np
from scipy.optimize import minimize
import slater, noci
import itertools


def expand_vecs(rbmvecs, hiddens=[0,1]):
    '''
    Expand the RBM vectors
    '''
    nvecs = len(rbm_vecs)
    if nvecs == 0:
        return []
    rbm_vecs = np.asarray(rbm_vecs)
    tvecs = [] 

    for iter in itertools.product(hiddens, repeat=nvecs):
        sum_coeff = np.asarray(iter)
        t = np.dot(sum_coeff, rbm_vecs)
        tvecs.append(t)

    return tvecs  


def vectors_to_thouless(rbm_vecs, tshape, hiddens=[0,1]):
    '''
    Expand RBM vectors to Thouless rotations.
    '''
    nvecs = len(rbm_vecs)
    if nvecs == 0:
        return []
    rbm_vecs = np.asarray(rbm_vecs)
    nvir, nocc = tshape
    tmats = [] 

    for iter in itertools.product(hiddens, repeat=nvecs):
        sum_coeff = np.asarray(iter)
        t = np.dot(sum_coeff, rbm_vecs).reshape(2, nvir, nocc)
        tmats.append(t)

    return tmats  

def vectors_to_rotations(rbm_vecs, tshape, hiddens=[0,1], normalize=True):
    '''
    Turn RBM parameters into rotation matrices.
    Args:
        rbm_vecs: 2D array of size (nvecs, 2 x Nocc x Nvir), neural network weights.
        tshape: shape of the Thouless matrix (nvir, nocc)
    Kwargs:
        hiddens: hidden variables.
    Returns:
        A 3D numpy array of size (l^d, Norb, Nocc), where l is the length of hidden variables.
    '''
    nvecs = len(rbm_vecs)
    if nvecs == 0:
        return []
    rbm_vecs = np.asarray(rbm_vecs)
    nvir, nocc = tshape
    rmats = []

    for iter in itertools.product(hiddens, repeat=nvecs):
        sum_coeff = np.asarray(iter)
        t = np.dot(sum_coeff, rbm_vecs).reshape(2, nvir, nocc)
        r = slater.thouless_to_rotation(t) # put the identity operator on top
        if normalize:
            r = slater.normalize_rotmat(r)
        rmats.append(r)
    return rmats


def rbm_all(h1e, h2e, mo_coeff, nocc, nvecs, 
            init_rbms=None, ao_ovlp=None, hiddens=[0, 1],
            tol=1e-7, MaxIter=100):
    '''
    Optimize the RBM parameters all together.
    Args:
        h1e: 2D array, one-body Hamiltonian
        h2e: 4D array, two-body Hamiltonian
        nocc: int, number of occupied orbitals
        nvecs: int, number of rbm_vectors
    kwargs:
        init_rbms: a list of vectors, initial guess of the RBM parameters.
        ao_ovlp: 2D array, overlap matrix among atomic orbitals
        hiddens: hidden variables for RBM neural networks.
    NOTE: hard to converge when optimizing all.
    '''
    norb = h1e.shape[-1]
    nvir = norb - nocc
    tshape = (nvir, nocc)
    lt = 2*nvir*nocc # 2 for spins

    if init_rbms is None:
        init_rbms = np.random.rand(nvecs, lt) 

    init_rbms = init_rbms.flatten(order='C')

    def cost_func(w):
        w_n = w.reshape(nvecs, lt)
        rmats = vectors_to_rotations(w_n, tshape, hiddens=hiddens, normalize=True)
        sdets = slater.gen_determinants(mo_coeff, rmats)
        ham_mat = noci.full_hamilt_w_sdets(sdets, h1e, h2e, ao_ovlp=ao_ovlp)
        ovlp_mat = noci.full_ovlp_w_rotmat(rmats)
        e = noci.solve_lc_coeffs(ham_mat, ovlp_mat)
        return e
    
    # optimize
    rbm_vecs = minimize(cost_func, init_rbms, method="CG", tol=tol, options={"maxiter":MaxIter, "disp": True}).x
    final_energy = cost_func(rbm_vecs)
    
    return final_energy, rbm_vecs

def rbm_sweep(h1e, h2e, mo_coeff, nocc, nvecs, 
            init_rbms=None, ao_ovlp=None, hiddens=[0, 1],
            max_nsweep=3, tol=1e-7, MaxIter=100):
    '''
    Kwargs:
        max_nsweep: maximum number of sweeps
    Optimize the RBM parameters one by one.
    '''
    norb = h1e.shape[-1]
    nvir = norb - nocc
    tshape = (nvir, nocc)

    if init_rbms is None:
        init_rbms = np.random.rand(nvecs, 2*nvir*nocc) # 2 for spins

    init_rbms = init_rbms.flatten(order='C')

    def cost_func_all(w, w_fix):
        w_n = np.vstack([w_fix, w])
        w_n = w_n.reshape(nvecs, 2*nvir*nocc)
        rmats = vectors_to_rotations(w_n, tshape, hiddens=hiddens, normalize=True)
        sdets = slater.gen_determinants(mo_coeff, rmats)
        ham_mat = noci.full_hamilt_w_sdets(sdets, h1e, h2e, ao_ovlp=ao_ovlp)
        ovlp_mat = noci.full_ovlp_w_rotmat(rmats)
        e = noci.solve_lc_coeffs(ham_mat, ovlp_mat)
        return e
    
    # sweep
    rbm_vecs = np.copy(init_rbms)
    rbm_vecs = rbm_vecs.reshape(nvecs, -1)
    for s in range(max_nsweep):
        for i in range(nvecs):
            w0 = np.copy(rbm_vecs[i])
            def cost_func(w):
                return cost_func_all(w, np.delete(rbm_vecs, i, axis=0))
            w = minimize(cost_func, w0, method="BFGS", tol=tol, options={"maxiter":MaxIter, "disp": False}).x
            rbm_vecs[i] = w
    final_energy = cost_func_all(rbm_vecs[0], np.delete(rbm_vecs, 0, axis=0))
    print("Final energy: ", final_energy)
    
    return final_energy, rbm_vecs

def rbm_fed(h1e, h2e, mo_coeff, nocc, nvecs, 
            init_rbms=None, ao_ovlp=None, hiddens=[0,1],
            nsweep=3, tol=1e-7, MaxIter=100):
    '''
    Kwargs:
        nsweep: maximum number of sweeps
    Optimize the RBM parameters one by one.
    '''
    norb = h1e.shape[-1]
    nvir = norb - nocc
    tshape = (nvir, nocc)

    if init_rbms is None:
        init_rbms = np.random.rand(nvecs, 2*nvir*nocc) # 2 for spins

    E0 = energy_rbm(init_rbms, mo_coeff, h1e, h2e, tshape, ao_ovlp=ao_ovlp, hiddens=hiddens)
    e_hf = E0

    opt_vecs = [] # optimized RBM vectors

    print("Start RBM FED...")
    for iter in range(nvecs):
        w0 = init_rbms[iter]
        e, w = opt_one_rbmvec(w0, opt_vecs, h1e, h2e, mo_coeff, tshape, 
                              ao_ovlp=ao_ovlp, hiddens=hiddens, tol=tol, MaxIter=MaxIter)
        de = e - E0
        E0 = e
        print("Iter {}: energy lowered {}".format(iter+1, de))
        opt_vecs.append(w)
   
    if nsweep > 0:
        print("Start sweeping...")
        for isw in range(nsweep):
            print("Sweep {}".format(isw+1))
            for iter in range(nvecs):
                # always pop the first vector and add the optimized to the end
                w0 = opt_vecs.pop(0)
                e, w = opt_one_rbmvec(w0, opt_vecs, h1e, h2e, mo_coeff, tshape, 
                                      ao_ovlp=ao_ovlp, hiddens=hiddens, tol=tol, MaxIter=MaxIter)
                de = e - E0
                E0 = e
                print("Iter {}: energy lowered {}".format(iter+1, de))
                opt_vecs.append(w)
    print("Total energy lowered: {}".format(e - e_hf))
    return e, opt_vecs

def opt_one_rbmvec(vec0, rbmvecs, h1e, h2e, mo_coeff, tshape, ao_ovlp=None, 
                   hiddens=[0,1], tol=1e-7, MaxIter=100):
    '''
    Optimize one RBM vector with the other fixed.
    Args:
        vec0: 1D array, the RBM vector to be optimized.
        rbmvecs: a list of 1D arrays

    Returns:
        float: energy
        1D array: optimized RBM vector.
    '''

    nvecs = len(rbmvecs) + 1

    def cost_func(w):
        w_n = rbmvecs + [w]
        w_n = np.asarray(w_n)
        w_n = w_n.reshape(nvecs, -1)
        rmats = vectors_to_rotations(w_n, tshape, hiddens=hiddens, normalize=True)
        sdets = slater.gen_determinants(mo_coeff, rmats)
        ham_mat = noci.full_hamilt_w_sdets(sdets, h1e, h2e, ao_ovlp=ao_ovlp)
        ovlp_mat = noci.full_ovlp_w_rotmat(rmats)
        e = noci.solve_lc_coeffs(ham_mat, ovlp_mat)
        return e
    
    # optimize 
    v = minimize(cost_func, vec0, method="BFGS", tol=tol, options={"maxiter":MaxIter, "disp": False}).x
    energy = cost_func(v)
    return energy, v

def energy_rbm(rbmvecs, mo_coeff, h1e, h2e, tshape, ao_ovlp=None, hiddens=[0,1]):

    rmats = vectors_to_rotations(rbmvecs, tshape, hiddens=hiddens, normalize=True)
    e = noci.noci_energy(rmats, mo_coeff, h1e, h2e, ao_ovlp=ao_ovlp, include_hf=True)
    return e

if __name__ == "__main__":
    print("Main function:\n")
    # params = np.random.rand(3,4)
    # params_to_rotations(params)
