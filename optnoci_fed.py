import numpy as np
from scipy.optimize import minimize
import slater, noci


def optimize_res(tmats0, mo_coeff, h1e, h2e, ao_ovlp=None, tol=1e-8, MaxIter=100):
    '''
    Given a set of Thouless rotations, optimize the parameters.
    Res HF approach, all parameters are optimized simultaneously.
    '''

    tmats0 = np.asarray(tmats0)
    tshape = tmats0.shape
    nvir, nocc = tmats0[0][0].shape # rmats[0] has two spins
    rot0_u = np.zeros((nvir+nocc, nocc))
    rot0_u[:nocc, :nocc] = np.eye(nocc)
    rot_hf = np.array([rot0_u, rot0_u]) # the HF state
    tmats0 = tmats0.flatten(order="C")
    E0 = noci.noci_energy([rot_hf], mo_coeff, h1e, h2e, ao_ovlp=ao_ovlp, include_hf=True)
    def cost_func(t):
        tmats = t.reshape(tshape)
        rmats = slater.thouless_to_rotation_all(tmats, normalize=True) # a list
        rmats = [rot_hf] + rmats
        sdets = slater.gen_determinants(mo_coeff, rmats, normalize=False)
        ham_mat = noci.full_hamilt_w_sdets(sdets, h1e, h2e, ao_ovlp=ao_ovlp)
        ovlp_mat = noci.full_ovlp_w_rotmat(rmats)
        e = noci.solve_lc_coeffs(ham_mat, ovlp_mat)
        return e

    t_vecs = minimize(cost_func, tmats0, method="BFGS", tol=tol, options={"maxiter":MaxIter, "disp": True}).x
    E = cost_func(t_vecs)
    de = E - E0
    print("Total energy lowered: {}".format(de))
    t_f = t_vecs.reshape(tshape)

    return E, t_f


def optimize_fed(tmats0, mo_coeff, h1e, h2e, ao_ovlp=None, 
                 tol=1e-8, MaxIter=100, nsweep=0):
    '''
    Given a set of Thouless rotations, optimize the parameters.
    Using FED (few-determinant) approach.
    Args:
        tmats0: a list of Thouless rotations as the initial guess.
        mo_coeff: a list of two 2D arrays
        h1e: 2D array, one-body Hamiltonian 
        h2e: 4D array, two-body Hamiltonian
    Kwargs:
        ao_ovlp: Overlap matrix among AO basis.
        tol: threshold to terminate minimization
        MaxIter: maximum number of iterations
        nsweep: number of sweeps

    Returns:
        float, final energy
        a list of arrays, the optimized Thouless parameters.
    '''

    # construct the list of rotations starting
    # from the HF state.

    nvir, nocc = tmats0[0][0].shape # rmats[0] has two spins
    rot0_u = np.zeros((nvir+nocc, nocc))
    rot0_u[:nocc, :nocc] = np.eye(nocc)
    rot_hf = np.array([rot0_u, rot0_u]) # the HF state
    rmats_new = [rot_hf]
    # evaluate hmat and smat
    sdets = slater.gen_determinants(mo_coeff, rmats_new)
    hmat = noci.full_hamilt_w_sdets(sdets, h1e, h2e, ao_ovlp=ao_ovlp)
    smat = noci.full_ovlp_w_rotmat(rmats_new)
    e_hf = noci.solve_lc_coeffs(hmat, smat)
    E0 = e_hf 
    num_t = len(tmats0)
    # Start optimization
    print("Starting NOCI optimization with FED approach...")

    for iter in range(num_t):
        t0 = tmats0[iter]
        smat0 = np.copy(smat)
        hmat0 = np.copy(hmat)
        E, t, hmat, smat = opt_one_thouless(t0, rmats_new, mo_coeff, h1e, h2e, hmat=hmat0, smat=smat0, ao_ovlp=ao_ovlp, tol=tol, MaxIter=MaxIter)
        de = E - E0
        print("Iter {}: energy lowered {}".format(iter+1, de))
        E0 = E
        tmats0[iter] = np.copy(t)
        r = slater.thouless_to_rotation(t, normalize=True)
        rmats_new.append(r)
    de_fed = E - e_hf  
    print("***Energy lowered after FED: {}".format(de_fed))

    # Start sweeping
    if nsweep > 0:
        print("Start sweeping...")
        sdets = slater.gen_determinants(mo_coeff, rmats_new)
        hmat = noci.full_hamilt_w_sdets(sdets, h1e, h2e, ao_ovlp=ao_ovlp)
        smat = noci.full_ovlp_w_rotmat(rmats_new) 
 
        for isw in range(nsweep):
            print("Sweep {}".format(isw+1))
            E_s = E0
            for iter in range(num_t):
                t0 = tmats0[iter]
                rmats_new.pop(1)
                hmat0 = np.delete(hmat, 1, axis=0)
                hmat0 = np.delete(hmat0, 1, axis=1)
                smat0 = np.delete(smat, 1, axis=0)
                smat0 = np.delete(smat0, 1, axis=1)
                E, t, hmat, smat = opt_one_thouless(t0, rmats_new, mo_coeff, h1e, h2e, hmat=hmat0, smat=smat0, ao_ovlp=ao_ovlp, tol=tol, MaxIter=MaxIter)
                de = E - E0
                print("Iter {}: energy lowered {}".format(iter+1, de))
                E0 = E
                tmats0[iter] = np.copy(t)
                r = slater.thouless_to_rotation(t, normalize=True)
                rmats_new.append(r)
            de_s = E - E_s 
            print("***Energy lowered after Sweep {}: {}".format(isw+1, de_s))

    len_new_rots = len(rmats_new)
    if len_new_rots < num_t:
        print("WARNING: only {} vectors are successfully optimized!".format(len_new_rots))
    de_tot = E - e_hf 
    print("SUMMARY: Total energy lowered {}".format(de_tot))
    return E, np.asarray(tmats0)


def opt_one_thouless(t0, rmats, mo_coeff, h1e, h2e, hmat=None, smat=None, ao_ovlp=None, tol=1e-8, MaxIter=100, grad=False):
    '''
    Adding a new determinant to the existing ones and optimize it's Thouless parameters.
    Args:
        t0: a list of two (Nvir, Nocc) arrays, initial guess of the new Thouless rotation matrix.
        rmats: a list of arrays, Thouless matrices of the existing determinants
        mo_coeff: 
        h1e: 2d array, one-body Hamiltonian
        h2e: 4d array, two-body Hamiltonian
    Kwargs:
        hmat: Hamiltonian matrix from rmats
        smat: overlap matrix from rmats
    Returns:
        array: optimized Thouless parameters
        float: energy
        # TODO could use a class to avoid returning large matrices.
    '''
    num_rots = len(rmats) + 1
    t0 = np.asarray(t0) 
    shape_t0 = t0.shape
    sdets = slater.gen_determinants(mo_coeff, rmats)
    if hmat is None: # construct previous Hamiltonian matrix
        hmat = noci.full_hamilt_w_sdets(sdets, h1e, h2e, ao_ovlp=ao_ovlp)
    if smat is None: # construct previous overlap matrix
        smat = noci.full_ovlp_w_rotmat(rmats)


    hn = np.zeros((num_rots, num_rots))
    sn = np.zeros((num_rots, num_rots))
    hn[:-1, :-1] = hmat.copy()
    sn[:-1, :-1] = smat.copy()


    def cost_func(t): 
        _t = np.asarray(t).reshape(shape_t0)
        # thouless to rotation
        r_n = slater.thouless_to_rotation(_t, normalize=True)
        sdet_n = slater.rotation(mo_coeff, r_n)
        energy, gradient, _, _ = _expand_hs(hn, sn, sdet_n, sdets, h1e, h2e, mo_coeff, ao_ovlp=ao_ovlp)
        # return energy, gradient.flatten(order="C")
        return energy  
          
    # Optimize
    tmat_n = minimize(cost_func, t0.flatten(), method="BFGS", tol=tol, jac=False, options={"maxiter":MaxIter, "disp":True}).x
    # Evaluate return values
    tmat_n = tmat_n.reshape(shape_t0)
    r_n = slater.thouless_to_rotation(tmat_n, normalize=True)
    sdet_n = slater.rotation(mo_coeff, r_n)
    E, gradient, h_n, s_n = _expand_hs(hn, sn, sdet_n, sdets, h1e, h2e, mo_coeff, ao_ovlp=ao_ovlp)
    #E = noci.solve_lc_coeffs(h, s)

    return E, tmat_n, h_n, s_n

def _expand_hs(hn, sn, sdet0, sdets, h1e, h2e, mo_coeff, ao_ovlp=None):
    '''
    Evaluate hamiltonian matrix and overlap matrix with the new determinant.
    Args:
        hn: 2D array, the last row and column are zeros
        sn: 2D array, the last row and column are zeros
        r0: the rotation matrix to be added
        sdet0: the corresponding determinant to be added.
        rmats: the existing rotation matrices
        sdets: the existing determinants
    '''
    hmat = np.copy(hn)
    smat = np.copy(sn)
    n_rots = len(sdets)
    nocc = sdet0[0].shape[-1]
    mo_vir = mo_coeff[:, :, nocc:]
    mo_occ = mo_coeff[:, :, :nocc]
    grads = []
    rdms = []
    for i in range(n_rots):
        dm, ovlp = slater.make_trans_rdm1(sdets[i], sdet0, ao_ovlp=ao_ovlp)
        rdms.append(dm)
        h, _g = slater.trans_hamilt_all(dm, h1e, h2e, get_grad=True)
        e = h * ovlp
        smat[i, -1] = ovlp 
        smat[-1, i] = ovlp.conj()
        hmat[i, -1] = e
        hmat[-1, i] = e.conj()
        g_up = mo_vir[0].T @ _g[0] @ mo_occ[0].conj() * ovlp
        g_dn = mo_vir[1].T @ _g[1] @ mo_occ[1].conj() * ovlp
        grads.append(np.array([g_up, g_dn]))

    # the last one 
    dm, ovlp = slater.make_trans_rdm1(sdet0, sdet0, ao_ovlp=ao_ovlp)
    rdms.append(dm)
    h, _g = slater.trans_hamilt_all(dm, h1e, h2e, get_grad=True)

    smat[-1, -1] = ovlp 
    hmat[-1, -1] = h * ovlp

    g_up = mo_vir[0].T @ _g[0] @ mo_occ[0].conj() * ovlp
    g_dn = mo_vir[1].T @ _g[1] @ mo_occ[1].conj() * ovlp
    grads.append(np.array([g_up, g_dn]))

    energy, lc = noci.solve_lc_coeffs(hmat, smat, return_vec=True)
    gradient = np.zeros_like(grads[0])
    for i in range(n_rots+1):
        g_up = mo_vir[0].T @ rdms[i][0] @ mo_occ[0].conj()
        g_dn = mo_vir[1].T @ rdms[i][1] @ mo_occ[1].conj()

        grad_all = grads[i] # + np.array([g_up, g_dn])*ovlp*(hmat[i, -1]-energy)
        gradient += lc[i].conj() * grad_all

    gradient *= lc[-1]
    gradient /= lc.conj().T @ smat @ lc

    return energy, gradient, hmat, smat
