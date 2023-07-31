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

