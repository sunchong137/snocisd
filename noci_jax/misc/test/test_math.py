import numpy as np 
from noci_jax.misc import math_helpers 
import time 


def test_gev():
    n = 1000
    h = np.random.rand(n, n) 
    h += h.T
    s = np.eye(n) + np.random.rand(n, n) * 0.05
    s += s.T 
    s /= 2
    t1 = time.time()
    e, v = math_helpers.generalized_eigh(h, s)
    t2 = time.time()
    e1, v1 = math_helpers.generalized_eigh_singular(h, s)
    t3 = time.time() 
    print(t2-t1, t3-t2)
    assert np.allclose(e, e1)
    a, b = np.linalg.eigh(s)
    print(e, e1)
test_gev()