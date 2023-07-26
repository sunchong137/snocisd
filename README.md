NOCI with RBM
-------------

Author: Chong Sun [email](sunchong137@gmail.com)

Dependencies:
- PySCF 
- Numpy
- Scipy

### Naming systems:
1. params: parameters of RBM, usually with the shape (nparam, lparam)
2. tvecs: the vector form of the Thouless matrices (flattened)
3. tmats: Thouless matrices 
4. rmats: rotation matrices of the MO coefficients (adding I on top of tmats)
5. sdets: MO coefficients of the Slater determinants