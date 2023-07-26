import numpy as np


def print_array(mat, prec=6):
    '''
    print the numpy array with commas.
    '''
    n_row, n_col = mat.shape

    str_row = "[["
    for j in range(n_col-1):
        str_row += "{0:.6f}, ".format(mat[0,j])
    str_row += "{0:.6f}],".format(mat[0, n_col-1])
    print(str_row)

    for i in range(1, n_row-1):
        str_row = "["
        for j in range(n_col-1):
            str_row += "{0:.6f}, ".format(mat[i,j])
        str_row += "{0:.6f}],".format(mat[i, n_col-1])
        print(str_row)

    str_row = "["
    for j in range(n_col-1):
        str_row += "{0:.6f}, ".format(mat[n_row-1,j])
    str_row += "{0:.6f}]]".format(mat[n_row-1, n_col-1])
    print(str_row)

def make_init_guess(dm0):
    '''
    Break symmetry of the UHF initial guess.
    NOTE: very case dependent, not useful.
    '''
    dm0[0][0,0] = 10 
    dm0[1][0,0] = 0
    return dm0 
