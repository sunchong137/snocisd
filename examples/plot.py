import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os 

save_dir = "./figures/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
params = {
        'font.family': 'serif',
        "mathtext.fontset" : "stix",
        'legend.fontsize': 'large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large',
         'figure.autolayout': True,
         'savefig.dpi': 400
         }
rcParams.update(params)

def plot_no_crossing():
    data = np.loadtxt("data/no_cross_lif.txt")
    plt.plot(data[:, 0], data[:, 1], label="singlet")
    plt.plot(data[:, 0], data[:, 2], label="triplet")
    plt.legend()
    plt.show() 

