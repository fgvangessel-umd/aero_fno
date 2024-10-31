import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib.tri as tri
from reorder import reorder_coords
import glob
from functions import *


# Define clipped region
xmin, xmax = -0.75, 1.75
ymin, ymax = -0.75, 0.75

# Define uniform grid
ngrid = 264
ngridx = ngrid
ngridy = ngrid
xi = np.linspace(xmin, xmax, ngridx)
yi = np.linspace(ymin, ymax, ngridy)
Xi, Yi = np.meshgrid(xi, yi)

ncols = 5
nvars = 8

cases = glob.glob('../field_data/case*')

for d in cases:
    slices = glob.glob(d+'/fc*_slices.dat')
    for fname in slices:
        with open(fname, 'r') as f:
            s = f.readlines()

        case = fname.split('_')[2].split('/')[0]
        slce = fname.split('_')[3]
        print(case+'_'+slce)
        
        
        d2_df = read_slice_data(s)

        
        sxy = np.concatenate((Xi.reshape(-1,1), Yi.reshape(-1,1)), axis=1)
        sdf = gen_sdf_data(d2_df, sxy)

        # Plot
        fname = f'sdf_data/sdf'+case+'_'+slce+'.png'
        plot_sdf(xy, sdf, sb, fname)

        # Save data
        fname = f'sdf_data/sdf'+case+'_'+slce+'.npz'
        save_sdf(xy, sdf, sb, fname)

        break

    break
