import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib.tri as tri
from reorder import reorder_coords
import glob
from functions import *

def get_sample_points(case, slce):
    '''
    Get poitn in domain at which we sample the SDF
    '''
    fname = 'data/field_data/'+case+'_'+slce+'_fields.npz'
    data = np.load(fname)
    Xi = data['x']
    Yi = data['y']

    sxy = np.concatenate((Xi.reshape(-1,1), Yi.reshape(-1,1)), axis=1)
    return sxy


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

cases = glob.glob('data/slice_data/case*')
print(cases)

for d in cases:
    slices = glob.glob(d+'/fc*_slices.dat')
    for fname in slices:
        with open(fname, 'r') as f:
            s = f.readlines()

        case = fname.split('_')[2].split('/')[0]
        slce = fname.split('_')[3]
        print(case, slce)
        
        d2_df = read_slice_data(s)

        
        sxy = get_sample_points(case, slce)
        
        sdf = gen_sdf_data(d2_df, sxy)

        # Plot
        fname = f'data/sdf_data/sdf'+case+'_'+slce+'.png'
        plot_sdf(sxy, sdf, fname)

        # Save data
        fname = f'data/sdf_data/sdf'+case+'_'+slce+'.npz'
        save_sdf(sxy, sdf, fname)

        break
    break