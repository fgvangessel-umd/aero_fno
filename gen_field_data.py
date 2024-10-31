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

ncols = 5
nvars = 8

cases = glob.glob('../field_data/case*')


for d in cases:
    slices = glob.glob(d+'/fc*_surf.dat')
    for fname in slices:
        with open(fname, 'r') as f:
            s = f.readlines()

        case = fname.split('_')[2].split('/')[0]
        slce = fname.split('_')[3]
        print(case+'_'+slce)

        fvars = read_field_data(s, 1.5*xmin, 1.5*xmax, 1.5*ymin, 1.5*ymax)
        x, y = fvars[0], fvars[1]
        Xi, Yi, fvi = interpolate_to_uniform_grid(xi, yi, fvars)

        fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(14, 18))

        for i in range(3):
            ax = axs[i,0]
            z = fvars[2+i]
            zi = fvi[i]

            ax.tricontour(x, y, z, levels=14, linewidths=0.5, colors='k')
            cntr1 = ax.tricontourf(x, y, z, levels=14, cmap="RdBu_r")
            fig.colorbar(cntr1)

            ax = axs[i,1]
            ax.contour(Xi, Yi, zi, levels=14, linewidths=0.5, colors='k')
            cntr2 = ax.contourf(Xi, Yi, zi, levels=14, cmap="RdBu_r")
            fig.colorbar(cntr2)

        fig.savefig('interpolated_field_data/'+case+'_'+slce+'_fields.png')
        np.savez_compressed('interpolated_field_data/'+case+'_'+slce+'_fields.npz', \
                            x=Xi, y=Yi, p=fvi[0], vx=fvi[1], vy=fvi[2])
        plt.close()
'''

data = np.load('interpolated_field_data/1023_056_fields.npz')
Xi = data['x']
Yi = data['y']
p = data['p']
vx = data['vx']
vy = data['vy']
fvars = [p, vx, vy]

print(Xi.shape, Yi.shape)

fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(7, 18))

for i in range(3):
    ax = axs[i]
    z = fvars[i]
    zi = fvars[i]

    ax.contour(Xi, Yi, zi, levels=14, linewidths=0.5, colors='k')
    cntr2 = ax.contourf(Xi, Yi, zi, levels=14, cmap="RdBu_r")
    fig.colorbar(cntr2)

fig.savefig('fields.png')
'''