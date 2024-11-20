import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib.tri as tri
from reorder import reorder_coords
import glob
from modulus.sym.geometry.primitives_2d import Polygon

def convert_list_to_array(input_list):
    # Initialize an empty list to store all the numbers
    all_numbers = []
    
    # Iterate through each string in the input list
    for string in input_list:
        # Split the string by whitespace and convert each number to float
        numbers = [float(num) for num in string.strip().split()]
        # Extend the all_numbers list with these numbers
        all_numbers.extend(numbers)
    
    # Convert the list of all numbers to a numpy array
    return np.array(all_numbers)

def mask_dims(vars, xmin, xmax, ymin, ymax):
    # Apply z mask
    mask = vars[2]<0.5
    vars = [var[mask] for var in vars]

    # Apply x mask
    mask = vars[0]>xmin
    vars = [var[mask] for var in vars]
    mask = vars[0]<xmax
    vars = [var[mask] for var in vars]

    # Apply y mask
    mask = vars[1]>ymin
    vars = [var[mask] for var in vars]
    mask = vars[1]<ymax
    vars = [var[mask] for var in vars]

    return vars

def read_field_data(s, xmin, xmax, ymin, ymax):
    # Extract symm zone
    nnods = []
    nelms = []

    ncols = 5
    nvars = 8

    fvars = []

    for i, line in enumerate(s):
        if line.strip().startswith('ZONE T="sym"'):
            nnod = int(s[i+2].split('=')[1].split()[0].split(',')[0])
            nelm = int(s[i+2].split('=')[2].split()[0].split(',')[0])

            if nnod % ncols == 0:
                nrows = nnod//ncols
            else:
                nrows = nnod//ncols + 1

            for j in range(nvars):
                offset = nrows*j
                dat = convert_list_to_array(s[offset+i+5:offset+i+nrows+5])
                fvars.append(dat)

            break

    fvars = mask_dims(fvars, xmin, xmax, ymin, ymax)

    # Keep only X-coordinate, Y-Coordinate, CoefPressure, VelocityX, and VelocityY data
    idxs = [0, 1, 3, 4, 6]
    fvars  = [fvars[i] for i in idxs]

    return fvars

def read_slice_data(s):
    # Extract Slice Data

    nnods = []
    nelms = []

    wall_vars = []

    for i, line in enumerate(s):
        if line.strip().startswith('DATAPACKING=POINT'):
            nnod = int(s[i-1].split('=')[1].split()[0].split(',')[0])
            nelm = int(s[i-1].split('=')[2].split()[0].split(',')[0])
            
            nrows = nnod

            wall_vars = convert_list_to_array(s[i+1:i+1+nrows]).reshape((-1,11))

            connectivity = np.genfromtxt(s[i+1+nrows:], dtype=float)

            df_conn = pd.DataFrame(connectivity, columns=['NodeC1', 'NodeC2'])

            break

    wall_vars = np.array(wall_vars)
    d2_df = pd.DataFrame(wall_vars[:,[0,1,2,6,7,8,9]], columns=['CoordinateX', 'CoordinateY', 'CoordinateZ', 'VX', 'VY', 'VZ', 'CP'])

    result = pd.concat([d2_df, df_conn], axis=1)
    d2_df = result.reindex(d2_df.index)

    index_list = list(reorder_coords(d2_df, return_indices=True))
    d2_df = d2_df.loc[index_list]
    return d2_df

def interpolate_to_uniform_grid(xi, yi, fvars):
    # Get raw xy values
    x, y = fvars[0], fvars[1]

    # Define uniform mesh
    Xi, Yi = np.meshgrid(xi, yi)

    # Initialize list to store interpolated field variables
    fvi = []

    for i in range(3):
        z = fvars[2+i]
        triang = tri.Triangulation(x, y)
        interpolator = tri.LinearTriInterpolator(triang, z)
        zi = interpolator(Xi, Yi)
        fvi.append(zi)

    return Xi, Yi, fvi

def plot_sdf(xy, sdf, sb, fname):
    x, y = xy[:,0].squeeze(), xy[:,1].squeeze()
    s, s_x, s_y = sdf['sdf'].squeeze(), sdf['sdf__x'].squeeze(), sdf['sdf__y'].squeeze()
    xb, yb = sb['x'], sb['y']

    fig, ax = plt.subplots(figsize=(18,10))

    ax.tricontour(x, y, s, levels=14, linewidths=0.5, colors='k')
    cntr2 = ax.tricontourf(x, y, s, levels=14, cmap="RdBu_r")

    ax.scatter(xb, yb, c='k')
    
    inds = np.linspace(0, x.shape[0], x.shape[0], dtype=int)
    idx = np.random.choice(inds, 100, replace=False)
    q = ax.quiver(x[idx], y[idx], s_x[idx], s_y[idx], pivot='mid', headwidth=1.0)

    ax.set_aspect('equal')
    #ax.set_xlim(-2.5, 2.5)
    #ax.set_ylim(-1.5, 1.5)
    plt.savefig(fname)
    plt.close()
    
    return fig, ax

def save_sdf(xy, sdf, fname, ngrid):
    x, y = xy[:,0].squeeze(), xy[:,1].squeeze()
    s, s_x, s_y = sdf['sdf'], sdf['sdf__x'], sdf['sdf__y']

    x = x.reshape((ngrid,ngrid))
    y = y.reshape((ngrid,ngrid))
    s = s.reshape((ngrid,ngrid))
    s_x = s_x.reshape((ngrid,ngrid))
    s_y = s_y.reshape((ngrid,ngrid))
    
    np.savez_compressed(fname, x=x, y=y, s=s, s_x=s_x, s_y=s_y)
    
    return None

def gen_sdf_data(df, sxy):
    # Load coordinates
    dat = df.values

    coordx = dat[:,0]
    coordy = dat[:,1]
    line = [(x,y) for x, y in zip(coordx, coordy)]

    # Create shape object
    geo = Polygon(line)

    # Number of sample points
    nd = int(300)
    nv = int(nd**2)

    # Boundary samples
    nb = nv
    sb = geo.sample_boundary(nb)

    # Uniform
    xy = {'x': sxy[:,0].reshape(-1,1), 'y': sxy[:,1].reshape(-1,1)}

    sdf = geo.sdf(xy, {}, compute_sdf_derivatives=True)

    return sdf, sb