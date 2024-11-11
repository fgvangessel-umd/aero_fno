import hydra
from omegaconf import DictConfig
import numpy as np
import sys
import glob
from functions import *

def flatten(xss):
    return [x for xs in xss for x in xs]

@hydra.main(version_base="1.3", config_path=".", config_name="config.yaml")
def aero_sdf_gen(cfg: DictConfig) -> None:

    # Read grid config
    ngrid = cfg.grid.ngrid
    ngridx = ngrid
    ngridy = ngrid
    xmin, xmax = cfg.grid.xmin, cfg.grid.xmax
    ymin, ymax = cfg.grid.ymin, cfg.grid.ymax

    # Generate grid
    xi = np.linspace(xmin, xmax, ngridx)
    yi = np.linspace(ymin, ymax, ngridy)
    Xi, Yi = np.meshgrid(xi, yi)
    sxy = np.concatenate((Xi.reshape(-1,1), Yi.reshape(-1,1)), axis=1)

    # Get names of all files that need to be converted
    fnames = []
    cases = glob.glob('data/slice_data/case*')
    for d in cases:
        slices = glob.glob(d+'/fc*_slices.dat')
        fnames.append(slices)
    fnames = flatten(fnames)

    for fname in fnames:
        print(fname)
        with open(fname, 'r') as f:
            s = f.readlines()

        case = fname.split('_')[2].split('/')[0]
        slce = fname.split('_')[3]
        print(case, slce)
        
        d2_df = read_slice_data(s)

        sdf, sb = gen_sdf_data(d2_df, sxy)

        # Plot
        fname = f'data/sdf_data/sdf_'+case+'_'+slce+'.png'
        plot_sdf(sxy, sdf, sb, fname)

        # Save data
        fname = f'data/sdf_data/sdf_'+case+'_'+slce+'.npz'
        save_sdf(sxy, sdf, fname, ngrid)

if __name__ == "__main__":
    aero_sdf_gen()
