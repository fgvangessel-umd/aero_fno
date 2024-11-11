import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from reorder import reorder_coords
import glob
from functions import *
from multiprocessing import Process, cpu_count, Pool
import time

def flatten(xss):
    return [x for xs in xss for x in xs]

def read_case_slice_file(fname):
    print(fname)
    with open(fname, 'r') as f:
        s = f.readlines()

    case = fname.split('_')[2].split('/')[0]
    slce = fname.split('_')[3]

    d2_df = read_slice_data(s)

    sxy = get_sample_points(case, slce)

    ngrid = int(np.sqrt(sxy.shape[0]))
        
    sdf, sb = gen_sdf_data(d2_df, sxy)

    # Plot
    fname = f'tmp_dir/sdf_'+case+'_'+slce+'.png'
    plot_sdf(sxy, sdf, sb, fname)

    # Save data
    fname = f'tmp_dir/sdf_'+case+'_'+slce+'.npz'
    save_sdf(sxy, sdf, fname, ngrid)

    return None

def get_sample_points(case, slce):
    '''
    Get point in domain at which we sample the SDF
    '''
    fname = 'interpolated_field_data/'+case+'_'+slce+'_fields.npz'
    data = np.load(fname)
    Xi = data['x']
    Yi = data['y']

    sxy = np.concatenate((Xi.reshape(-1,1), Yi.reshape(-1,1)), axis=1)
    return sxy


if __name__ == '__main__':
    
    # get the number of cpu cores
    num_cores = cpu_count()
    # report details
    print(num_cores)

    # GEt names of all files that need to be converted
    fnames = []
    cases = glob.glob('data/slice_data/case*')
    for d in cases:
        slices = glob.glob(d+'/fc*_slices.dat')
        fnames.append(slices)

    fnames = flatten(fnames)

    fnames = fnames[:8]

    print(len(fnames))

    pool = Pool(8)
    start_time = time.perf_counter()
    #processes = [pool.apply_async(read_case_slice_file, args=(fname,)) for fname in fnames]
    #result = [p.get() for p in processes]
    pool.map(read_case_slice_file, fnames)
    pool.close()
    finish_time = time.perf_counter()
    print(f"Program finished in {finish_time-start_time} seconds")

    '''
    processes = []

    # Creates 10 processes then starts them
    for i in range(10):
        p = Process(target = read_case_slice_file)
        p.start()
        processes.append(p)
    
    # Joins all the processes 
    for p in processes:
        p.join()

    finish_time = time.perf_counter()

    print(f"Program finished in {finish_time-start_time} seconds")
    '''
