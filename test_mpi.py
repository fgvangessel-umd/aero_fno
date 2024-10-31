import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib.tri as tri
from reorder import reorder_coords
import glob
#from functions import *
from mpi4py import MPI

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

def process_single_file(fname):
    with open(fname, 'r') as f:
        s = f.readlines()

    case = fname.split('_')[2].split('/')[0]
    slce = fname.split('_')[3]
    print(case, slce)
    
    d2_df = read_slice_data(s)

    sxy = get_sample_points(case, slce)

    return d2_df, sxy

def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print(comm)
        print(rank)
        print(size)

    # Root process (rank 0) gets the list of files
    if rank == 0:
        # Get list of all files to process
        cases = ['data/slice_data/case_0']
        data_dir = cases[0]  # Replace with your directory
        all_files = sorted(glob.glob(os.path.join(data_dir, "fc*_slices.dat")))  # Adjust file pattern
        
        # Calculate workload distribution
        files_per_process = len(all_files) // size
        remainder = len(all_files) % size
        
        # Distribute files among processes
        distributions = []
        start = 0
        for i in range(size):
            # Add one more file to some processes if there's a remainder
            extra = 1 if i < remainder else 0
            end = start + files_per_process + extra
            distributions.append(all_files[start:end])
            start = end
    else:
        distributions = None
    
    # Scatter the file lists to all processes
    my_files = comm.scatter(distributions, root=0)
    
    '''
    # Process files assigned to this rank
    my_results = []
    for filename in my_files:
        try:
            result = process_single_file(filename)
            my_results.append((filename, result))
        except Exception as e:
            print(f"Process {rank} failed to process {filename}: {str(e)}")
    
    # Gather all results back to root process
    all_results = comm.gather(my_results, root=0)
    
    # Root process combines and saves results
    if rank == 0:
        combined_results = {}
        for process_results in all_results:
            for filename, result in process_results:
                combined_results[filename] = result
        
        # Save or further process the combined results
        np.save("combined_results.npy", combined_results)
        print(f"Processed {len(combined_results)} files successfully")
    '''
    
    
    '''
    sdf = gen_sdf_data(d2_df, sxy)

    # Plot
    fname = f'data/sdf_data/sdf'+case+'_'+slce+'.png'
    plot_sdf(xy, sdf, sb, fname)

    # Save data
    fname = f'data/sdf_data/sdf'+case+'_'+slce+'.npz'
    save_sdf(xy, sdf, sb, fname)
    '''

if __name__ == "__main__":
    main()