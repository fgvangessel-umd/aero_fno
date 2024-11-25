import glob
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pickle
import sys, os
from mpl_toolkits.axes_grid1 import make_axes_locatable
from itertools import chain
import gc

def process_chunk(chunk_data, chunk_start, aero_dict, chunk_size):
    """Process a chunk of data and return inputs and outputs arrays"""
    chunk_inputs = np.zeros((chunk_size, 5, 256, 256), dtype=np.float32)
    chunk_outputs = np.zeros((chunk_size, 3, 256, 256), dtype=np.float32)
    
    for i, dname in enumerate(chunk_data):
        try:
            # Input data
            with np.load('sdfs/sdf_'+dname+'.npz') as data:
                case = dname.split('_')[0]
                
                # Copy data directly into arrays
                chunk_inputs[i,0] = data['x']
                chunk_inputs[i,1] = data['y']
                chunk_inputs[i,2] = data['s']
                chunk_inputs[i,3] = np.ones((256, 256), dtype=np.float32) * aero_dict[case]['mach']
                chunk_inputs[i,4] = np.ones((256, 256), dtype=np.float32) * aero_dict[case]['reynolds']

            # Output data
            with np.load('fields/fields_'+dname+'.npz') as data:
                chunk_outputs[i,0] = data['p']
                chunk_outputs[i,1] = data['vx']
                chunk_outputs[i,2] = data['vy']
                
        except Exception as e:
            print(f"Error processing {dname}: {str(e)}")
            return None, None
            
        if i % 100 == 0:
            print(f"Processing item {chunk_start + i}")
            gc.collect()
    
    return chunk_inputs, chunk_outputs

# Open aero conditions dictionary
with open('aero_dict.pkl', 'rb') as f:
    aero_dict = pickle.load(f)

# Load dictionary of unique cases
with open('unique_slices_dict.pkl', 'rb') as handle:
    unique_slices_dict = pickle.load(handle)

# Create list of all unique data
data_list = []
ndata = 0
for case, slices in unique_slices_dict.items():
    if case=='0':
        continue
    data_names = [case+'_'+slc for slc in slices]
    data_list.append(data_names)
data_list = list(chain(*data_list))

# Segment available datasets into train-validate-test split
arr = np.array(data_list)

# Shuffle the array
np.random.shuffle(arr)

# Calculate split points
n = len(arr)
first_split = int(0.8 * n)
second_split = int(0.9 * n)

# Split the array
train_split = arr[:first_split].tolist()
val_split = arr[first_split:second_split].tolist()
test_split = arr[second_split:].tolist()

Ntrain, Nval, Ntest = len(train_split), len(val_split), len(test_split)
    
# Process data in chunks
chunk_size = 500  # Adjust this based on available memory
num_chunks = (Ntrain + chunk_size - 1) // chunk_size

# Create temporary directory for chunks if it doesn't exist
os.makedirs('temp_chunks', exist_ok=True)

# Process each chunk
for chunk_idx in range(num_chunks):
    print(f"\nProcessing chunk {chunk_idx + 1}/{num_chunks}")
    start_idx = chunk_idx * chunk_size
    end_idx = min((chunk_idx + 1) * chunk_size, Ntrain)
    current_chunk_size = end_idx - start_idx
    
    # Get current chunk data
    chunk_data = train_split[start_idx:end_idx]
    
    # Process chunk
    chunk_inputs, chunk_outputs = process_chunk(chunk_data, start_idx, aero_dict, current_chunk_size)
    
    if chunk_inputs is None:
        print(f"Error processing chunk {chunk_idx + 1}")
        continue
        
    # Save chunk to disk
    np.savez_compressed(
        f'temp_chunks/chunk_{chunk_idx}.npz',
        inputs=chunk_inputs,
        outputs=chunk_outputs
    )
    
    # Clear memory
    del chunk_inputs, chunk_outputs
    gc.collect()

# Combine chunks into final arrays
print("\nCombining chunks into final arrays...")
inputs = np.zeros((Ntrain, 5, 256, 256), dtype=np.float32)
outputs = np.zeros((Ntrain, 3, 256, 256), dtype=np.float32)

for chunk_idx in range(num_chunks):
    start_idx = chunk_idx * chunk_size
    end_idx = min((chunk_idx + 1) * chunk_size, Ntrain)
    
    # Load chunk
    chunk_data = np.load(f'temp_chunks/chunk_{chunk_idx}.npz')
    inputs[start_idx:end_idx] = chunk_data['inputs']
    outputs[start_idx:end_idx] = chunk_data['outputs']
    
    # Clean up
    del chunk_data
    gc.collect()

# Save final arrays
print("Saving final arrays...")
np.savez_compressed('final_data.npz', inputs=inputs, outputs=outputs)

# Clean up temporary files
print("Cleaning up temporary files...")
for chunk_idx in range(num_chunks):
    os.remove(f'temp_chunks/chunk_{chunk_idx}.npz')
os.rmdir('temp_chunks')

print("Processing complete!")

sys.exit('DEBUG')

'''
stats_dict = {'x': [X.mean(), X.std()],\
              'y': [Y.mean(), Y.std()],\
              's': [S.mean(), S.std()],\
              'ma': [Ma.mean(), Ma.std()],\
              're': [Re.mean(), Re.std()],\
              'P': [S.mean(), S.std()],\
              'VX': [S.mean(), S.std()],\
              'VY': [S.mean(), S.std()],\
             }

print(inputs.shape)
print(outputs.shape)
sys.exit('DEBUG')

np.save('inputs_train.npy', inputs)
np.save('outputs_train.npy', outputs)
with open('stats_dict.pkl', 'wb') as f:
    pickle.dump(stats_dict, f)

'''