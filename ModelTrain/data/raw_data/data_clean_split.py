import glob
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pickle
import sys
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Open aero conditions dictionary
with open('aero_dict.pkl', 'rb') as f:
    aero_dict = pickle.load(f)

# Get available cases and slice (all available field data) 
field_data_names = glob.glob('fields/*')
field_data_names = [s[14:-4] for s in field_data_names]
field_data_cases = list(set([s.split('_')[0] for s in field_data_names]))

# Subsample all data down into fraction
frac = 0.1 # Keep only this fraction of slices from all cases
n = 12     # Keep only this number of slices from each case

data_list = []
ndata = 0
for case in field_data_cases:
    if case=='0':
        continue

    data_names = [s for s in field_data_names if s.startswith(case+'_')]
    arr = np.array(data_names)
    np.random.shuffle(arr)
    data_names = list(data_names[:n])

    for dname in data_names:
        data_list.append(dname)

# Segment available datasets into train-validate-test split
arr = np.array(data_list)

# Shuffle the array
np.random.shuffle(arr)

# Calculate split points
n = len(arr)
first_split = int(32*16) #int(0.8 * n)
second_split = int(32*2)+first_split #int(0.9 * n)

# Split the array
train_split = arr[:first_split].tolist()
val_split = arr[first_split:second_split].tolist()
test_split = arr[second_split:].tolist()

Ntrain, Nval, Ntest = len(train_split), len(val_split), len(test_split)
    
# Calculate mean/var statistic of input and output data from training set only
X  = np.zeros((Ntrain, 256, 256))
Y  = np.zeros((Ntrain, 256, 256))
S  = np.zeros((Ntrain, 256, 256))
Ma = np.zeros((Ntrain, 256, 256))
Re = np.zeros((Ntrain, 256, 256))

P  = np.zeros((Ntrain, 256, 256))
VX = np.zeros((Ntrain, 256, 256))
VY = np.zeros((Ntrain, 256, 256))

nbatch = 32

print(Ntrain)
print(Ntrain//nbatch)
inputs = np.zeros((Ntrain//nbatch+1, nbatch, 5, 256, 256))
outputs = np.zeros((Ntrain//nbatch+1, nbatch, 3, 256, 256))

ibatch = 0
for i, dname in enumerate(train_split):

    # Input data
    data = np.load('sdfs/sdf_'+dname+'.npz')
    X[i,:,:] = data['x']
    Y[i,:,:] = data['y']
    S[i,:,:] = data['s']
    s_x = data['s_x']
    s_y = data['s_y']

    case = dname.split('_')[0]
    Ma[i,:,:] = np.ones((256, 256))*aero_dict[case]['mach']
    Re[i,:,:] = np.ones((256, 256))*aero_dict[case]['reynolds']

    # Output Data
    data = np.load('fields/fields_'+dname+'.npz')
    Xi = data['x']
    Yi = data['y']
    P[i,:,:] = data['p']
    VX[i,:,:] = data['vx']
    VY[i,:,:] = data['vy']

    # Get counters for storing inter/intra-bnatch indices
    ibatch = i//nbatch
    j = i % nbatch
    print(ibatch, j)
    inputs[ibatch, j, :, :, :] = np.stack([X[i,:,:], Y[i,:,:], S[i,:,:], Ma[i,:,:], Re[i,:,:]], axis=0)
    outputs[ibatch, j, :, :, :] = np.stack([P[i,:,:], VX[i,:,:], VY[i,:,:]], axis=0)

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

np.save('inputs_train.npy', inputs)
np.save('outputs_train.npy', outputs)
with open('stats_dict.pkl', 'wb') as f:
    pickle.dump(stats_dict, f)

fig, axs = plt.subplots(nrows=6, ncols=5, figsize=(12,16))

batch_idx = [0, 3, 9]
idx = [2, 13, 26]

for i in range(3):
    ii = 2*i
    for j in range(5):
        axs[ii,j].contourf(inputs[batch_idx[i],idx[i],j,:,:], levels=14, colors='k')
        cntr = axs[ii, j].contourf(inputs[batch_idx[i],idx[i],j,:,:], levels=14, cmap="RdBu_r")
        
        # Add colorbar above each subplot in first row
        divider = make_axes_locatable(axs[ii,j])
        cax = divider.append_axes('top', size='5%', pad=0.05)
        fig.colorbar(cntr, cax=cax, orientation='horizontal')
        cax.xaxis.set_ticks_position('top')  # Put ticks on top

    for j in range(3):
        axs[ii+1,j].contourf(outputs[batch_idx[i],idx[i],j,:,:], levels=14, colors='k')
        cntr = axs[ii+1, j].contourf(outputs[batch_idx[i],idx[i],j,:,:], levels=14, cmap="RdBu_r")

        # Add colorbar below each subplot in second row
        divider = make_axes_locatable(axs[ii+1, j])
        cax = divider.append_axes('bottom', size='5%', pad=0.05)
        fig.colorbar(cntr, cax=cax, orientation='horizontal')

for ax in axs.flat:
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.savefig('input_output.png')
plt.close()

del inputs, outputs, X, Y, S, Ma, Re, P, VX, VY

print('HERE')

Ma = np.zeros((Nval, 256, 256))
Re = np.zeros((Nval, 256, 256))
inputs_val = np.zeros((Nval//nbatch+1, nbatch, 5, 256, 256))
outputs_val = np.zeros((Nval//nbatch+1, nbatch, 3, 256, 256))

ibatch = 0
for i, dname in enumerate(val_split):
    # Get counters for storing inter/intra-bnatch indices
    ibatch = i//nbatch
    j = i % nbatch

    # Input data
    data = np.load('sdfs/sdf_'+dname+'.npz')
    case = dname.split('_')[0]
    Ma[i,:,:] = np.ones((256, 256))*aero_dict[case]['mach']
    Re[i,:,:] = np.ones((256, 256))*aero_dict[case]['reynolds']
    inputs_val[ibatch, j, :, :, :] = np.stack([data['x'], data['y'], data['s'], Ma[i,:,:], Re[i,:,:]], axis=0)

    # Output Data
    data = np.load('fields/fields_'+dname+'.npz')
    Xi = data['x']
    Yi = data['y']
    outputs_val[ibatch, j, :, :, :] = np.stack([data['p'], data['vx'], data['vy']], axis=0)

np.save('inputs_val.npy', inputs_val)
np.save('outputs_val.npy', outputs_val)
with open('data_counts.txt', 'w') as f:
  f.write('%d\n' % Ntrain)
  f.write('%d\n' % Nval)
  f.write('%d' % Ntest)
