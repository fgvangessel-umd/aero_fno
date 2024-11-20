import numpy as np
import torch

def load_data(input_file, output_file):
    inputs = np.load(input_file)
    outputs = np.load(output_file)
    return inputs, outputs

def scale_data(inputs, outputs, stats_dict):
    # Standard Scale datasets using precomputed statistics
    inputs[:,:,0,:,:] = (inputs[:,:,0,:,:] - stats_dict['x'][0]) / stats_dict['x'][1]
    inputs[:,:,1,:,:] = (inputs[:,:,1,:,:] - stats_dict['y'][0]) / stats_dict['y'][1]
    inputs[:,:,2,:,:] = (inputs[:,:,2,:,:] - stats_dict['s'][0]) / stats_dict['s'][1]
    inputs[:,:,3,:,:] = (inputs[:,:,3,:,:] - stats_dict['ma'][0]) / stats_dict['ma'][1]
    inputs[:,:,4,:,:] = (inputs[:,:,4,:,:] - stats_dict['re'][0]) / stats_dict['re'][1]

    outputs[:,:,0,:,:] = (outputs[:,:,0,:,:] - stats_dict['P'][0]) / stats_dict['P'][1]
    outputs[:,:,1,:,:] = (outputs[:,:,1,:,:] - stats_dict['VX'][0]) / stats_dict['VX'][1]
    outputs[:,:,2,:,:] = (outputs[:,:,2,:,:] - stats_dict['VY'][0]) / stats_dict['VY'][1]

    return inputs, outputs

def to_device(inputs, outputs, device):
    inputs = torch.tensor(inputs, dtype=torch.float).to(device)
    outputs = torch.tensor(outputs, dtype=torch.float).to(device)
    return inputs, outputs

def to_cpu(inputs, outputs):
    if inputs.requires_grad:
        return inputs.detach().cpu().numpy(), outputs.detach().cpu().numpy() 
    else:
        return inputs.cpu().numpy(), outputs.cpu().numpy() 
        

def loss_mask(inputs):
    # Create masks to mask  loss function for terms internal to the airfoil
    dim1, dim2, dim3, dim4, dim5 = inputs.shape
    mask = np.zeros((dim1, dim2, 3, dim4, dim5))
    
    for ibatch in range(dim1):
        for idata in range(dim2):
            for ivar in range(3):
                mask[ibatch, idata, ivar, :, :] = (inputs[ibatch,idata,2,:,:] < 0).astype(np.float64)

    return mask