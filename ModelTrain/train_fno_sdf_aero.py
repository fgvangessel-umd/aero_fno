# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hydra
from omegaconf import DictConfig
from math import ceil
import sys
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle

import torch
from torch.nn import MSELoss
from torch.optim import Adam, lr_scheduler

from modulus.models.fno import FNO
from modulus.distributed import DistributedManager
from modulus.utils import StaticCaptureTraining, StaticCaptureEvaluateNoGrad
from modulus.launch.utils import load_checkpoint, save_checkpoint
from modulus.launch.logging import PythonLogger, LaunchLogger
from modulus.launch.logging.mlflow import initialize_mlflow

from validator import GridValidator

from data_funcs import load_data, scale_data, to_device, to_cpu, loss_mask

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@hydra.main(version_base="1.3", config_path=".", config_name="config.yaml")
def aero_sdf_trainer(cfg: DictConfig) -> None:
    """Training for the Aero Prediction Task

    This script was adapted from the 2D Darcy FNO example provided from NVIDIA Modulus

    """
    DistributedManager.initialize()  # Only call this once in the entire script!
    dist = DistributedManager()  # call if required elsewhere

    # initialize monitoring
    log = PythonLogger(name="sdf_aero_fno")
    log.file_logging()
    initialize_mlflow(
        experiment_name=f"SDF_AERO_FNO",
        experiment_desc=f"training an FNO model for SDF to Aero mapping",
        run_name=f"SDF AERO FNO training",
        run_desc=f"training FNO for DAero",
        user_name="Frank",
        mode="offline",
    )
    LaunchLogger.initialize(use_mlflow=True)  # Modulus launch logger

    # define model, loss, optimiser, scheduler, data loader
    model = FNO(
        in_channels=cfg.arch.fno.in_channels,
        out_channels=cfg.arch.decoder.out_features,
        decoder_layers=cfg.arch.decoder.layers,
        decoder_layer_size=cfg.arch.decoder.layer_size,
        dimension=cfg.arch.fno.dimension,
        latent_channels=cfg.arch.fno.latent_channels,
        num_fno_layers=cfg.arch.fno.fno_layers,
        num_fno_modes=cfg.arch.fno.fno_modes,
        padding=cfg.arch.fno.padding,
    ).to(dist.device)

    loss_fun = MSELoss(reduction="mean")
    optimizer = Adam(model.parameters(), lr=cfg.scheduler.initial_lr)
    scheduler = lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: cfg.scheduler.decay_rate**step
    )

    data_dir = cfg.data.data_dir

    # Load mean and variance scaling parameters
    with open(data_dir+cfg.data.files.norm, 'rb') as f: 
        stats_dict = pickle.load(f)

    # Read in data counts to avoid traininig on padded tensor values
    with open(data_dir+cfg.data.files.counts, 'r') as f:
        Ntrain = int(f.readline().strip())
        Nval = int(f.readline().strip())
        Ntest = int(f.readline().strip())

    # Load input and output training data
    data_dir = cfg.data.data_dir
    inputs_train, outputs_train = load_data(data_dir+cfg.data.files.input_train, data_dir+cfg.data.files.output_train)
    inputs_val,   outputs_val   = load_data(data_dir+cfg.data.files.input_val, data_dir+cfg.data.files.output_val)

    # Create loss masks
    mask_train = loss_mask(inputs_train)
    mask_val   = loss_mask(inputs_val)
    mask_train, mask_val = to_device(mask_train, mask_val, dist.device)

    # Standard scale training data
    inputs_train, outputs_train = scale_data(inputs_train, outputs_train, stats_dict)
    inputs_val, outputs_val = scale_data(inputs_val, outputs_val, stats_dict)

    # Move data to device
    inputs_train, outputs_train = to_device(inputs_train, outputs_train, dist.device)
    inputs_val, outputs_val = to_device(inputs_val, outputs_val, dist.device)

    validator = GridValidator(loss_fun=MSELoss(reduction="mean"))

    ckpt_args = {
        "path": f"./checkpoints",
        "optimizer": optimizer,
        "scheduler": scheduler,
        "models": model,
    }
    loaded_epoch = load_checkpoint(device=dist.device, **ckpt_args)

    # calculate steps per pseudo epoch
    steps_per_epoch = ceil(
        Ntrain / cfg.training.batch_size
    )
    validation_iters = ceil(Nval / cfg.training.batch_size)
    log_args = {
        "name_space": "train",
        "num_mini_batch": steps_per_epoch,
        "epoch_alert_freq": 1,
    }
    if Ntrain % cfg.training.batch_size != 0:
        log.warning(
            f"increased pseudo_epoch_sample_size to multiple of \
                      batch size: {steps_per_epoch*cfg.training.batch_size}"
        )
    if Nval % cfg.training.batch_size != 0:
        log.warning(
            f"increased validation sample size to multiple of \
                      batch size: {validation_iters*cfg.training.batch_size}"
        )

    # define forward passes for training and inference
    @StaticCaptureTraining(
        model=model, optim=optimizer, logger=log, use_amp=False, use_graphs=False
    )
    def forward_train(invars, target, mask):
        pred = model(invars)
        loss = loss_fun(pred*mask, target*mask)
        return loss

    @StaticCaptureEvaluateNoGrad(
        model=model, logger=log, use_amp=False, use_graphs=False
    )
    def forward_eval(invars):
        return model(invars)

    if loaded_epoch == 0:
        log.success("Training started...")
    else:
        log.warning(f"Resuming training from epoch {loaded_epoch+1}.")

    # Usage
    num_params = count_parameters(model)
    print(f'The model has {num_params:,} trainable parameters')

    for epoch in range(cfg.training.max_epochs):
        for ibatch in range(steps_per_epoch):
            loss = forward_train(inputs_train[ibatch,:,:,:,:], \
                                 outputs_train[ibatch,:,:,:,:], \
                                 mask_train[ibatch, :, :, :, :])

        if epoch%cfg.validation.validation_epochs==0:
            save_checkpoint(**ckpt_args, epoch=epoch)
            print("loss: %4.3e"%loss.detach())

            with LaunchLogger("valid", epoch=epoch) as logger:
                total_loss = 0.0
                for i in range(validation_iters):
                    mask = mask_val[i, :, :, :, :]
                    #val_loss = validator.compare(
                    #    inputs_val[i,:,:,:,:],
                    #    outputs_val[i,:,:,:,:]*mask,
                    #    forward_eval(inputs_val[i,:,:,:,:]*mask),
                    #    epoch,
                    #    logger,
                    #)
                    val_loss = forward_train(inputs_val[i,:,:,:,:], \
                                             outputs_val[i,:,:,:,:], \
                                             mask_val[i, :, :, :, :]).detach()
                    total_loss += val_loss
                logger.log_epoch({"Validation error": total_loss / validation_iters})

    ###
    ###
    ###
    save_checkpoint(**ckpt_args, epoch=epoch)

    # Visualize predictions on validation dataset
    for ibatch in range(2):

        invars = inputs_val[ibatch,:,:,:,:]
        pred = model(invars)
        target = outputs_val[ibatch,:,:,:,:]

        masked_pred = pred*mask_val[ibatch, :, :, :, :]
        masked_target = target*mask_val[ibatch, :, :, :, :]

        for iplot in range(32):
            fig, axs = plt.subplots(nrows=3, ncols=5, figsize=(18,12))

            mp, mt = to_cpu(masked_pred[iplot,:,:,:], masked_target[iplot,:,:,:])
            m, inv = to_cpu(mask_val[ibatch, :, :, :, :], invars)

            for i in range(5):
                axs[0,i].contourf(inv[iplot, i, :, :]*m[iplot, 0, :, :], levels=14, colors='k')
                cntr = axs[0, i].contourf(inv[iplot, i, :, :]*m[iplot, 0, :, :], levels=14, cmap="RdBu_r")
                
                # Add colorbar above each subplot in first row
                #divider = make_axes_locatable(axs[0,i])
                #cax = divider.append_axes('top', size='5%', pad=0.05)
                #fig.colorbar(cntr, cax=cax, orientation='horizontal')
                #cax.xaxis.set_ticks_position('top')  # Put ticks on top

            for i in range(3):
                axs[1,i].contourf(mt[i,:,:], levels=14, colors='k')
                cntr = axs[1, i].contourf(mt[i,:,:], levels=14, cmap="RdBu_r")

                # Add colorbar below each subplot in second row
                #divider = make_axes_locatable(axs[1, i])
                #cax = divider.append_axes('bottom', size='5%', pad=0.05)
                #fig.colorbar(cntr, cax=cax, orientation='horizontal')

            for i in range(3):
                axs[2,i].contourf(mp[i,:,:], levels=14, colors='k')
                cntr = axs[2,i].contourf(mp[i,:,:], levels=14, cmap="RdBu_r")

                # Add colorbar below each subplot in second row
                #divider = make_axes_locatable(axs[2, i])
                #cax = divider.append_axes('bottom', size='5%', pad=0.05)
                #fig.colorbar(cntr, cax=cax, orientation='horizontal')


            for ax in axs.flat:
                ax.set_xticks([])
                ax.set_yticks([])

            plt.savefig('preds_val_'+str(ibatch)+'_'+str(iplot)+'.png')
            plt.close()

if __name__ == "__main__":
    aero_sdf_trainer()
