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

import torch
from torch.nn import MSELoss
from torch.optim import Adam, lr_scheduler

from modulus.models.fno import FNO
from modulus.datapipes.benchmarks.darcy import Darcy2D
from modulus.distributed import DistributedManager
from modulus.utils import StaticCaptureTraining, StaticCaptureEvaluateNoGrad
from modulus.launch.utils import load_checkpoint, save_checkpoint
from modulus.launch.logging import PythonLogger, LaunchLogger
from modulus.launch.logging.mlflow import initialize_mlflow

from validator import GridValidator

@hydra.main(version_base="1.3", config_path=".", config_name="config.yaml")
def aero_sdf_trainer(cfg: DictConfig) -> None:
    """Training for the 2D Darcy flow benchmark problem.

    This training script demonstrates how to set up a data-driven model for a 2D Darcy flow
    using Fourier Neural Operators (FNO) and acts as a benchmark for this type of operator.
    Training data is generated in-situ via the Darcy2D data loader from Modulus. Darcy2D
    continuously generates data previously unseen by the model, i.e. the model is trained
    over a single epoch of a training set consisting of
    (cfg.training.max_pseudo_epochs*cfg.training.pseudo_epoch_sample_size) unique samples.
    Pseudo_epochs were introduced to leverage the LaunchLogger and its MLFlow integration.
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
    
    # Load Aero FNO Data
    ndata = 20
    datasets = ['0_{0:03}'.format(i) for i in range(ndata)]
    
    pressure = np.zeros((ndata,cfg.arch.decoder.out_features,cfg.training.resolution, cfg.training.resolution))
    sdf = np.zeros((ndata,cfg.arch.decoder.out_features,cfg.training.resolution, cfg.training.resolution))
    
    for i, dataset in enumerate(datasets):
        data = np.load('data/field_data/'+dataset+'_fields.npz')
        Xi = data['x']
        Yi = data['y']
        p = data['p']
        vx = data['vx']
        vy = data['vy']

        Xi = Xi[:cfg.training.resolution,:cfg.training.resolution]
        Yi = Yi[:cfg.training.resolution,:cfg.training.resolution]
        p  = p[:cfg.training.resolution,:cfg.training.resolution]

        pressure[i,0,:,:] = p

        data = np.load('data/sdf_data/sdf_'+dataset+'.npz')
        x = data['x']
        y = data['y']
        s = data['s']
        s_x = data['s_x']
        s_y = data['s_y']

        x = x.reshape((264,264))
        y = y.reshape((264,264))
        s = s.reshape((264,264))

        s  = s[:cfg.training.resolution,:cfg.training.resolution]

        sdf[i,0,:,:] = s

    # Standard Scale datasets
    sdf = (sdf - np.mean(sdf)) /np.std(sdf)
    pressure = (pressure - np.mean(pressure)) / np.std(pressure)

    fig, axs = plt.subplots(nrows=10, ncols=4, figsize=(16,24))

    for i in range(ndata):
        if i<10:
            axs[i,0].contourf(Xi, Yi, sdf[i,0,:,:], levels=14, linewidths=0.5, colors='k')
            cntr2 = axs[i,0].contourf(Xi, Yi, sdf[i,0,:,:], levels=14, cmap="RdBu_r")
            axs[i,1].contourf(Xi, Yi, pressure[i,0,:,:], levels=14, linewidths=0.5, colors='k')
            cntr2 = axs[i,1].contourf(Xi, Yi, pressure[i,0,:,:], levels=14, cmap="RdBu_r")
        else:
            j = i-10
            axs[j,2].contourf(Xi, Yi, sdf[i,0,:,:], levels=14, linewidths=0.5, colors='k')
            cntr2 = axs[j,2].contourf(Xi, Yi, sdf[i,0,:,:], levels=14, cmap="RdBu_r")
            axs[j,3].contourf(Xi, Yi, pressure[i,0,:,:], levels=14, linewidths=0.5, colors='k')
            cntr2 = axs[j,3].contourf(Xi, Yi, pressure[i,0,:,:], levels=14, cmap="RdBu_r")

    plt.savefig('data.png')

    # Create single training batch
    batch = {'sdf': torch.tensor(sdf, dtype=torch.float).to(dist.device), 'pressure': torch.tensor(pressure, dtype=torch.float).to(dist.device)}
    print(batch['sdf'].type())

    validator = GridValidator(loss_fun=MSELoss(reduction="mean"))

    ckpt_args = {
        "path": f"./checkpoints",
        "optimizer": optimizer,
        "scheduler": scheduler,
        "models": model,
    }
    loaded_pseudo_epoch = load_checkpoint(device=dist.device, **ckpt_args)

    # calculate steps per pseudo epoch
    steps_per_pseudo_epoch = ceil(
        cfg.training.pseudo_epoch_sample_size / cfg.training.batch_size
    )
    validation_iters = ceil(cfg.validation.sample_size / cfg.training.batch_size)
    log_args = {
        "name_space": "train",
        "num_mini_batch": steps_per_pseudo_epoch,
        "epoch_alert_freq": 1,
    }
    if cfg.training.pseudo_epoch_sample_size % cfg.training.batch_size != 0:
        log.warning(
            f"increased pseudo_epoch_sample_size to multiple of \
                      batch size: {steps_per_pseudo_epoch*cfg.training.batch_size}"
        )
    if cfg.validation.sample_size % cfg.training.batch_size != 0:
        log.warning(
            f"increased validation sample size to multiple of \
                      batch size: {validation_iters*cfg.training.batch_size}"
        )

    # define forward passes for training and inference
    @StaticCaptureTraining(
        model=model, optim=optimizer, logger=log, use_amp=False, use_graphs=False
    )
    def forward_train(invars, target):
        pred = model(invars)
        loss = loss_fun(pred, target)
        return loss

    @StaticCaptureEvaluateNoGrad(
        model=model, logger=log, use_amp=False, use_graphs=False
    )
    def forward_eval(invars):
        return model(invars)

    if loaded_pseudo_epoch == 0:
        log.success("Training started...")
    else:
        log.warning(f"Resuming training from pseudo epoch {loaded_pseudo_epoch+1}.")

    #for pseudo_epoch in range(
    #    max(1, loaded_pseudo_epoch + 1), cfg.training.max_pseudo_epochs + 1
    #):
    for pseudo_epoch in range(1):
        # Wrap epoch in launch logger for console / MLFlow logs
        with LaunchLogger(**log_args, epoch=pseudo_epoch) as logger:
            #for _, batch in zip(range(steps_per_pseudo_epoch), dataloader):
            for _ in range(1000):
                loss = forward_train(batch["sdf"], batch["pressure"])
                print("loss: %4.3e"%loss.detach())
                logger.log_minibatch({"loss": loss.detach()})
            logger.log_epoch({"Learning Rate": optimizer.param_groups[0]["lr"]})


        # save checkpoint
        #if pseudo_epoch % cfg.training.rec_results_freq == 0:
        if True:
            save_checkpoint(**ckpt_args, epoch=pseudo_epoch)

        # validation step
        #if pseudo_epoch % cfg.validation.validation_pseudo_epochs == 0:
        if True:
            with LaunchLogger("valid", epoch=pseudo_epoch) as logger:
                total_loss = 0.0
                #for _, batch in zip(range(validation_iters), dataloader):
                for i in range(1):
                    val_loss = validator.compare(
                        batch["sdf"],
                        batch["pressure"],
                        forward_eval(batch["sdf"]),
                        pseudo_epoch,
                        logger,
                    )
                    total_loss += val_loss
                logger.log_epoch({"Validation error": total_loss / validation_iters})

        # update learning rate
        if pseudo_epoch % cfg.scheduler.decay_pseudo_epochs == 0:
            scheduler.step()

    save_checkpoint(**ckpt_args, epoch=cfg.training.max_pseudo_epochs)
    log.success("Training completed *yay*")

    # Load validation dataset
    datasets = ['0_{0:03}'.format(i) for i in range(20,25)]
    ndata = len(datasets)

    pressure_val = np.zeros((ndata,cfg.arch.decoder.out_features,cfg.training.resolution, cfg.training.resolution))
    sdf_val = np.zeros((ndata,cfg.arch.decoder.out_features,cfg.training.resolution, cfg.training.resolution))
    
    for i, dataset in enumerate(datasets):
        data = np.load('data/field_data/'+dataset+'_fields.npz')
        Xi = data['x']
        Yi = data['y']
        p = data['p']
        vx = data['vx']
        vy = data['vy']

        Xi = Xi[:cfg.training.resolution,:cfg.training.resolution]
        Yi = Yi[:cfg.training.resolution,:cfg.training.resolution]
        p  = p[:cfg.training.resolution,:cfg.training.resolution]

        pressure_val[i,0,:,:] = p

        data = np.load('data/sdf_data/sdf_'+dataset+'.npz')
        x = data['x']
        y = data['y']
        s = data['s']
        s_x = data['s_x']
        s_y = data['s_y']

        x = x.reshape((264,264))
        y = y.reshape((264,264))
        s = s.reshape((264,264))

        s = s[:cfg.training.resolution,:cfg.training.resolution]

        sdf_val[i,0,:,:] = s

    # Standard Scale datasets
    sdf_val = (sdf_val - np.mean(sdf)) /np.std(sdf)
    pressure_val = (pressure_val - np.mean(pressure)) / np.std(pressure)

    # Create single validation training batch
    batch_val = {'sdf': torch.tensor(sdf_val, dtype=torch.float).to(dist.device), 'pressure': torch.tensor(pressure_val, dtype=torch.float).to(dist.device)}

    preds_val = forward_eval(batch_val["sdf"]).detach().cpu().numpy()

    fig, axs = plt.subplots(nrows=ndata, ncols=4, figsize=(30,24))

    for i in range(ndata):
        axs[i,0].contourf(Xi, Yi, sdf_val[i,0,:,:], levels=14, linewidths=0.5, colors='k')
        cntr2 = axs[i,0].contourf(Xi, Yi, sdf_val[i,0,:,:], levels=14, cmap="RdBu_r")

        axs[i,1].contourf(Xi, Yi, preds_val[i,0,:,:], levels=14, linewidths=0.5, colors='k')
        cntr2 = axs[i,1].contourf(Xi, Yi, preds_val[i,0,:,:], levels=14, cmap="RdBu_r")

        axs[i,2].contourf(Xi, Yi, pressure_val[i,0,:,:], levels=14, linewidths=0.5, colors='k')
        cntr2 = axs[i,2].contourf(Xi, Yi, pressure_val[i,0,:,:], levels=14, cmap="RdBu_r")

        axs[i,3].contourf(Xi, Yi, pressure_val[i,0,:,:]-preds_val[i,0,:,:], levels=14, linewidths=0.5, colors='k')
        cntr2 = axs[i,3].contourf(Xi, Yi, pressure_val[i,0,:,:]-preds_val[i,0,:,:], levels=14, cmap="RdBu_r")

    plt.savefig('preds_val.png')

    #Trianing prediction
    preds = forward_eval(batch["sdf"]).detach().cpu().numpy()

    fig, axs = plt.subplots(nrows=6, ncols=4, figsize=(30,24))

    for i in range(6):
        axs[i,0].contourf(Xi, Yi, sdf[i,0,:,:], levels=14, linewidths=0.5, colors='k')
        cntr2 = axs[i,0].contourf(Xi, Yi, sdf[i,0,:,:], levels=14, cmap="RdBu_r")

        axs[i,1].contourf(Xi, Yi, preds[i,0,:,:], levels=14, linewidths=0.5, colors='k')
        cntr2 = axs[i,1].contourf(Xi, Yi, preds[i,0,:,:], levels=14, cmap="RdBu_r")

        axs[i,2].contourf(Xi, Yi, pressure[i,0,:,:], levels=14, linewidths=0.5, colors='k')
        cntr2 = axs[i,2].contourf(Xi, Yi, pressure[i,0,:,:], levels=14, cmap="RdBu_r")

        axs[i,3].contourf(Xi, Yi, pressure[i,0,:,:]-preds[i,0,:,:], levels=14, linewidths=0.5, colors='k')
        cntr2 = axs[i,3].contourf(Xi, Yi, pressure[i,0,:,:]-preds[i,0,:,:], levels=14, cmap="RdBu_r")

    plt.savefig('preds.png')

if __name__ == "__main__":
    aero_sdf_trainer()
