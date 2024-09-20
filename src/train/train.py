'''
the core training entry, simply run this
u can also pass config in cmd line, e.g.:
    `python train.py training_config.debug=true`
'''

import os
import rootutils
import numpy as np
from omegaconf import OmegaConf

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import Subset
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from dataset import SubImageDataset
from plmodules import DetermModule
from callbacks import WellPreparedPause


if __name__ == '__main__':
    base = rootutils.find_root(search_from=__file__, indicator=".project-root")
    config = OmegaConf.load('config.yaml')
    config_cli = OmegaConf.from_cli() 
    # update layer-2 nested config
    for key, nested_config in config.items():
        if key in config_cli:
            nested_config.update(config_cli[key])
    
    pl.seed_everything(config.training_config.seed)
    
    trainset = SubImageDataset(
        year_lim = [2008, 2016], 
        **config.dataset_config, 
    )
    valset = SubImageDataset(
        year_sel = [2017], 
        **config.dataset_config
    )
    
    if config.training_config.debug:
        trainset = Subset(trainset, np.arange(0, 256))
        valset = Subset(valset, np.arange(0, 64))
    elif config.dataset_config.get('subset_proportion'):
        prop = config.dataset_config.subset_proportion
        trainset = Subset(trainset, np.random.choice(len(trainset), int(len(trainset)*prop), replace=False))
        valset = Subset(valset, np.random.choice(len(valset), int(len(valset)*prop), replace=False))
        
    trainloader = DataLoader(trainset, shuffle=True, **config.dataloader_config)
    valloader = DataLoader(valset, shuffle=True, **config.dataloader_config)
    
    '''
    assets for trainer
    '''
    logger_path = os.path.join(base, config.training_config.log_dir, 'wandb_logs')
    os.makedirs(logger_path, exist_ok=True)
    wandb_logger = WandbLogger(
        save_dir=logger_path,
        project=config.training_config.project_name, 
        name=config.training_config.model_name,
        config=OmegaConf.to_container(config),
    )
    
    manual_path = os.path.join(base, config.training_config.log_dir, 'manual_logs', 
                f'{config.training_config.model_name}-{config.training_config.model_version}')
    os.makedirs(manual_path, exist_ok=True)
    pause_callback = WellPreparedPause(
        save_dir=manual_path,
        save_config=False, 
    ) # save ckpt on pause
    everystep_ckpt_callback = ModelCheckpoint(
        dirpath=manual_path,
        every_n_epochs=100,
        filename='ckpt-{epoch:02d}',
        save_last='link',
        save_top_k=-1,
    ) # save ckpt every epoch
    best_ckpt_callback0 = ModelCheckpoint(
        dirpath=manual_path,
        monitor='val_rmse',
        save_top_k=1,
        mode='min',
        filename='best-{epoch:02d}-{val_rmse:.3f}',
    ) # save best rmse ckpt
    best_ckpt_callback1 = ModelCheckpoint(
        dirpath=manual_path,
        monitor='val_ssim',
        save_top_k=3,
        mode='max',
        filename='best-{epoch:02d}-{val_ssim:.3f}',
    ) # save best ssim ckpt
    early_stop_callback = EarlyStopping(
        monitor='val_rmse', min_delta=1e-3, patience=5, verbose=True,
        mode='min'
    )
    if config.training_config.module_class == 'GANModule':
        callbacks = [ pause_callback, everystep_ckpt_callback, 
                      best_ckpt_callback0, best_ckpt_callback1 ]
    elif config.training_config.module_class == 'DetermModule':
        callbacks = [ best_ckpt_callback0, early_stop_callback ]
    
    '''
    model training
    '''
    module_class = globals()[config.training_config.module_class]
    module = module_class(
        **config.plmodule_config
    )
    
    trainer = pl.Trainer(
        accelerator="gpu", 
        max_epochs=config.training_config.max_epochs,
        logger=wandb_logger, 
        callbacks=callbacks,
        auto_scale_batch_size=True,
        log_every_n_steps=1,
        fast_dev_run=config.training_config.debug,
    )
    
    if config.training_config.get('ckpt_path'):
        print(f"Resume from ckpt")
        trainer.fit(module, trainloader, valloader, ckpt_path=config.training_config.ckpt_path)
    else: 
        print("Initialize training")
        trainer.fit(module, trainloader, valloader)
