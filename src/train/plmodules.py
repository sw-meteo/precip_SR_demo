'''
defines training schedule, logging strategy, etc.
'''

from typing import Union
import numpy as np
import wandb
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.functional.image import \
    structural_similarity_index_measure as ssim
from torchmetrics.functional import \
    peak_signal_noise_ratio as psnr
from models import UNet
from omegaconf import DictConfig


class BaseModule(pl.LightningModule):
    '''
    Base module for models with various training strategies
    accept common args, implement common methods
    '''
    def __init__(self, 
                 G_config: DictConfig = None, 
                 optimizer: str = "Adam",
                 lr: float = 1e-5,
                 b1: float = 0.5, 
                 b2: float = 0.999, 
                 n_opt_G: int = 1,
                 log_images: bool = True,
                 *args, **kwargs
                 ):
        super(BaseModule, self).__init__()
        self.save_hyperparameters()
        pass  # you can add basic check for config here
    
    def identity_score(self, fake, real):
        if self.hparams.loss_id == 'L2':
            return F.mse_loss(fake, real, reduce=False)
        elif self.hparams.loss_id == 'L1':
            return F.l1_loss(fake, real, reduce=False)
        else:
            raise NotImplementedError

    def log_images(self, imgs_list):
        imgs = [wandb.Image(img) for img in imgs_list]
        wandb.log({'images': imgs})
    
    def log_precip_map(self, lr_patch, hr_patch, hr_generated):
        hr_interp = F.interpolate(lr_patch, scale_factor=4)
        idx_sel = torch.randint(0, lr_patch.shape[0], (4,))
        self.log_images([hr_interp[idx_sel], hr_patch[idx_sel], hr_generated[idx_sel]])
    
    def get_optimizers(self, params):
        if self.hparams.optimizer == "Adam":
            opt = optim.Adam(params, 
                            lr=self.hparams.lr, 
                            betas=(self.hparams.b1, self.hparams.b2)
                            )
        elif self.hparams.optimizer == "RMSprop":
            opt = optim.RMSprop(params, 
                                lr=self.hparams.lr
                                )
        else:
            raise NotImplementedError
        return opt
    

class DetermModule(BaseModule):
    '''
    pairwise training with deterministic model
    '''
    def __init__(self, 
                 model_name: str = 'UNet',
                 G_config: DictConfig = None, 
                 loss_id: str = 'L2',
                 scheduler: str = None,  # [ None | 'ReduceLROnPlateau' ]
                 *args, **kwargs
                 ):
        super(DetermModule, self).__init__(
            G_config=G_config, *args, **kwargs
        )
        self.save_hyperparameters()
        self.G = globals()[model_name](**G_config)
    
    def forward(self, x):
        y = self.G(x)
        return y
    
    def training_step(self, batch, batch_idx):
        lr_patch, hr_patch = batch
        G = self.G
        
        hr_generated = G(lr_patch)
        
        id_loss = self.identity_score(hr_generated, hr_patch).mean()
        g_loss = id_loss
        
        self.log('g_loss', g_loss, prog_bar=True)
        self.log('ssim', ssim(hr_generated, hr_patch))
        self.log('psnr', psnr(hr_generated, hr_patch))
        self.log('rmse', F.mse_loss(hr_generated, hr_patch).sqrt())
        
        return g_loss
    
    def validation_step(self, batch, batch_idx):
        lr_patch, hr_patch = batch
        G = self.G
        
        hr_generated = G(lr_patch)
        
        self.log('val_ssim', ssim(hr_generated, hr_patch))
        self.log('val_psnr', psnr(hr_generated, hr_patch))
        self.log('val_rmse', F.mse_loss(hr_generated, hr_patch).sqrt())
        
        if batch_idx == 0 and self.hparams.log_images:
            self.log_precip_map(lr_patch, hr_patch, hr_generated)
    
    def test_step(self, batch, batch_idx):
        lr_patch, hr_patch = batch
        G = self.G
        
        hr_generated = G(lr_patch)
        
        self.log('test_ssim', ssim(hr_generated, hr_patch))
        self.log('test_psnr', psnr(hr_generated, hr_patch))
        self.log('test_rmse', F.mse_loss(hr_generated, hr_patch).sqrt())
    
    def configure_optimizers(self):
        if self.hparams.scheduler == 'ReduceLROnPlateau':
            opt_G = self.get_optimizers(self.G.parameters())
            sch_G = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt_G, 'min', patience=2, threshold=1e-3,
                threshold_mode='abs', min_lr=1e-5, verbose=True)
            return {
                'optimizer': opt_G,
                'lr_scheduler': sch_G,
                'monitor': 'val_rmse'
            }   
        else:
            opt_G = self.get_optimizers(self.G.parameters())
            return opt_G


if __name__ == '__main__':
    import rootutils
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    from pytorch_lightning import Trainer
    from dataset import SubImageDataset
    
    base = rootutils.find_root(search_from=__file__, indicator=".project-root")
    
    config = OmegaConf.load('config.yaml')
    config_cli = OmegaConf.from_cli() 
    # update layer-2 nested config from cmd line
    for key, nested_config in config.items():
        if key in config_cli:
            nested_config.update(config_cli[key])
    
    trainset = SubImageDataset(
        year_sel = [2010], mth_sel = [10],
        **config.dataset_config
    )
    valset = SubImageDataset(
        year_sel = [2010], mth_sel = [11],
        **config.dataset_config
    )
    
    trainloader = DataLoader(trainset, shuffle=True, **config.dataloader_config)
    valloader = DataLoader(valset, **config.dataloader_config)
    
    print("\n--- single batch forward test... ---")
    module_class = globals()[config.training_config.module_class]
    print(f'initiate module {module_class} for testing...')
    module = module_class( # plz switch to the module that you want to test
        log_images=False, 
        **config.plmodule_config
    )
    batch = next(iter(trainloader))
    lr_patch, hr_patch = batch
    print('load data: ', lr_patch.shape, hr_patch.shape)
    hr_generated = module.forward(lr_patch)
    print('after inference: ', hr_generated.shape)
    
    print("\n--- training test... --- ")
    trainer = Trainer(
        max_epochs=10,
        accelerator='gpu', devices=[0],
        fast_dev_run=True,
    )
    
    trainer.fit(module, trainloader, valloader)
