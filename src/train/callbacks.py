import os.path
import yaml
from datetime import datetime
from pytorch_lightning.callbacks import Callback

class WellPreparedPause(Callback):
    '''
    on KeyboardInterrupt, save ckpt and configs, then pause training
    '''
    def __init__(self, save_dir=None, save_ckpt=True, save_config=True):
        self.save_dir = save_dir
        self.save_ckpt = save_ckpt
        self.save_config = save_config
        
    def on_exception(self, trainer, pl_module, exception):
        if isinstance(exception, KeyboardInterrupt):
            
            print("Detected KeyboardInterrupt...\n")
            self._paused = True
            if self.save_dir is None:
                self.save_dir = trainer.default_root_dir
            
            timestr = datetime.now().strftime('%y%m%d%H%M%S')
            ckpt_path = os.path.join(self.save_dir, f"paused-{timestr}.ckpt")
            config_path = os.path.join(self.save_dir, f"config-{timestr}.yaml")
            if self.save_ckpt:
                trainer.save_checkpoint(ckpt_path)
            if self.save_config:
                with open(config_path, 'w') as f:
                    yaml.dump(dict(pl_module.hparams), f)
            print(f"Save requied info, then pause training.\n")
            
            raise KeyboardInterrupt
        
        else:
            raise exception
