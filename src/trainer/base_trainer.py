import logging
from typing import Any, List

import pytorch_lightning as pl
from pytorch_lightning import callbacks
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
import torch


logger = logging.getLogger('training.base_trainer')

def setup_modelcheckpoint(monitor, filename, mode):
    ckp_cb = ModelCheckpoint(dirpath='model_checkpoint',
            filename=filename + '-best-model-{epoch:02d}-{val_acc_epoch:.3f}',
            monitor=monitor,
            save_top_k=1,
            mode=mode
            )
    return ckp_cb

def setup_logger(exp_name,version=None):
    pl_logger = TensorBoardLogger(
        save_dir='tensorboard_logs',
        name=exp_name,
        version=version,
        )
    return pl_logger

def setup_earlystop(monitor,patience,mode):
    earlystop = EarlyStopping(
        monitor=monitor,
        patience=patience,
        mode=mode
    )
    return earlystop

def get_trainer(gpus, epochs, earlystop, ckp, exp_name, version=None):
    trainer = pl.Trainer(
        gpus=gpus,
        #auto_select_gpus = True,
        max_epochs=epochs,
        progress_bar_refresh_rate=0.5,
        flush_logs_every_n_steps=100,
        logger=setup_logger(exp_name, version),
        callbacks=[earlystop, ckp],
    )
    return trainer


class BaseTrainer(pl.LightningModule):
    def __init__(self):
        super(BaseTrainer,self).__init__()


    def forward(self, batch):
        pass 

    def loss(self, pred, batch):
        pass

    def configure_optimizers(self):
        pass

    def metrics(self, phase, pred, batch):
        pass

    def metrics_end(self, phase):
        pass

    def shared_my_step(self, batch, batch_nb, phase):
        # batch
        y_hat = self.forward(batch)
        loss = self.loss(y_hat, batch)
        
        # acc
        a, y_hat = torch.max(y_hat, dim=1)

        self.log(f'{phase}_loss_step', loss, sync_dist=True, prog_bar=True)
        self.log(f'{phase}_loss_epoch', loss, sync_dist=True, on_step=False, on_epoch=True, prog_bar=True)

        self.metrics(phase, y_hat, batch)
        
        return loss

    def epoch_end(self, outputs, phase):
        self.metrics_end(phase)

    def training_step(self, batch, batch_nb):
        phase = 'train'
        outputs = self.shared_my_step(batch, batch_nb, phase)
        return outputs

    def training_epoch_end(self, outputs) -> None:
        phase = 'train'
        self.epoch_end(outputs, phase)

    def validation_step(self, batch, batch_nb):
        phase = 'val'
        outputs = self.shared_my_step(batch, batch_nb, phase)
        return outputs

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        phase = 'val'
        self.epoch_end(outputs, phase)

    def test_step(self, batch, batch_nb):
        phase = 'test'
        # fwd
        y_hat = self.forward(batch)
        loss = self.loss(y_hat, batch)
        # acc
        a, y_hat = torch.max(y_hat, dim=1)

        self.log(f'{phase}_loss_step', loss, sync_dist=True, prog_bar=True)
        self.log(f'{phase}_loss_epoch', loss, sync_dist=True, on_step=False, on_epoch=True, prog_bar=True)
        self.metrics(phase, y_hat, batch)
        
        return 

    def test_epoch_end(self, outputs: List[Any]) -> None:
        phase = 'test'
        self.epoch_end(outputs, phase)
        
    def get_early_stop(self, patience):
        return setup_earlystop('val_acc_epoch', patience, 'max')

    def get_checkpoint_callback(self, filename):
        return setup_modelcheckpoint('val_acc_epoch', filename, 'max')