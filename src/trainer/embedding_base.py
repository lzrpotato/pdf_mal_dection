import logging
from typing import Any, List

import pytorch_lightning as pl
from pytorch_lightning import callbacks
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
import torch
from torchmetrics.classification import Accuracy
from torchmetrics.classification import F1

logger = logging.getLogger('training.base_trainer')

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

def get_trainer(gpus, epochs, earlystop, exp_name, version=None):
    trainer = pl.Trainer(
        gpus=gpus,
        #auto_select_gpus = True,
        max_epochs=epochs,
        progress_bar_refresh_rate=0.5,
        flush_logs_every_n_steps=100,
        logger=setup_logger(exp_name, version),
        callbacks=[earlystop],
    )
    return trainer


class EmbeddingBase(pl.LightningModule):
    def __init__(self):
        super(EmbeddingBase,self).__init__()
        
    def forward(self, batch):
        pass 

    def loss(self, pred, batch):
        pass

    def configure_optimizers(self):
        pass

    def metrics(self, pred, batch):
        pass

    def shared_my_step(self, batch, batch_nb, phase):
        # batch
        y_hat = self.forward(batch)
        loss = self.loss(y_hat, batch)
        
        # acc
        a, y_hat = torch.max(y_hat, dim=1)

        self.log(f'{phase}_loss_step', loss, sync_dist=True, prog_bar=True)
        self.log(f'{phase}_loss_epoch', loss, sync_dist=True, on_step=False, on_epoch=True, prog_bar=True)

        self.metrics(y_hat, batch)
        
        return loss

    def epoch_end(self, outputs, phase):
        pass

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
        self.metrics(y_hat, batch)
        return 

    def test_epoch_end(self, outputs: List[Any]) -> None:
        phase = 'test'

    def get_early_stop(self, patience):
        return setup_earlystop('val_loss_epoch', patience, 'min')
