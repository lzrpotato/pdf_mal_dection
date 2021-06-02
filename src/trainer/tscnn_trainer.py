from src.trainer.base_trainer import BaseTrainer
from src.dataloader.ts_dataloader import TSDataLoader
from src.model.TimeSeriesModel import TimeSeriesModel
from torch.optim import Adam
import torch.nn.functional as F
import pytorch_lightning as pl
import argparse
import logging
logger = logging.getLogger('trainer.TSCNNTrainer')

def train():
    args = setup_arg()
    print(args.no_cuda)
    args.determ = True
    dl = TSDataLoader()
    dl.setup()
    args.nclass = dl.nclass
    args.nc = dl.nc
    args.max_len = dl.maxlen
    
    if args.no_cuda:
        args.gpus = 0
    else:
        args.gpus = 1
    logger.info(f'args: {args}')
    trainer = pl.Trainer(gpus=args.gpus,
                #auto_select_gpus = True,
                max_epochs=args.epochs,
                progress_bar_refresh_rate=0.5,
                flush_logs_every_n_steps=100
            )
    for i in range(dl.nfold):
        model = TSCNNTrainer(args)
        dl.get_fold(i)
        trainer.fit(model,train_dataloader=dl.train_dataloader,val_dataloaders=dl.val_dataloader)
        results = trainer.predict(model,dataloaders=dl.test_dataloader)

def setup_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=64,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--determ', action='store_true', default=False,
                        help='Deterministic flag')

    return parser.parse_args()


class TSCNNTrainer(BaseTrainer):
    def __init__(self,args):
        super().__init__(args.nclass)
        self.args = args
        self.model = TimeSeriesModel(self.args.nclass,self.args.nc,self.args.max_len)

    def forward(self, batch):
        X, y = batch
        y_hat = self.model(X)
        return y_hat

    def loss(self, pred, batch):
        label = batch.y
        loss = F.cross_entropy(pred, label)
        return loss
    
    def metrics(self, pred, batch):
        label = batch.y
        self.acc_score(pred, label)
        self.f1_score(pred, label)

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(),
                lr=self.args.lr, weight_decay=self.args.weight_decay)
        
        return [optimizer]

