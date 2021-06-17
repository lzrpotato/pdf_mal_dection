import argparse
import logging

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from src.dataloader.ts_dataloader import TSDataLoader
from src.model.CNN_LSTM import CNN_LSTM
from src.trainer.base_trainer import BaseTrainer, get_trainer, setup_earlystop
from src.util.cuda_status import get_num_gpus
from src.database.expe_dm import Results, Database
from torch.optim import Adam


logger = logging.getLogger('trainer.CNNLSTMTrainer')

def train(args):
    # database setting
    db = Database('exp_cnn_lstm.db')
    exp = 1

    args.determ = True
    dl = TSDataLoader(batch_size=args.batch_size)
    dl.setup()
    args.nclass = dl.nclass
    args.nc = dl.nc
    args.max_len = dl.maxlen
    
    if args.no_cuda:
        args.gpus = 0
    else:
        args.gpus = get_num_gpus()
    logger.info(f'args: {args}')
    
    dnn_mode = ''
    if args.slide_window:
        dnn_mode = 'sw'
    else:
        dnn_mode = 'grid'

    resuls = []
    for i in range(dl.nfold):
        exp_keys = {'exp':exp, 'nclass': args.nclass, 'dnn': f'CNN_LSTM_{dnn_mode}', 'fold':i}
        if db.check_finished(exp_keys):
            continue
        model = CNNLSTMTrainer(args)
        
        early_stop = model.get_early_stop(patience=args.patience)
        trainer = get_trainer(args.gpus, args.epochs, early_stop)
        
        dl.get_fold(i)
        trainer.fit(model,train_dataloader=dl.train_dataloader,val_dataloaders=dl.val_dataloader)
        test_result = trainer.test(model,test_dataloaders=dl.test_dataloader)[0]
        
        r = Results(exp,args.nclass,f'CNN_LSTM_{dnn_mode}',i, test_result['test acc'], test_result['fscore'],early_stop.stopped_epoch,early_stop.stopped_epoch-early_stop.patience)
        db.save_results(r)
        resuls.append(test_result)
        logger.info('test results {}'.format(test_result))
    
    logger.info('all results {}'.format(resuls))

def setup_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train.')
    parser.add_argument('--patience', type=int, default=10,
                            help='Patience for early stopping.')
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
    parser.add_argument('--width', type=int, default=128,
                        help='Width of CNN input')
    parser.add_argument('--height', type=int, default=128,
                        help='Height of CNN input')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of layers for LSTM')
    parser.add_argument('--batch-size', type=int, default=32,
                            help='Batch size')
    parser.add_argument('--debug', action='store_true', default=False,
                            help='Debug flag')
    parser.add_argument('--slide-window', action='store_true', default=False,
                            help='sliding window')
    return parser.parse_args()


class CNNLSTMTrainer(BaseTrainer):
    def __init__(self,args):
        super().__init__(args.nclass)
        self.args = args
        patch_size = (self.args.width,self.args.height)
        if self.args.slide_window:
            slide_window = (self.args.width*self.args.height, self.args.width*self.args.height/2)
        else:
            slide_window = None
        self.model = CNN_LSTM(self.args.nclass,self.args.nc,self.args.hidden,
                        batch_size=self.args.batch_size, num_layers=self.args.num_layers,
                        patch_size=patch_size, slide_window=slide_window)

    def forward(self, batch):
        X, y = batch
        #batch, seq_len = X.size()
        #X = X.view(batch,1,seq_len)
        logger.debug('cnnlstm {} {} {}'.format(batch, X, y))
        y_hat = self.model(X)
        return y_hat

    def loss(self, pred, batch):
        X, label = batch
        logger.debug('loss {} {}'.format(pred, label))
        loss = F.cross_entropy(pred, label)
        return loss
    
    def metrics(self, pred, batch):
        X, label = batch
        self.acc_score(pred, label)
        self.f1_score(pred, label)

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(),
                lr=self.args.lr, weight_decay=self.args.weight_decay)
        return [optimizer]

