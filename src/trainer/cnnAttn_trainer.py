import argparse
import logging
import os
import time

import torch.nn.functional as F
from src.database.expe_dm import Database, Results
from src.dataloader.ts_dataloader import TSDataLoader
from src.model.CNN_Attn import CNN_Attn
from src.trainer.base_trainer import (BaseTrainer, get_trainer,
                                      setup_earlystop, setup_modelcheckpoint)
from src.util.cuda_status import get_num_gpus
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import get_cosine_schedule_with_warmup
from torchmetrics import ConfusionMatrix
from torchmetrics.classification import F1, Accuracy

logger = logging.getLogger('trainer.CNNAttnTrainer')

def param_setting(args):
    return f"""ep={args.epochs}_h={args.height}_nl={args.num_layers}{'_sw' if args.slide_window else '_gd'}_pt={args.patience}"""

def unique_name(exp_keys):
    return f"""exp={exp_keys['exp']}_nclass={exp_keys['nclass']}_dnn={exp_keys['dnn']}_ds={exp_keys['dataset']}_stride={exp_keys['stride']}_fold={exp_keys['fold']}"""

def test_results_to_results(exp_keys, test_result, early_stop, label):
    r = Results(exp_keys['exp'],
                exp_keys['nclass'],
                exp_keys['dnn'],
                exp_keys['dataset'],
                exp_keys['stride'],
                exp_keys['fold'],
                test_result['test_acc'],
                test_result['test_f1micro'],
                test_result['test_f1macro'],
                test_result['test_fbenign'],
                test_result['test_fmal'],
                early_stop.stopped_epoch,
                max(early_stop.stopped_epoch-early_stop.patience,0),
                label,
                time.strftime('%h/%d/%Y-%H:%M:%S'))
    return r

def train(args):
    args.determ = True
    dl = TSDataLoader(args.dataset, batch_size=args.batch_size, nfold=args.nfold)
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
        # sliding window setting
        dnn_mode = 'sw'
    else:
        # patch-grid setting
        dnn_mode = 'grid'

    # setup database
    db_name = f'exp_cnn_attn_{dnn_mode}_1.db'
    db = Database(db_name)

    resuls = []
    for i in range(dl.nfold):
        if args.fold != i:
            continue
        exp_keys = {'exp':args.exp, 'nclass': args.nclass, 'dnn': f'CNNAttn_{dnn_mode}', 'dataset':args.dataset, 'stride': args.stride ,'fold':i}
        if db.check_finished(exp_keys):
            continue
        model = CNNAttnTrainer(args, dl.class_to_index)
        
        early_stop = model.get_early_stop(patience=args.patience)
        ckp_cb = model.get_checkpoint_callback(unique_name(exp_keys))
        jobid = os.environ.get('SLURM_JOB_ID')
        if jobid is not None:
            version = f'{jobid}_fold_{i}_{time.strftime("%h-%d-%Y-%H:%M:%S")}'
        else:
            version = f'fold_{i}_{time.strftime("%h-%d-%Y-%H:%M:%S")}'
        trainer = get_trainer(args.gpus, args.epochs, early_stop, ckp_cb, unique_name(exp_keys), version)
        
        dl.get_fold(i)
        trainer.fit(model,train_dataloader=dl.train_dataloader,val_dataloaders=dl.val_dataloader)
        test_result = trainer.test(ckpt_path=ckp_cb.best_model_path, test_dataloaders=dl.test_dataloader)[0]
        
        ## convert test_result dictionary to Result class object
        r = test_results_to_results(exp_keys, test_result, early_stop, param_setting(args))
        db.save_results(r)

        resuls.append(test_result)
        logger.info('test results {}'.format(test_result))
    
    logger.info('all results \n{}'.format(
            db.get_by_query_as_dataframe(
                {   'exp':args.exp, 
                    'nclass': args.nclass,
                    'dnn': f'CNN_LSTM_{dnn_mode}',
                    'dataset':args.dataset,
                    'stride':args.stride,
            })
        ))

def setup_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train.')
    parser.add_argument('--patience', type=int, default=12,
                            help='Patience for early stopping.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--determ', action='store_true', default=False,
                        help='Deterministic flag')
    parser.add_argument('--width', type=int, default=128,
                        help='Width of CNN input')
    parser.add_argument('--height', type=int, default=128,
                        help='Height of CNN input')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='Number of layers for LSTM')
    parser.add_argument('--batch-size', type=int, default=32,
                            help='Batch size')
    parser.add_argument('--debug', action='store_true', default=False,
                            help='Debug flag')
    parser.add_argument('--slide-window', action='store_true', default=False,
                            help='sliding window')
    parser.add_argument('--exp', type=int, default=1,
                            help='Unique experiment number')
    parser.add_argument('--fold', type=int, default=0,
                            help='The fold number for current training')
    parser.add_argument('--stride', type=float, default=1/2, help='Stride, 1/2 or 3/4 of the window size')
    parser.add_argument('--attn-pool', type=str, default='maxpool', choices=['maxpool','avgpool',''])
    parser.add_argument('--dataset', type=str, default='word2vec_skipgram', help='dataset ', 
                        choices=['word2vec_cbow','word2vec_skipgram','byte'])
    parser.add_argument('--nfold', type=int, default=10, help='nfold cross-validation', choices=[5,10])
    return parser.parse_args()


class CNNAttnTrainer(BaseTrainer):
    def __init__(self,args,class_to_index):
        super().__init__()
        self.args = args
        self.class_to_index = class_to_index
        patch_size = (self.args.width,self.args.height)
        if self.args.slide_window:
            # slide_window = (window_size, stride)
            window_size = self.args.width*self.args.height
            stride = window_size * self.args.stride
            slide_window = (window_size, stride)
        else:
            slide_window = None
        self.model = CNN_Attn(self.args.nclass,self.args.nc,
                        batch_size=self.args.batch_size, attn_pool=self.args.attn_pool, num_layers=self.args.num_layers,
                        patch_size=patch_size, slide_window=slide_window)
        
        # metrics
        self.acc_score = Accuracy()
        self.f1_score_macro = F1(num_classes=args.nclass,average='macro')
        self.f1_score_micro = F1(num_classes=args.nclass,average='micro')
        self.f1_score_none = F1(num_classes=args.nclass,average='none')
        self.confmx = ConfusionMatrix(args.nclass)

    def forward(self, batch):
        X, y = batch
        #batch, seq_len = X.size()
        #X = X.view(batch,1,seq_len)
        logger.debug('cnnAttn {} {}'.format(X[0].shape, y[0].shape))
        y_hat = self.model(X)
        return y_hat

    def loss(self, pred, batch):
        X, label = batch
        logger.debug('loss {} {}'.format(pred, label))
        loss = F.cross_entropy(pred, label)
        return loss
    
    def metrics(self, phase, pred, batch):
        X, label = batch
        acc = self.acc_score(pred, label)
        self.f1_score_micro(pred, label)
        self.f1_score_macro(pred, label)
        self.f1_score_none(pred, label)
        self.confmx(pred, label)
        self.log(f'{phase}_acc_step', acc, sync_dist=True, prog_bar=True)

    def metrics_end(self, phase):
        acc_score = self.acc_score.compute()
        f1_score_micro = self.f1_score_micro.compute()
        f1_score_macro = self.f1_score_macro.compute()
        f1_score_none = self.f1_score_none.compute()
        cfm = self.confmx.compute()
        
        self.log(f'{phase}_acc_epoch', acc_score)
        self.log(f'{phase}_f1micro', f1_score_micro)
        self.log(f'{phase}_f1macro', f1_score_macro)
        self.log(f'{phase}_fbenign', f1_score_none[self.class_to_index['benign']])
        self.log(f'{phase}_fmal', f1_score_none[self.class_to_index['malicious']])
        self.log(f'{phase}_acc', acc_score)

        logger.info(f'[{phase}_acc_epoch] {acc_score} at {self.current_epoch}')
        logger.info(f'[{phase}_f1_score] {f1_score_micro}')
        logger.info(f'[{phase}_f1_score_macro] {f1_score_macro}')
        logger.info(f"[{phase}_fbenign] {f1_score_none[self.class_to_index['benign']]}", )
        logger.info(f"[{phase}_fmal] {f1_score_none[self.class_to_index['malicious']]}")
        logger.info(f'[{phase}_confmx] \n {cfm}')

        self.acc_score.reset()
        self.f1_score_micro.reset()
        self.f1_score_macro.reset()
        self.f1_score_none.reset()
        self.confmx.reset()

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(),
                lr=self.args.lr, weight_decay=self.args.weight_decay)
        scheduler1 = get_cosine_schedule_with_warmup(optimizer,
                    num_warmup_steps=7, num_training_steps=self.args.epochs)
        #scheduler = CosineAnnealingLR(optimizer, self.args.epochs)
        return {
            'optimizer': optimizer, 
            'lr_scheduler': scheduler1
        }

