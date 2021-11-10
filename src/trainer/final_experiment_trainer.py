import argparse
import logging
import os
import time
import re

import torch.nn as nn
import torch.nn.functional as F
from src.database.expe_dm import Database, Results
from src.dataloader.ts_dataloader import TSDataLoader
from src.model.Ablation import get_ablation
from src.model.ByteCNN_YSJ import CNNYSJ
from src.model.ByteMalConv import ByteMalConv
from src.model.ByteCNN_RF import ModelC
from src.model.CNN_tl import get_cnn_tl_model
from src.trainer.base_trainer import (BaseTrainer, get_trainer,
                                      setup_earlystop, setup_modelcheckpoint)
from src.util.cuda_status import get_num_gpus


logger = logging.getLogger('trainer.final_trainer')

def param_setting(args):
    return f"""ep={args.epochs}_h={args.height}_nl={args.num_layers}_pt={args.patience}"""

def unique_name(exp_keys):
    return "-".join([f"{k}={v}" for k,v in exp_keys.items()])
    #return f"""exp={exp_keys['exp']}-nclass={exp_keys['nclass']}-dnn={exp_keys['dnn']}-ds={exp_keys['dataset']}-stride={exp_keys['stride']}=fold={exp_keys['fold']}"""

def parse_best_epoch(best_model_path):
    name = os.path.basename(best_model_path)
    ret = {}
    for pair in re.split('-',name):
        if "=" in pair:
            k, v = pair.split("=")
            if k == 'epoch':
                ret[k] = v
            elif k == 'val_acc_epoch':
                ret[k] = v.strip('.ckpt')
                break

    return ret

def test_results_to_results(args, exp_keys, test_result, early_stop, best_info, label):
    actual_stopped = early_stop.stopped_epoch if early_stop.stopped_epoch != 0 else args.epochs
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
                actual_stopped,
                best_info['epoch'],
                label,
                time.strftime('%h/%d/%Y-%H:%M:%S'))
    return r

def train(args):
    args.determ = True

    if 'MIMS' in args.model:
        nbyte = [1,2]
    else:
        nbyte = 1

    mlist = args.model.split('_')
    if mlist[0] in ['MIMS','SI']:
        if 'EB' in mlist:
            embedding = True
        elif 'NE' in mlist:
            embedding = False
    elif mlist[0] in ['CNNYSJ','ByteMalConv','CNNRF']:
        embedding = True
    elif mlist[0] in ['TF']:
        embedding = False

    dl = TSDataLoader(args.dataset, args.split, nbyte=nbyte, batch_size=args.batch_size, nfold=args.nfold, embedding=embedding)
    dl.setup()
    args.nclass = dl.nclass
    args.nc = dl.nc
    args.max_len = dl.maxlen
    
    if args.no_cuda:
        args.gpus = 0
    else:
        args.gpus = get_num_gpus()
    logger.info(f'args: {args}')

    # setup database
    dnn_name = args.model
    db_name = f'final_exp.db'
    db = Database(db_name)

    if args.split == 'tvt':
        dataset_key = args.dataset
    else:
        dataset_key = args.dataset + '_' + args.split
    
    resuls = []
    for i in range(dl.nfold):
        if args.fold != i:
            continue
        
        exp_keys = {'exp':args.exp, 'nclass': args.nclass, 'dnn': dnn_name, 'dataset':dataset_key, 'stride': args.stride ,'fold':i}
        if db.check_finished(exp_keys):
            continue
        model = FCNCATrainer(args, dl.class_to_index)
        
        early_stop = model.get_early_stop(patience=args.patience)
        ckp_cb = model.get_checkpoint_callback(unique_name(exp_keys))
        jobid = os.environ.get('SLURM_JOB_ID')
        if jobid is not None:
            version = f'{jobid}_fold_{i}_{time.strftime("%h-%d-%Y-%H:%M:%S")}'
        else:
            version = f'fold_{i}_{time.strftime("%h-%d-%Y-%H:%M:%S")}'
        trainer = get_trainer(args.gpus, args.epochs, early_stop, ckp_cb, unique_name(exp_keys), version)
        
        dl.get_fold(i)
        trainer.fit(model, datamodule=dl)
        test_result = trainer.test(ckpt_path=ckp_cb.best_model_path, datamodule=dl)[0]
        
        ## convert test_result dictionary to Result class object
        r = test_results_to_results(args, exp_keys, test_result, early_stop, parse_best_epoch(ckp_cb.best_model_path), param_setting(args))
        db.save_results(r)

        resuls.append(test_result)
        logger.info('test results {}'.format(test_result))
    
    logger.info('all results \n{}'.format(
            db.get_by_query_as_dataframe(
                {   'exp':args.exp, 
                    'nclass': args.nclass,
                    'dnn': dnn_name,
                    'dataset':dataset_key,
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
    parser.add_argument('--num-layers', type=int, default=1,
                        help='Number of layers for LSTM')
    parser.add_argument('--batch-size', type=int, default=32,
                            help='Batch size')
    parser.add_argument('--debug', action='store_true', default=False,
                            help='Debug flag')
    parser.add_argument('--exp', type=int, default=1,
                            help='Unique experiment number')
    parser.add_argument('--fold', type=int, default=0,
                            help='The fold number for current training')
    parser.add_argument('--stride', type=float, default=1/2, help='Stride, 1/2 or 3/4 of the window size')
    parser.add_argument('--globalpool', type=str, default='maxpool', choices=['maxpool','avgpool'])
    parser.add_argument('--dataset', type=str, default='byte', help='dataset ', 
                        choices=['word2vec_cbow','word2vec_skipgram','byte'])
    parser.add_argument('--split', type=str, default= 'tvt', choices=['tvt', 'ttv', '6000'])
    parser.add_argument('--nfold', type=int, default=10, help='nfold cross-validation', choices=[5,10])
    parser.add_argument('--model', type=str, default='MIMS_SEVT_P2_NO_NE', choices=['MIMS_SEVT_P2_NO_NE','MIMS_SEVT_P2_VT_NE','MIMS_SEVT_P2_SE_NE','MIMS_SEVT_P2_A_NE',
                                                                     'MIMS_SEVT_P2_A_EB', 'SI_SEVT_P2_A_NE', 'SI_SEVT_P2_A_EB','MIMS_CCA_NE','MIMS_CA_NE','MIMS_CBAM_NE',
                                                                     'MIMS_CCA_EB','MIMS_CA_EB','ByteMalConv','MIMS_SEVT_P2_A_NE_SM','MIMS_SEVT_P1_A_NE','MIMS_SEVT_S1_A_NE',
                                                                     'MIMS_SEVT_S2_A_NE','MIMS_SEVT_P3_A_NE','CNNRF',
                                                                     'TF_MBNET3','TF_RESNET101','TF_SZNET11','TF_VGG19', 'CNN_LSTM'])
    return parser.parse_args()


class FCNCATrainer(BaseTrainer):
    def __init__(self,args,class_to_index):
        super().__init__(args.nclass)
        self.args = args
        self.class_to_index = class_to_index
        patch_size = (self.args.width,self.args.height)
        # slide_window = (window_size, stride)
        window_size = self.args.width*self.args.height
        stride = window_size * self.args.stride
        slide_window = (window_size, stride)

        mlist = self.args.model.split('_')
        if mlist[0] in ['SI','MIMS']:
            # MIMS_SEVT_P2_NO_EB
            mlist = self.args.model.split('_')
            if mlist[1] == 'SEVT':
                atten_param = {'structure':mlist[2],'ablation': mlist[3]}

            elif mlist[1] in ['CCA', 'CA', 'CBAM']:
                atten_param = {}
            
            if 'EB' in mlist:
                embedding = True
            elif 'NE' in mlist:
                embedding = False

            if 'SM' in mlist:
                fusion = 'sum'
            else:
                fusion = 'concat'
            self.model = get_ablation(scale=mlist[0],attention=mlist[1],atten_param=atten_param, embedding=embedding,fusion=fusion,nclass=self.args.nclass,inplane=self.args.nc,
                            globalpool=self.args.globalpool, num_layers=self.args.num_layers,
                            patch_size=patch_size, slide_window=slide_window)
        elif mlist[0] in ['CNNYSJ']:
            pass
        elif mlist[0] in ['ByteMalConv']:
            self.model = ByteMalConv(self.args.nclass)
        elif mlist[0] in ['CNNRF']:
            self.model = ModelC(args.nclass)
        elif mlist[0] in ['TF']:
            self.model = get_cnn_tl_model(mlist[1], self.args.nclass, (256,256))