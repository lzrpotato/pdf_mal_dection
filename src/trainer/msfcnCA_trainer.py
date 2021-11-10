import argparse
import logging
import os
import time
import re

import torch.nn as nn
import torch.nn.functional as F
from src.database.expe_dm import Database, Results
from src.dataloader.ts_dataloader import TSDataLoader
from src.model.MSFCN_CA import get_msfcnca_twoheadfcn, get_msfcnca_twoheadfcnca
from src.model.MSFCN1_CA import get_msfcn1ca_twoheadfcn, get_msfcn1ca_twoheadfcnca
from src.model.MSFCN_LSTM_CA import get_msfcnlstmca_twoheadfcn, get_msfcnlstmca_twoheadfcnca
from src.model.MSFCN2_CA import get_msfcn2ca_oneheadfcn, get_msfcn2ca_twoheadfcn, get_msfcn2ca_twoheadfcnca
from src.model.MSFCN1_CCA import get_msfcn1cca_twoheadfcn, get_msfcn1cca_twoheadfcnca, get_msfcn1cca_sharedfcnca
from src.model.MSFCN3_CCA import get_msfcn3cca_twoheadfcn, get_msfcn3cca_twoheadfcncca
from src.model.MSFCN4_CCA import get_msfcn4cca_twoheadfcn, get_msfcn4cca_twoheadfcncca
from src.model.MSFCN1_CCCA import get_msfcn1ccca_oneheadfcn, get_msfcn1ccca_twoheadfcn, get_msfcn1ccca_twoheadfcnccca
from src.model.MSFCN1_CCCA1 import get_msfcn1ccca1_twoheadfcn, get_msfcn1ccca1_twoheadfcnccca
from src.model.MSFCN1_CCCA2 import get_msfcn1ccca2_twoheadfcn, get_msfcn1ccca2_twoheadfcnccca
from src.model.FXMSFCN1_CCA import get_fxmsfcn1cca_twoheadfcn, get_fxmsfcn1cca_twoheadfcnca
from src.model.MSFCN6_CCA import get_msfcn6cca_twoheadfcn, get_msfcn6cca_twoheadfcncca
from src.model.MSFCN5_CCA import get_msfcn5cca_oneheadfcn
from src.model.MSFCN1_SCCA import get_msfcn1scca_twoheadfcn, get_msfcn1scca_twoheadfcnscca
from src.model.MIMSFCN_CCA import get_mimsfcncca_twoheadfcn, get_mimsfcncca_twoheadfcncca
from src.model.MSFCN1_SA import get_msfcn1sa_twoheadfcn, get_msfcn1sa_twoheadfcnsa
from src.model.MIMSFCN_SA import get_mimsfcnsa_oneheadfcn, get_mimsfcnsa_twoheadfcnsa, get_mimsfcnsa_twoheadfcn
from src.model.MIMSFCN_PA import get_mimsfcnpa_twoheadfcnpa
from src.model.MSFCN1_PA import get_msfcn1pa_twoheadfcnpa
from src.model.MSFCN1_SEA import get_msfcn1sea_twoheadfcnsea_para, get_msfcn1sea_twoheadfcnsea_serial1, get_msfcn1sea_twoheadfcnsea_serial2, \
        get_msfcn1sea_twoheadfcnsea_para_r, get_msfcn1sea_twoheadfcnsea_serial1_r, get_msfcn1sea_twoheadfcnsea_serial2_r, \
        get_msfcn1sea_twoheadfcnsea_para1, get_msfcn1sea_twoheadfcnsea_para1_r, get_msfcn1sea_twoheadfcnsea_para2, get_msfcn1sea_twoheadfcnsea_para2_r
from src.model.MSFCN1_PBA import get_msfcn1pba_twoheadfcnpba
from src.model.MSFCN1_PCA import get_msfcn1pca_twoheadfcn
from src.model.FCN1D import get_fcn1d, get_fcn1d_attn
from src.model.MIMSFCN_SEA import get_mimsfcnsea_twoheadfcnsea_para1, get_mimsfcnsea_twoheadfcnsea_para2
from src.model.MIMSFCN_SEVT import get_mimsfcnsevt_twoheadfcnsevt_para1, get_mimsfcnsevt_twoheadfcnsevt_para2,get_mimsfcnsevt_twoheadfcnsevt_para3,\
    get_mimsfcnsevt_twoheadfcnsevt_serial1, get_mimsfcnsevt_twoheadfcnsevt_serial2
from src.trainer.base_trainer import (BaseTrainer, get_trainer,
                                      setup_earlystop, setup_modelcheckpoint)
from src.util.cuda_status import get_num_gpus


logger = logging.getLogger('trainer.msfcnCA')

def param_setting(args):
    return f"""ep={args.epochs}_h={args.height}_nl={args.num_layers}{'_sw' if args.slide_window else '_gd'}_pt={args.patience}"""

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

    dl = TSDataLoader(args.dataset, nbyte=nbyte, batch_size=args.batch_size, nfold=args.nfold)
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
    dnn_name = f'{args.model}_{dnn_mode}'
    db_name = f'exp_{dnn_name}_1.db'
    db = Database(db_name)

    resuls = []
    for i in range(dl.nfold):
        if args.fold != i:
            continue
        exp_keys = {'exp':args.exp, 'nclass': args.nclass, 'dnn': dnn_name, 'dataset':args.dataset, 'stride': args.stride ,'fold':i}
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
        trainer.fit(model,train_dataloaders=dl.train_dataloader,val_dataloaders=dl.val_dataloader)
        test_result = trainer.test(ckpt_path=ckp_cb.best_model_path, test_dataloaders=dl.test_dataloader)[0]
        
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
    parser.add_argument('--num-layers', type=int, default=1,
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
    parser.add_argument('--globalpool', type=str, default='maxpool', choices=['maxpool','avgpool'])
    parser.add_argument('--dataset', type=str, default='byte', help='dataset ', 
                        choices=['word2vec_cbow','word2vec_skipgram','byte'])
    parser.add_argument('--nfold', type=int, default=10, help='nfold cross-validation', choices=[5,10])
    parser.add_argument('--model', type=str, default='MSFCNCATH', choices=['MSFCNCATH','MSFCNCATHCA','MSFCN1CATH',
                                        'MSFCN1CATHCA','FCN1D','FCN1DATTN','MSFCNLSTMCATH','MSFCNLSTMCATHCA',
                                        'FCN2CA','MSFCN2CATH','MSFCN2CATHCA','MSFCN1CCATH','MSFCN1CCATHCCA','MSFCN1CCASH',
                                        'MSFCN3CCATH','MSFCN3CCATHCCA','MSFCN3CATH','MSFCN3CATHCA',
                                        'MSFCN4CCATH','MSFCN4CCATHCCA','MSFCN1SHCCA',
                                        'MSFCN1CCCATH','MSFCN1CCCATHCCCA','MSFCN1CCCA1TH','MSFCN1CCCA1THCCCA',
                                        'MSFCN5CCA','MSFCN1CCCA2TH','MSFCN1CCCA2THCCCA','FXMSFCN1CCATHCCA','FXMSFCN1CCATH',
                                        'MSFCN6CCATH','MSFCN6CCATHCCA','MSFCN1SCCATH','MSFCN1SCCATHSCCA',
                                        'MIMSFCNCCATH','MIMSFCNCCATHCCA','MSFCN1SATH','MSFCN1SATHSA','MIMSFCNSATH','MIMSFCNSATHSA',
                                        'MIMSFCNPATHPA','MSFCN1PATHPA','MSFCN1SEAPTHSEA','MSFCN1SEAS1THSEA','MSFCN1SEAS2THSEA',
                                        'MSFCN1SEAPRTHSEA','MSFCN1SEAS1RTHSEA','MSFCN1SEAS2RTHSEA','MSFCN1PBATHPBA',
                                        'MSFCN1SEAP1THSEA','MSFCN1SEAP2THSEA','MSFCN1SEAP1RTHSEA','MSFCN1SEAP2RTHSEA',
                                        'MIMSFCNSEAP1THSEA','MIMSFCNSEAP2THSEA','MIMSFCNSEVTP1THSEVT','MIMSFCNSEVTP2THSEVT',
                                        'MIMSFCNSEVTP3THSEVT','MIMSFCNSEVTS1THSEVT','MIMSFCNSEVTS2THSEVT'])
    return parser.parse_args()


class FCNCATrainer(BaseTrainer):
    def __init__(self,args,class_to_index):
        super().__init__(args.nclass)
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
        if self.args.model == 'MSFCNCATH':
            self.model = get_msfcnca_twoheadfcn(self.args.nclass,self.args.nc,
                            globalpool=self.args.globalpool, num_layers=self.args.num_layers,
                            patch_size=patch_size, slide_window=slide_window)
        elif self.args.model == 'MSFCNCATHCA':
            self.model =  get_msfcnca_twoheadfcnca(self.args.nclass,self.args.nc,
                            globalpool=self.args.globalpool, num_layers=self.args.num_layers,
                            patch_size=patch_size, slide_window=slide_window)
        elif self.args.model == 'MSFCN1CATH':
            self.model =  get_msfcn1ca_twoheadfcn(self.args.nclass,self.args.nc,
                            globalpool=self.args.globalpool, num_layers=self.args.num_layers,
                            patch_size=patch_size, slide_window=slide_window)
        elif self.args.model == 'MSFCN1CATHCA':
            self.model =  get_msfcn1ca_twoheadfcnca(self.args.nclass,self.args.nc,
                            globalpool=self.args.globalpool, num_layers=self.args.num_layers,
                            patch_size=patch_size, slide_window=slide_window)
        elif self.args.model == 'FCN1D':
            self.model = get_fcn1d(self.args.nclass,self.args.nc,
                            globalpool=self.args.globalpool, num_layers=self.args.num_layers,
                            patch_size=patch_size, slide_window=slide_window)
        elif self.args.model == 'FCN1DATTN':
            self.model = get_fcn1d_attn(self.args.nclass,self.args.nc,
                            globalpool=self.args.globalpool, num_layers=self.args.num_layers,
                            patch_size=patch_size, slide_window=slide_window)
        elif self.args.model == 'MSFCNLSTMCATH':
            self.model = get_msfcnlstmca_twoheadfcn(self.args.nclass,self.args.nc,
                            globalpool=self.args.globalpool, num_layers=self.args.num_layers,
                            patch_size=patch_size, slide_window=slide_window)
        elif self.args.model == 'MSFCNLSTMCATHCA':
            self.model = get_msfcnlstmca_twoheadfcnca(self.args.nclass,self.args.nc,
                            globalpool=self.args.globalpool, num_layers=self.args.num_layers,
                            patch_size=patch_size, slide_window=slide_window)
        elif self.args.model == 'MSFCN2CATH':
            self.model = get_msfcn2ca_twoheadfcn(self.args.nclass,self.args.nc,
                            globalpool=self.args.globalpool, num_layers=self.args.num_layers,
                            patch_size=patch_size, slide_window=slide_window)
        elif self.args.model == 'MSFCN2CATHCA':
            self.model = get_msfcn2ca_twoheadfcnca(self.args.nclass,self.args.nc,
                            globalpool=self.args.globalpool, num_layers=self.args.num_layers,
                            patch_size=patch_size, slide_window=slide_window)
        elif self.args.model == 'FCN2CA':
            self.model = get_msfcn2ca_oneheadfcn(self.args.nclass,self.args.nc,
                            globalpool=self.args.globalpool, num_layers=self.args.num_layers,
                            patch_size=patch_size, slide_window=slide_window)
        elif self.args.model == 'MSFCN1CCATH':
            self.model = get_msfcn1cca_twoheadfcn(self.args.nclass,self.args.nc,
                            globalpool=self.args.globalpool, num_layers=self.args.num_layers,
                            patch_size=patch_size, slide_window=slide_window)
        elif self.args.model == 'MSFCN1CCATHCCA':
            self.model = get_msfcn1cca_twoheadfcnca(self.args.nclass,self.args.nc,
                            globalpool=self.args.globalpool, num_layers=self.args.num_layers,
                            patch_size=patch_size, slide_window=slide_window)
        elif self.args.model == 'MSFCN1CCASH':
            self.model = get_msfcn1cca_sharedfcnca(self.args.nclass,self.args.nc,
                            globalpool=self.args.globalpool, num_layers=self.args.num_layers,
                            patch_size=patch_size, slide_window=slide_window)
        elif self.args.model == 'MSFCN3CCATH':
            self.model = get_msfcn3cca_twoheadfcn(self.args.nclass,self.args.nc,
                            globalpool=self.args.globalpool, num_layers=self.args.num_layers,
                            patch_size=patch_size, slide_window=slide_window)
        elif self.args.model == 'MSFCN3CCATHCCA':
            self.model = get_msfcn3cca_twoheadfcncca(self.args.nclass,self.args.nc,
                            globalpool=self.args.globalpool, num_layers=self.args.num_layers,
                            patch_size=patch_size, slide_window=slide_window)
        elif self.args.model == 'MSFCN3CATH':
            self.model = get_msfcn3cca_twoheadfcn(self.args.nclass,self.args.nc,
                            globalpool=self.args.globalpool, num_layers=self.args.num_layers,
                            patch_size=patch_size, slide_window=slide_window)
        elif self.args.model == 'MSFCN3CATHCA':
            self.model = get_msfcn3cca_twoheadfcncca(self.args.nclass,self.args.nc,
                            globalpool=self.args.globalpool, num_layers=self.args.num_layers,
                            patch_size=patch_size, slide_window=slide_window)
        elif self.args.model == 'MSFCN4CCATH':
            self.model = get_msfcn4cca_twoheadfcn(self.args.nclass,self.args.nc,
                            globalpool=self.args.globalpool, num_layers=self.args.num_layers,
                            patch_size=patch_size, slide_window=slide_window)
        elif self.args.model == 'MSFCN4CCATHCCA':
            self.model = get_msfcn4cca_twoheadfcncca(self.args.nclass,self.args.nc,
                            globalpool=self.args.globalpool, num_layers=self.args.num_layers,
                            patch_size=patch_size, slide_window=slide_window)
        elif self.args.model == 'MSFCN1SHCCA':
            self.model = get_msfcn4cca_twoheadfcncca(self.args.nclass,self.args.nc,
                            globalpool=self.args.globalpool, num_layers=self.args.num_layers,
                            patch_size=patch_size, slide_window=slide_window)
        elif self.args.model == 'MSFCN1CCCATH':
            self.model = get_msfcn1ccca_twoheadfcn(self.args.nclass,self.args.nc,
                            globalpool=self.args.globalpool, num_layers=self.args.num_layers,
                            patch_size=patch_size, slide_window=slide_window)
        elif self.args.model == 'MSFCN1CCCATHCCCA':
            self.model = get_msfcn1ccca_twoheadfcnccca(self.args.nclass,self.args.nc,
                            globalpool=self.args.globalpool, num_layers=self.args.num_layers,
                            patch_size=patch_size, slide_window=slide_window)
        elif self.args.model == 'MSFCN1CCCA1THCCCA':
            self.model = get_msfcn1ccca1_twoheadfcnccca(self.args.nclass,self.args.nc,
                            globalpool=self.args.globalpool, num_layers=self.args.num_layers,
                            patch_size=patch_size, slide_window=slide_window)
        elif self.args.model == 'MSFCN1CCCA1TH':
            self.model = get_msfcn1ccca1_twoheadfcn(self.args.nclass,self.args.nc,
                            globalpool=self.args.globalpool, num_layers=self.args.num_layers,
                            patch_size=patch_size, slide_window=slide_window)
        elif self.args.model == 'MSFCN5CCA':
            self.model = get_msfcn5cca_oneheadfcn(self.args.nclass,self.args.nc,
                            globalpool=self.args.globalpool, num_layers=self.args.num_layers,
                            patch_size=patch_size, slide_window=slide_window)
        elif self.args.model == 'MSFCN1CCCA2TH':
            self.model = get_msfcn1ccca2_twoheadfcn(self.args.nclass,self.args.nc,
                            globalpool=self.args.globalpool, num_layers=self.args.num_layers,
                            patch_size=patch_size, slide_window=slide_window)
        elif self.args.model == 'MSFCN1CCCA2THCCCA':
            self.model = get_msfcn1ccca2_twoheadfcnccca(self.args.nclass,self.args.nc,
                            globalpool=self.args.globalpool, num_layers=self.args.num_layers,
                            patch_size=patch_size, slide_window=slide_window)
        elif self.args.model == 'FXMSFCN1CCATHCCA':
            self.model = get_fxmsfcn1cca_twoheadfcnca(self.args.nclass,self.args.nc,
                            globalpool=self.args.globalpool, num_layers=self.args.num_layers,
                            patch_size=patch_size, slide_window=slide_window)
        elif self.args.model == 'FXMSFCN1CCATH':
            self.model = get_fxmsfcn1cca_twoheadfcn(self.args.nclass,self.args.nc,
                            globalpool=self.args.globalpool, num_layers=self.args.num_layers,
                            patch_size=patch_size, slide_window=slide_window)
        elif self.args.model == 'MSFCN6CCATHCCA':
            self.model = get_msfcn6cca_twoheadfcncca(self.args.nclass,self.args.nc,
                            globalpool=self.args.globalpool, num_layers=self.args.num_layers,
                            patch_size=patch_size, slide_window=slide_window)
        elif self.args.model == 'MSFCN6CCATH':
            self.model = get_msfcn6cca_twoheadfcn(self.args.nclass,self.args.nc,
                            globalpool=self.args.globalpool, num_layers=self.args.num_layers,
                            patch_size=patch_size, slide_window=slide_window)
        elif self.args.model == 'MSFCN1SCCATH':
            self.model = get_msfcn1scca_twoheadfcn(self.args.nclass,self.args.nc,
                            globalpool=self.args.globalpool, num_layers=self.args.num_layers,
                            patch_size=patch_size, slide_window=slide_window)
        elif self.args.model == 'MSFCN1SCCATHSCCA':
            self.model = get_msfcn1scca_twoheadfcnscca(self.args.nclass,self.args.nc,
                            globalpool=self.args.globalpool, num_layers=self.args.num_layers,
                            patch_size=patch_size, slide_window=slide_window)
        elif self.args.model == 'MIMSFCNCCATH':
            self.model = get_mimsfcncca_twoheadfcn(self.args.nclass,self.args.nc,
                            globalpool=self.args.globalpool, num_layers=self.args.num_layers,
                            patch_size=patch_size, slide_window=slide_window)
        elif self.args.model == 'MIMSFCNCCATHCCA':
            self.model = get_mimsfcncca_twoheadfcncca(self.args.nclass,self.args.nc,
                            globalpool=self.args.globalpool, num_layers=self.args.num_layers,
                            patch_size=patch_size, slide_window=slide_window)
        elif self.args.model == 'MSFCN1SATH':
            self.model = get_msfcn1sa_twoheadfcn(self.args.nclass,self.args.nc,
                            globalpool=self.args.globalpool, num_layers=self.args.num_layers,
                            patch_size=patch_size, slide_window=slide_window)
        elif self.args.model == 'MSFCN1SATHSA':
            self.model = get_msfcn1sa_twoheadfcnsa(self.args.nclass,self.args.nc,
                            globalpool=self.args.globalpool, num_layers=self.args.num_layers,
                            patch_size=patch_size, slide_window=slide_window)
        elif self.args.model == 'MIMSFCNSATHSA':
            self.model = get_mimsfcnsa_twoheadfcnsa(self.args.nclass,self.args.nc,
                            globalpool=self.args.globalpool, num_layers=self.args.num_layers,
                            patch_size=patch_size, slide_window=slide_window)
        elif self.args.model == 'MIMSFCNSATH':
            self.model = get_mimsfcnsa_twoheadfcn(self.args.nclass,self.args.nc,
                            globalpool=self.args.globalpool, num_layers=self.args.num_layers,
                            patch_size=patch_size, slide_window=slide_window)
        elif self.args.model == 'MIMSFCNPATHPA':
            self.model = get_mimsfcnpa_twoheadfcnpa(self.args.nclass,self.args.nc,
                            globalpool=self.args.globalpool, num_layers=self.args.num_layers,
                            patch_size=patch_size, slide_window=slide_window)
        elif self.args.model == 'MSFCN1PATHPA':
            self.model = get_msfcn1pa_twoheadfcnpa(self.args.nclass,self.args.nc,
                            globalpool=self.args.globalpool, num_layers=self.args.num_layers,
                            patch_size=patch_size, slide_window=slide_window)
        elif self.args.model == 'MSFCN1SEAPTHSEA':
            self.model = get_msfcn1sea_twoheadfcnsea_para(self.args.nclass,self.args.nc,
                            globalpool=self.args.globalpool, num_layers=self.args.num_layers,
                            patch_size=patch_size, slide_window=slide_window)
        elif self.args.model == 'MSFCN1SEAS1THSEA':
            self.model = get_msfcn1sea_twoheadfcnsea_serial1(self.args.nclass,self.args.nc,
                            globalpool=self.args.globalpool, num_layers=self.args.num_layers,
                            patch_size=patch_size, slide_window=slide_window)
        elif self.args.model == 'MSFCN1SEAS2THSEA':
            self.model = get_msfcn1sea_twoheadfcnsea_serial2(self.args.nclass,self.args.nc,
                            globalpool=self.args.globalpool, num_layers=self.args.num_layers,
                            patch_size=patch_size, slide_window=slide_window)
        elif self.args.model == 'MSFCN1SEAPRTHSEA':
            self.model = get_msfcn1sea_twoheadfcnsea_para_r(self.args.nclass,self.args.nc,
                            globalpool=self.args.globalpool, num_layers=self.args.num_layers,
                            patch_size=patch_size, slide_window=slide_window)
        elif self.args.model == 'MSFCN1SEAS1RTHSEA':
            self.model = get_msfcn1sea_twoheadfcnsea_serial1_r(self.args.nclass,self.args.nc,
                            globalpool=self.args.globalpool, num_layers=self.args.num_layers,
                            patch_size=patch_size, slide_window=slide_window)
        elif self.args.model == 'MSFCN1SEAS2RTHSEA':
            self.model = get_msfcn1sea_twoheadfcnsea_serial2_r(self.args.nclass,self.args.nc,
                            globalpool=self.args.globalpool, num_layers=self.args.num_layers,
                            patch_size=patch_size, slide_window=slide_window)
        elif self.args.model == 'MSFCN1PBATHPBA':
            self.model = get_msfcn1pba_twoheadfcnpba(self.args.nclass,self.args.nc,
                            globalpool=self.args.globalpool, num_layers=self.args.num_layers,
                            patch_size=patch_size, slide_window=slide_window)
        elif self.args.model == 'MSFCN1SEAP1THSEA':
            self.model = get_msfcn1sea_twoheadfcnsea_para1(self.args.nclass,self.args.nc,
                            globalpool=self.args.globalpool, num_layers=self.args.num_layers,
                            patch_size=patch_size, slide_window=slide_window)
        elif self.args.model == 'MSFCN1SEAP1RTHSEA':
            self.model = get_msfcn1sea_twoheadfcnsea_para1_r(self.args.nclass,self.args.nc,
                            globalpool=self.args.globalpool, num_layers=self.args.num_layers,
                            patch_size=patch_size, slide_window=slide_window)
        elif self.args.model == 'MSFCN1SEAP2THSEA':
            self.model = get_msfcn1sea_twoheadfcnsea_para2(self.args.nclass,self.args.nc,
                            globalpool=self.args.globalpool, num_layers=self.args.num_layers,
                            patch_size=patch_size, slide_window=slide_window)
        elif self.args.model == 'MSFCN1SEAP2RTHSEA':
            self.model = get_msfcn1sea_twoheadfcnsea_para2_r(self.args.nclass,self.args.nc,
                            globalpool=self.args.globalpool, num_layers=self.args.num_layers,
                            patch_size=patch_size, slide_window=slide_window)
        elif self.args.model == 'MIMSFCNSEAP1THSEA':
            self.model = get_mimsfcnsea_twoheadfcnsea_para1(self.args.nclass,self.args.nc,
                            globalpool=self.args.globalpool, num_layers=self.args.num_layers,
                            patch_size=patch_size, slide_window=slide_window)
        elif self.args.model == 'MIMSFCNSEAP2THSEA':
            self.model = get_mimsfcnsea_twoheadfcnsea_para2(self.args.nclass,self.args.nc,
                            globalpool=self.args.globalpool, num_layers=self.args.num_layers,
                            patch_size=patch_size, slide_window=slide_window)

        elif self.args.model == 'MIMSFCNSEVTP1THSEVT':
            self.model = get_mimsfcnsevt_twoheadfcnsevt_para1(self.args.nclass,self.args.nc,
                            globalpool=self.args.globalpool, num_layers=self.args.num_layers,
                            patch_size=patch_size, slide_window=slide_window)
        elif self.args.model == 'MIMSFCNSEVTP2THSEVT':
            self.model = get_mimsfcnsevt_twoheadfcnsevt_para2(self.args.nclass,self.args.nc,
                            globalpool=self.args.globalpool, num_layers=self.args.num_layers,
                            patch_size=patch_size, slide_window=slide_window)
        elif self.args.model == 'MIMSFCNSEVTP3THSEVT':
            self.model = get_mimsfcnsevt_twoheadfcnsevt_para3(self.args.nclass,self.args.nc,
                            globalpool=self.args.globalpool, num_layers=self.args.num_layers,
                            patch_size=patch_size, slide_window=slide_window)
        elif self.args.model == 'MIMSFCNSEVTS1THSEVT':
            self.model = get_mimsfcnsevt_twoheadfcnsevt_serial1(self.args.nclass,self.args.nc,
                            globalpool=self.args.globalpool, num_layers=self.args.num_layers,
                            patch_size=patch_size, slide_window=slide_window)
        elif self.args.model == 'MIMSFCNSEVTS2THSEVT':
            self.model = get_mimsfcnsevt_twoheadfcnsevt_serial2(self.args.nclass,self.args.nc,
                            globalpool=self.args.globalpool, num_layers=self.args.num_layers,
                            patch_size=patch_size, slide_window=slide_window)