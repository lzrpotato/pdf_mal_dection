import os
import sys
from pathlib import Path
cur_path = Path(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(str(cur_path.parent))
from src.model.PCNN_LSTM import PCNN_LSTM
import torch
import logging
import time


jobid = os.environ.get('SLURM_JOB_ID')


logger = logging.getLogger()

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(message)s')

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(formatter)
logger.addHandler(handler)

if not os.path.isdir('logging'):
    os.mkdir('logging')

if jobid is not None:
    filehandler = logging.FileHandler(f"logging/pcnnlstm_{jobid}_{time.strftime('%Y-%m-%d-%H:%M:%S')}_{os.getpid()}")
else:
    filehandler = logging.FileHandler(f"logging/pcnnlstm_{time.strftime('%Y-%m-%d-%H:%M:%S')}_{os.getpid()}")
filehandler.setFormatter(formatter)
logger.addHandler(filehandler)

def test_cnn_lstm():
    i = torch.ones((1,1,2048000),dtype=torch.float32)
    tsm = PCNN_LSTM(2,1,64,1,1)

def test_pcnnlstmtrainer():
    from src.trainer.pcnnlstm_trainer import train, setup_arg
    args = setup_arg()
    if args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    train(args)
    

if __name__ == '__main__':
    test_pcnnlstmtrainer()