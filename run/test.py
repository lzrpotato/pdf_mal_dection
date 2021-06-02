import os
import sys
from pathlib import Path
cur_path = Path(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(str(cur_path.parent))
from src.model.CNN_LSTM import CNN_LSTM
import torch
import logging

logger = logging.getLogger()
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

logger.info('test')

def test_cnn_lstm():
    i = torch.ones((1,1,2048000),dtype=torch.float32)
    tsm = CNN_LSTM(2,1,64,1,1)

def test_cnnlstmtrainer():
    from src.trainer.cnnlstm_trainer import train, setup_arg
    args = setup_arg()
    if args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    train(args)
    

if __name__ == '__main__':
    test_cnnlstmtrainer()