import os
import sys
from pathlib import Path
cur_path = Path(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(str(cur_path.parent))
from src.model.CNN_LSTM import CNN_LSTM
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
    filehandler = logging.FileHandler(f"logging/cnnlstm_{jobid}_{time.strftime('%Y-%m-%d-%H:%M:%S')}_{os.getpid()}")
else:
    filehandler = logging.FileHandler(f"logging/cnnlstm_{time.strftime('%Y-%m-%d-%H:%M:%S')}_{os.getpid()}")
filehandler.setFormatter(formatter)
logger.addHandler(filehandler)

def pdf_dataset_generate_tem():
    from src.dataset.pdf_dataset import PDFDataset
    dataset = PDFDataset('byte')

if __name__ == '__main__':
    pdf_dataset_generate_tem()