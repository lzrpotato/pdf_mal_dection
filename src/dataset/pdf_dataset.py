from .base_dataset import BaseDataset
import os
from pathlib import Path
from typing import Dict
import numpy as np
from tqdm import tqdm
import torch
from src.util.decorator import buffer_value
import logging

logger = logging.getLogger('dataset.pdfdataset')

def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent

def byte_plot(bytes_array, nbyte: int = 1) -> np.ndarray:
    n = len(bytes_array)
    rem = n % nbyte
    n -= rem
    byte_map = np.frombuffer(bytes_array[0:n], dtype=np.dtype(f'i{nbyte}'))
    return byte_normal(byte_map, nbyte)

def byte_normal(byte_map, nbyte):
    nbm = byte_map / ((256//nbyte)-1)
    return nbm

def transition_matrix(transitions) -> np.ndarray:
    n = 1+ max(transitions) #number of states

    M = [[0]*n for _ in range(n)]

    for (i,j) in zip(transitions,transitions[1:]):
        M[i][j] += 1

    #now convert to probabilities:
    for row in M:
        s = sum(row)
        if s > 0:
            row[:] = [f/s for f in row]
    return np.array(M)

def markov_plot(bytes_array):
    M = transition_matrix(bytes_array)
    return M

@buffer_value(protocol='joblib',folder='temporary')
def load_pdf(path, plot, **param):
    outputs = []
    n, gen = get_iterate_folder_gen(path)
    for cont in tqdm(gen(),total=n):
        output = plot(cont, **param)
        outputs.append(output)
        
    return outputs

def get_iterate_folder_gen(path):
    count = 0
    for fn in os.listdir(path):
        if fn == 'log.txt':
                    continue
        count += 1
    
    def gen():
        for fn in os.listdir(path):
            if fn == 'log.txt':
                continue
            c = None
            with open(os.path.join(path,fn),'rb') as f:
                c = f.read()
            yield c
    return count, gen

@buffer_value('joblib','temporary')
def padding(X):
    max = 0
    lengths = []
    for r in X:
        lengths.append(len(r))
        if len(r) > max:
            max = len(r)

    logger.debug(f'feature statistic: min {np.min(lengths)} max {np.max(lengths)} std {np.std(lengths)} avg {np.average(lengths)}')
    new_X = []
    for r in X:
        if len(r) < max:
            new_X.append(np.pad(r,(0,max-len(r))))
        else:
            new_X.append(r)
        
    return new_X

class PDFDataset(BaseDataset):
    def __init__(self, plot_strategy, nbyte=1):
        """
        nbyte: number of bytes for each pixel
        """
        super().__init__()
        self.name = 'pdf_dataset'
        self.nbyte = nbyte
        rpath = get_project_root()
        self.dpath = os.path.join(rpath,'dataset')
        self.plot_strategy = plot_strategy
        logger.info('Setup PDF Dataset')
        self._load()
        logger.info('Setup PDF Dataset Finished')

    @property
    def key_name(self):
        if self.plot_strategy == 'byte':
            return f'{self.name}_{self.plot_strategy}_w={self.nbyte}'
        else:
            return f'{self.name}_{self.plot_strategy}'

    def _load(self):
        pclean = os.path.join(self.dpath,'CLEAN_PDF_9000_files')
        pmal = os.path.join(self.dpath,'MALWARE_PDF_PRE_04-2011_10982_files')

        data_by_class = {'benign':None,'malicious':None}
        if self.plot_strategy == 'byte':
            nbyte = self.nbyte
            data_by_class['benign'] = load_pdf(f'byte_c=b_nb={nbyte}', pclean, byte_plot, nbyte=nbyte)
            data_by_class['malicious'] = load_pdf(f'byte_c=m_nb={nbyte}', pmal, byte_plot, nbyte=nbyte)
            logger.info(f'Using {self.plot_strategy} plot strategy with byte size {nbyte}')
        elif self.plot_strategy == 'markov':
            data_by_class['benign'] = load_pdf('markov_c=b',pclean, markov_plot)
            data_by_class['malicious']  = load_pdf('markov_c=m',pmal, markov_plot)
            logger.info(f'Using {self.plot_strategy} plot strategy')

        ### class length
        nb = len(data_by_class['benign'])
        nm = len(data_by_class['malicious'])

        ### create labels
        labels = []
        for k in data_by_class.keys():
            n = len(data_by_class[k])
            label = np.repeat(k,n)
            labels.extend(label)
        index, _ = self._find_class_(labels,one_hot=False)
        self.labels = index

        ### create input X
        X = []
        for k in data_by_class.keys():
            X.extend(data_by_class[k])
        self.X = X
        #self.X = padding(f'padded_{self.key_name}',X)

        ### feature size
        self.fea_size = self.X[0].shape[0]
        ### channel size
        self.nc = 1

        logger.info(f'feature size {self.fea_size}, label size {len(labels)}, benign {nb} mal {nm}')

        ### to tensor
        self._to_tensor()

    def _to_tensor(self):
        self.X = [torch.tensor(r,dtype=torch.float32) for r in self.X]
        self.labels = [torch.tensor(r,dtype=torch.long) for r in self.labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.X[idx], self.labels[idx]