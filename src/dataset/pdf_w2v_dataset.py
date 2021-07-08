from .base_dataset import BaseDataset
import os
from pathlib import Path
from typing import Dict
from gensim.models import KeyedVectors
import numpy as np
from tqdm import tqdm
import torch
from src.util.decorator import buffer_value
import logging

logger = logging.getLogger('dataset.pdfw2vdataset')

def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent

def byte_stream_to_int_array(bytes_array, nbyte):
    n = len(bytes_array)
    rem = n % nbyte
    npad = nbyte - rem
    # padding bytearray with zero
    bytes_array += bytes([0]*npad)
    integer_array = np.frombuffer(bytes_array, dtype=np.dtype(f'uint{nbyte*8}'))
    return integer_array

def embedding_plot(bytes_array, keyvectors, nbyte):
    int_array = byte_stream_to_int_array(bytes_array, nbyte)
    embedding_array = keyvectors[int_array]
    return embedding_array

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

class PDFW2VDataset(BaseDataset):
    def __init__(self, plot_strategy, nbyte=1):
        """
        nbyte: number of bytes for each pixel
        """
        super().__init__()
        self.name = 'pdf_w2v_dataset'
        self.nbyte = nbyte
        rpath = get_project_root()
        self.dpath = os.path.join(rpath,'dataset')
        self.plot_strategy = plot_strategy
        self.keyvectors = self.load_keyvectors(plot_strategy)
        logger.info('Setup PDF Dataset')
        self._load()
        logger.info('Setup PDF Dataset Finished')

    def load_keyvectors(self, plot_strategy):
        kw = None
        if plot_strategy == 'word2vec_skipgram':
            kw = KeyedVectors.load('embedding_weight/pdf_malware_word2vec_skipgram.wordvectors')
        elif plot_strategy == 'word2vec_cbow':
            kw = KeyedVectors.load('embedding_weight/pdf_malware_word2vec_skipgram.wordvectors')
        
        return kw

    @property
    def key_name(self):
        return f'{self.name}_{self.plot_strategy}_nb={self.nbyte}'

    def _load(self):
        pclean = os.path.join(self.dpath,'CLEAN_PDF_9000_files')
        pmal = os.path.join(self.dpath,'MALWARE_PDF_PRE_04-2011_10982_files')

        nbyte = self.nbyte

        data_by_class = {'benign':None,'malicious':None}
        if self.plot_strategy == 'word2vec_skipgram':
            data_by_class['benign'] = load_pdf(f'w2vskipgram_c=b_nb={nbyte}', pclean, embedding_plot, keyvectors=self.keyvectors, nbyte=nbyte)
            data_by_class['malicious'] = load_pdf(f'w2vskipgram_c=m_nb={nbyte}', pmal, embedding_plot, keyvectors=self.keyvectors, nbyte=nbyte)
            logger.info(f'Using {self.plot_strategy} plot strategy with byte size {nbyte}')
        elif self.plot_strategy == 'word2vec_cbow':
            data_by_class['benign'] = load_pdf(f'w2vcbow_c=b_nb={nbyte}',pclean, embedding_plot, keyvectors=self.keyvectors, nbyte=nbyte)
            data_by_class['malicious']  = load_pdf(f'w2vcbow_c=m_nb={nbyte}',pmal, embedding_plot, keyvectors=self.keyvectors, nbyte=nbyte)
            logger.info(f'Using {self.plot_strategy} plot strategy with byte size {nbyte}')


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
        self.nc = self.X[0].shape[1]

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
