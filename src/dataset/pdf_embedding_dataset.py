from torch._C import default_generator
from .base_dataset import BaseDataset
from collections import defaultdict
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch
from src.util.decorator import buffer_value
import logging

logger = logging.getLogger('dataset.PDFEmbeddingDataset')

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

def pdf_to_ngram(bytes_array, nbyte, context_size):
    integer_array = byte_stream_to_int_array(bytes_array, nbyte)
    ngrams = zip(*[integer_array[i:] for i in range(context_size)])
    return ngrams

class DataReader:
    NEGATIVE_TABLE_SIZE = 1e8

    def __init__(self, min_count):

        self.negatives = []
        self.discards = []
        self.negpos = 0

        self.word2id = dict()
        self.id2word = dict()
        self.sentences_count = 0
        self.token_count = 0
        self.word_frequency = dict()

        self.data_by_pdf_file = []

        self.read_words(min_count)
        self.initTableNegatives()
        self.initTableDiscards()

    def pdf_to_skipgram(self, bytes_array, nbyte, context_size):
        integer_array = byte_stream_to_int_array(bytes_array, nbyte)
        array_len = len(integer_array)
        training_data = []
        word_frequency = defaultdict(int)

        for i in range(array_len):
            word_frequency[integer_array[i]] += 1
            w_target = integer_array[i]
            w_context = []
            for j in range(i - context_size, i + context_size + 1):
                if j != i and j <= array_len and j >= 0:
                    w_context.append(integer_array[i])

            training_data.append([w_target, w_context])
        self.data_by_pdf_file.append(training_data)
        self.word_frequency.append(word_frequency)

    def summary_reading(self):
        word_frequency = defaultdict(int)
        for wf in self.word_frequency:
            for w,c in wf.items():
                word_frequency[w] += c
        self.word_frequency = word_frequency

    def read_words(self, min_count):
        word_frequency = dict()
        for line in open(self.inputFileName, encoding="utf8"):
            line = line.split()
            if len(line) > 1:
                self.sentences_count += 1
                for word in line:
                    if len(word) > 0:
                        self.token_count += 1
                        word_frequency[word] = word_frequency.get(word, 0) + 1

                        if self.token_count % 1000000 == 0:
                            print("Read " + str(int(self.token_count / 1000000)) + "M words.")

        wid = 0
        for w, c in word_frequency.items():
            if c < min_count:
                continue
            self.word2id[w] = wid
            self.id2word[wid] = w
            self.word_frequency[wid] = c
            wid += 1
        print("Total embeddings: " + str(len(self.word2id)))

    def initTableDiscards(self):
        t = 0.0001
        f = np.array(list(self.word_frequency.values())) / self.token_count
        self.discards = np.sqrt(t / f) + (t / f)

    def initTableNegatives(self):
        pow_frequency = np.array(list(self.word_frequency.values())) ** 0.5
        words_pow = sum(pow_frequency)
        ratio = pow_frequency / words_pow
        count = np.round(ratio * DataReader.NEGATIVE_TABLE_SIZE)
        for wid, c in enumerate(count):
            self.negatives += [wid] * int(c)
        self.negatives = np.array(self.negatives)
        np.random.shuffle(self.negatives)

    def getNegatives(self, target, size):  # TODO check equality with target
        response = self.negatives[self.negpos:self.negpos + size]
        self.negpos = (self.negpos + size) % len(self.negatives)
        if len(response) != size:
            return np.concatenate((response, self.negatives[0:self.negpos]))
        return response

@buffer_value(protocol='joblib',folder='temporary')
def load_pdf(path, plot, **param):
    outputs = []
    n, gen = get_iterate_folder_gen(path)
    for cont in tqdm(gen(),total=n):
        output = plot(cont, **param)
        outputs.append(output)
        
    return outputs

@buffer_value(protocol='joblib',folder='temporary')
def load_pdf_embedding(path, plot, **param):
    outputs = []
    dr = DataReader(1)
    n, gen = get_iterate_folder_gen(path)
    for cont in tqdm(gen(),total=n):
        dr.pdf_to_skipgram(cont, **param)
        #output = plot(cont, **param)
    
    dr.summary_reading()
    dr.getNegatives()

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

class PDFEmbeddingDataset(BaseDataset):
    def __init__(self, context_size, nbyte=1):
        """
        nbyte: number of bytes for each pixel
        """
        super().__init__()
        self.name = 'pdf_dataset'
        self.nbyte = nbyte
        self.context_size = context_size
        rpath = get_project_root()
        self.dpath = os.path.join(rpath,'dataset')
        logger.info('Setup PDF Dataset')
        self._load()
        logger.info('Setup PDF Dataset Finished')

        self.vocab = set(range(nbyte * 256))
        self.vocab_size = len(self.vocab)

    @property
    def key_name(self):
        return f'{self.name}_word2vec_nb={self.nbyte}'

    def _load(self):
        pclean = os.path.join(self.dpath,'CLEAN_PDF_9000_files')
        pmal = os.path.join(self.dpath,'MALWARE_PDF_PRE_04-2011_10982_files')

        data_by_class = {'benign':None,'malicious':None}
        nbyte = self.nbyte
        if self.data_mode == 'ngram':
            data_by_class['benign'] = load_pdf(f'ngram_c=b_nb={nbyte}', pclean, pdf_to_ngram, nbyte=nbyte, context_size=self.context_size)
            data_by_class['malicious'] = load_pdf(f'ngram_c=m_nb={nbyte}', pmal, pdf_to_ngram, nbyte=nbyte, context_size=self.context_size)
        elif self.data_mode == 'skipgram':
            data_by_class['benign'] = load_pdf(f'skipgram_c=b_nb={nbyte}', pclean, pdf_to_skipgram, nbyte=nbyte, context_size=self.context_size)
            data_by_class['malicious'] = load_pdf(f'skipgram_c=m_nb={nbyte}', pmal, pdf_to_skipgram, nbyte=nbyte, context_size=self.context_size)
        
        ### create input X
        X = []
        for k in data_by_class.keys():
            X.extend(data_by_class[k])
            print(data_by_class[k])
        self.X = X

        ### create labels
        labels = []
        for k in data_by_class.keys():
            n = len(data_by_class[k])
            label = np.repeat(k,n)
            labels.extend(label)
        index, _ = self._find_class_(labels,one_hot=False)
        self.labels = index

        ### to tensor
        self._to_tensor()

    def _to_tensor(self):
        self.X = [torch.tensor(r,dtype=torch.long) for r in self.X]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.X[idx][0:self.context_size], self.X[idx][-1]