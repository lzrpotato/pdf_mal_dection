from gensim.test.utils import common_texts
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import os
from tqdm import tqdm
from gensim.models.callbacks import CallbackAny2Vec
from numpy.lib.stride_tricks import sliding_window_view
import numpy as np
import logging
import argparse
import time
import io

class NoGensimWord2vec(logging.Filter):
    def filter(self, record: logging.LogRecord):
        return not ('gensim.models.word2vec' in record.name)

class TqdmToLogger(io.StringIO):
    """
        Output stream for TQDM which will output to logger module instead of
        the StdOut.
    """
    logger = None
    level = None
    buf = ''
    def __init__(self,logger,level=None):
        super(TqdmToLogger, self).__init__()
        self.logger = logger
        self.level = level or logging.INFO
    def write(self,buf):
        self.buf = buf.strip('\r\n\t ')
    def flush(self):
        self.logger.log(self.level, self.buf)

logger = logging.getLogger()
formatter = logging.Formatter('%(process)d -%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(message)s')

#handler = logging.StreamHandler(sys.stdout)
#handler.setFormatter(formatter)
#logger.addHandler(handler)

if not os.path.isdir('logging'):
    os.mkdir('logging')
filehandler = logging.FileHandler(f"logging/gensim_word2vec_{time.strftime('%Y-%m-%d-%H:%M:%S')}_{os.getpid()}")
filehandler.setFormatter(formatter)
filehandler.addFilter(NoGensimWord2vec())
logger.addHandler(filehandler)

##### tqdm handler

logger.setLevel(logging.DEBUG)
#logger.addFilter(NoGensimWord2vec())

class MyCorpus:
    def __init__(self, sentence_len, sentence_stride):
        self.corpus_paths = ['dataset/CLEAN_PDF_9000_files','dataset/MALWARE_PDF_PRE_04-2011_10982_files']
        #self.corpus_paths = ['dataset/CLEAN_PDF_9000_files']
        self.nbyte = 1
        self.sentence_len = sentence_len
        self.sentence_stride = sentence_stride

    def _get_iterate_folder_gen(self, path):
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
                yield c,fn
        return count, gen

    def _byte_stream_to_int_array(self,bytes_array, nbyte):
        n = len(bytes_array)
        rem = n % nbyte
        npad = nbyte - rem
        # padding bytearray with zero
        bytes_array += bytes([0]*npad)
        integer_array = np.frombuffer(bytes_array, dtype=np.dtype(f'uint{nbyte*8}'))
        return integer_array

    def _pdf2sentences(self, content, nbyte, sentence_len, stride):
        integer_array = self._byte_stream_to_int_array(content,nbyte)
        sentences = sliding_window_view(integer_array, window_shape=sentence_len)[::stride,:]
        sentences = sentences.tolist()
        return sentences

    def __iter__(self):
        for path in self.corpus_paths:
            n, gen = self._get_iterate_folder_gen(path)
            tqdm_out = TqdmToLogger(logger,level=logging.INFO)
            logger.info('[iter] {}'.format(path))
            
            for cont, fn in tqdm(gen(),total=n,disable=False,file=tqdm_out):
                sentences = self._pdf2sentences(cont,self.nbyte,self.sentence_len, self.sentence_stride)
                for s in sentences:
                    yield s

class MonitorCallback(CallbackAny2Vec):
    def __init__(self):
        pass

    def on_epoch_end(self, model):
        logging.info("Model loss:{}".format(model.get_latest_training_loss()))  # print loss

def setup_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sg', type=int, default=0,
        help='Training algorithm: cbow=0 or skipgram=1')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--vector-size', type=int, default=4)
    parser.add_argument('--window', type=int, default=5)
    parser.add_argument('--workers',type=int, default=1)
    parser.add_argument('--sentence_len',type=int,default=20)
    parser.add_argument('--sentence_stride',type=int,default=10)

    return parser.parse_args()

args = setup_arg()

logger.info('[Word2vec args] {}'.format(args))
mc = MyCorpus(args.sentence_len,args.sentence_stride)
monitorcallback = MonitorCallback()

model = Word2Vec(sentences=mc,
            vector_size=args.vector_size,
            window=args.window,
            min_count=1,
            workers=args.workers,
            sg=args.sg,
            callbacks=[monitorcallback]
            )

if not os.path.isdir('embedding_weight/'):
    os.mkdir('embedding_weight/')

if args.sg == 0:
    mode = 'cbow'
elif args.sg == 1:
    mode = 'skipgram'

model.save('embedding_weight/pdf_malware_word2vec_{}.model'.format(mode))
model.wv.save('embedding_weight/pdf_malware_word2vec_{}.wordvectors'.format(mode))