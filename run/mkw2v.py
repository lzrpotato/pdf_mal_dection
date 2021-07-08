import os
import sys
from pathlib import Path
cur_path = Path(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(str(cur_path.parent))
import logging

logger = logging.getLogger()
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.info('test')


def mk_pdfw2v():
    from src.dataset.pdf_w2v_dataset import PDFW2VDataset
    logger.info('make word2vec_cbow')
    dataset = PDFW2VDataset('word2vec_cbow')
    logger.info('finish word2vec_cbow')
    logger.info('make word2vec_skipgram')
    dataset = PDFW2VDataset('word2vec_skipgram')
    logger.info('finish word2vec_skipgram')
    

if __name__ == '__main__':
    mk_pdfw2v()