import logging

import pytorch_lightning as pl
import torch
from src.dataset.pdf_embedding_dataset import PDFEmbeddingDataset
from src.util.model_selection import DatasetSpliter
from torch.utils.data.dataloader import DataLoader

logger = logging.getLogger('dataloader.word2vec_dataloader')


class Word2vecDataloader(pl.LightningDataModule):
    def __init__(self, context_size, nbyte, batch_size=32, shuffle=True,deterministic=True,num_workers=8):
        super().__init__()
        self.nbyte = nbyte
        self.context_size = context_size
        self.batch_size=batch_size
        self.shuffle = shuffle
        self.determ = deterministic
        self.num_workers = num_workers
    
    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        pass
    
    def setup(self):
        dataset = PDFEmbeddingDataset(self.context_size,self.nbyte)
        self.dss = DatasetSpliter(dataset, dataset.key_name, strategy='tvt', nfold=10, deterministic=self.determ)
        self.dss.setup()
        self.vocab_size = dataset.vocab_size
        self.nfold = self.dss.nfold

    def cv_gen(self) -> int:
        for i, data in enumerate(self.dss.split()):
            self._build_dataloader(*data)
            yield i
    
    def get_fold(self, fold):
        data = self.dss.get_fold(fold)
        self._build_dataloader(*data)

    def _to_dataloader(self, subset, shuffle, batch_size):
        dataloader = DataLoader(subset,
                batch_size=batch_size,
                pin_memory=True,
                drop_last=False,
                shuffle=shuffle,
                num_workers=self.num_workers)
        return dataloader

    def _build_dataloader(self, train, val, test):
        dataloaders = {}
        dataloaders['train'] = self._to_dataloader(train, self.shuffle, self.batch_size)
        dataloaders['val'] = self._to_dataloader(val, False, self.batch_size)
        dataloaders['test'] = self._to_dataloader(test, False, self.batch_size)
        self.dataloaders = dataloaders

    @property
    def train_dataloader(self) -> DataLoader:
        return self.dataloaders['train']
    
    @property
    def val_dataloader(self) -> DataLoader:
        return self.dataloaders['val']
    
    @property
    def test_dataloader(self) -> DataLoader:
        return self.dataloaders['test']
