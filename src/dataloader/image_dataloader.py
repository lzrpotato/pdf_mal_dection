import logging

import pytorch_lightning as pl
from src.dataset.byte_plot_dataset import PDFDataset
from src.dataset.markov_plot_dataset import PDFDatasetNoIn
from src.util.model_selection import DatasetSpliter
from torch.utils.data.dataloader import DataLoader

logger = logging.getLogger('dataloader.image_dataloader')



class ImageDataLoader(pl.LightningDataModule):
    def __init__(self, dataset_name, batch_size=32, nfold=10, shuffle=True,deterministic=True,num_workers=8):
        super().__init__()
        self.dataset_name = dataset_name
        self.batch_size=batch_size
        self.nfold = nfold
        self.shuffle = shuffle
        self.determ = deterministic
        self.num_workers = num_workers
    
    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        pass
    
    def setup(self):
        if self.dataset_name == 'markov':
            dataset = PDFDatasetNoIn(nbyte=1)
        elif self.dataset_name == 'byte':
            dataset = PDFDataset(width=256)
        self.dss = DatasetSpliter(dataset, dataset.key_name, strategy='tvt', nfold=self.nfold, deterministic=self.determ)
        self.dss.setup()
        self.nclass = dataset.nclass
        self.nc = dataset.nc
        self.class_to_index = dataset.class_to_index
        self.dataset = dataset

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
