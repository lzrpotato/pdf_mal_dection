import sys
sys.path.append('.')
import os
print(os.getcwd())
from src.model.ts_cnn import CNNTimeSeries
import torch.nn as nn
import torch
import logging

logger = logging.getLogger('model.TSModel')

__all__ = ['TimeSeriesModel']


class TimeSeriesModel(nn.Module):
    def __init__(self, nclass, nc, max_len):
        super().__init__()
        self.nclass = nclass
        self.cnn_repre = CNNTimeSeries(feature_d=nc,seq_max_len=max_len)
        self.classifier = self.make_classifier(self.cnn_repre.out_dim,nclass)

    def make_classifier(self, hidden_size, nclass, layer_num=1):
        layers = []
        sz = hidden_size
        
        for l in range(layer_num-1):
            layers += [nn.Linear(sz, sz//2)]
            layers += [nn.ReLU(True)]
            layers += [nn.Dropout()]
            sz //= 2
        
        layers += [nn.Dropout(0.5)]
        layers += [nn.Linear(sz, nclass)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.cnn_repre(x.view(x.size(0),1,-1))
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    i = torch.ones((32,2048000))
    tsm = TimeSeriesModel(2,1,2048000)
    print(tsm(i).shape)