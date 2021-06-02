import sys

sys.path.append('.')
import os

print(os.getcwd())
import logging

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

logger = logging.getLogger('model.CNN_LSTM')

__all__ = ['CNN_LSTM']

class CNN_BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel, stride, padding=0, pool=True, pooling_type='maxpool', pool_fix_size=False):
        super(CNN_BasicBlock,self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(True)
        self.pool = pool
        self.factor = 2
        if self.pool:
            self.factor = 4
            if pool_fix_size:
                kernel = 3
                stride = 3
                padding = 1
                self.factor = 4
            if pooling_type == 'avgpool':
                self.pool = nn.AvgPool2d(kernel_size=kernel, stride=stride, padding=padding)
            elif pooling_type == 'maxpool':
                self.pool = nn.MaxPool2d(kernel_size=kernel, stride=stride, padding=padding)
            
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.pool:
            x = self.pool(x)
        return x

class CNN2D(nn.Module):
    def __init__(self, inplane, grid_size, dropout=0.5, pooling_type='maxpool'):
        super().__init__()

        num_block = 1
        i = grid_size[0]/4
        while i/4 > 1:
            i /= 4
            num_block += 1

        logger.debug('nb {}'.format(num_block))
        in_p = inplane
        plane = 32
        bls = []
        factors = 1
        for i in range(num_block):
            if i == num_block - 1:
                pool = False
            else:
                pool = True

            kernel, stride, pad = 3, 2, 1

            cnn_basic = CNN_BasicBlock(in_p,plane,kernel,stride,pad,pool,pooling_type)
            factors *= cnn_basic.factor
            bls += [cnn_basic]
            in_p = plane
            plane *= 2

        bls += [nn.AdaptiveAvgPool2d(1),
                nn.Flatten(1,-1)]
            
        if dropout:
            bls += [nn.Dropout(0.5)]
        
        self.model = nn.Sequential(*bls)
        self.out_dim = plane//2
        logger.info(f'##### CNN2D inplane {inplane} blocks {num_block} dropout {dropout} {pooling_type} outdim {self.out_dim}')

    def forward(self, X):
        X = self.model(X)
        return X


def get_lstm(input_size,hidden_size=32,num_layers=1,batch_first=True,dropout=0.5,bidirectional=False):
    return nn.LSTM(input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional)

class LSTM_rnn(nn.Module):
    def __init__(self,
        in_size,
        hidden,
        num_layers,
        batch_size,
        pool,
    ):
        super().__init__()
        self.in_size = in_size
        self.hidden = hidden
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.pool = pool
        self.lstm = get_lstm(self.in_size,self.hidden,self.num_layers)
        self.h0 = torch.zeros((self.num_layers,self.batch_size,self.hidden),dtype=torch.float32)
        self.c0 = torch.zeros((self.num_layers,self.batch_size,self.hidden),dtype=torch.float32)

        if pool:
            self.pooling = nn.AdaptiveAvgPool1d(1)
        
        self.flatten = nn.Flatten(1,-1)
        self.out_dim = self.hidden

    def forward(self, x):
        h0 = self.h0.to(x.device)
        c0 = self.c0.to(x.device)
        logger.debug('lstm forward {} {}'.format(x.size(), h0.size()))
        out, (hn,cn) = self.lstm(x,(h0,c0))
        if self.pool:
            logger.debug('lstm out {}'.format(out.size()))
            batch, seq, hidden = out.size()
            out = self.pooling(out.view(batch,hidden,seq))
        else:
            out = hn
        
        out = self.flatten(out)
        return out

class CNN_LSTM(nn.Module):
    def __init__(self, 
        nclass, 
        nc, 
        hidden, 
        batch_size, 
        num_layers,
        grid_size=(256,256)
    ):
        super().__init__()
        self.nclass = nclass
        self.nc = nc
        self.hidden = hidden
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.grid_size = grid_size
        # input C_in, W, H, output C_out, W_o, H_o
        num_block = grid_size[0]//4
        self.cnn_repre = CNN2D(inplane=nc,grid_size=grid_size)
        # input 
        self.lstm = LSTM_rnn(self.cnn_repre.out_dim,self.hidden,self.num_layers,1,pool=True)
        
        self.classifier = self.make_classifier(self.lstm.out_dim,nclass)

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

    def _pad_to_grid(self, seq: torch.Tensor):
        batch, c, seq_len = seq.size()
        h,w = self.grid_size
        need = h*w - seq_len % (h*w)
        logger.debug('need {}'.format(need))
        seq = F.pad(seq,(0,need))
        return seq
    
    def _view_to_grid(self, seq: torch.Tensor):
        batch, c, seq_len = seq.size()
        h,w = self.grid_size
        ng = seq_len // (h*w)
        return seq.view(batch*ng, c, h,w), ng

    def forward_(self, x):
        batch, c, seq_len = x.size()
        logger.debug('{} {} {}'.format(batch, c, seq_len))
        x = self._pad_to_grid(x)
        c_in, ng = self._view_to_grid(x)
        c_out = self.cnn_repre(c_in)
    
        r_in = c_out.view(batch,ng,-1)
        logger.debug('in {} cout {} rin {}'.format(c_in.size(), c_out.size(), r_in.size()))
        r_out = self.lstm(r_in)
        logger.debug('r_out {}'.format(r_out.size()))
        x = self.classifier(r_out)

        return x
    
    def forward(self, x):
        batch = len(x)
        r_outs = []
        for i in x:
            seq_len = i.size()[0]
            logger.debug('{}'.format(seq_len))
            xi = i.view(1,self.nc,seq_len)
            xi = self._pad_to_grid(xi)
            c_in, ng = self._view_to_grid(xi)
            c_out = self.cnn_repre(c_in)
        
            r_in = c_out.view(1,ng,-1)
            logger.debug('in {} cout {} rin {}'.format(c_in.size(), c_out.size(), r_in.size()))
            r_out = self.lstm(r_in)
            logger.debug('r_out {}'.format(r_out.size()))
            r_outs.append(r_out)
        
        r_outs = torch.stack(r_outs).view(batch,-1)
        logger.debug('r_outs {}'.format(r_outs.size()))
        output = self.classifier(r_outs)
        
        return output

if __name__ == '__main__':
    i = torch.ones((1,1,2048000))
    tsm = CNN_LSTM(2,1)
    print(tsm(i).shape)
