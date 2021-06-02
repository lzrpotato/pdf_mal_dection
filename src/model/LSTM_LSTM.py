import sys

sys.path.append('.')
import os

print(os.getcwd())
import logging

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

logger = logging.getLogger('model.LSTM_LSTM')

__all__ = ['LSTM_LSTM']

def get_lstm(input_size,hidden_size=32,num_layers=1,batch_first=True,dropout=0.5,bidirectional=False):
    return nn.LSTM(input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional)

class LSTM_1stage(nn.Module):
    def __init__(self,
        in_size,
        hidden,
        num_layers,
        batch_size,
    ):
        super().__init__()
        self.in_size = in_size
        self.hidden = hidden
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.lstm = get_lstm(self.in_size, self.hidden,self.num_layers)
        self.h0 = torch.zeros((self.batch_size,self.num_layers,self.hidden),dtype=torch.float32)
        self.c0 = torch.zeros((self.batch_size,self.num_layers,self.hidden),dtype=torch.float32)

        self.out_dim = self.hidden

    def forward(self, x):
        h0 = self.h0.to(x.device)
        c0 = self.c0.to(x.device)
        out, (hn,cn) = self.lstm(x,(h0,c0))
        return out

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
        self.h0 = torch.zeros((self.batch_size,self.num_layers,self.hidden),dtype=torch.float32)
        self.c0 = torch.zeros((self.batch_size,self.num_layers,self.hidden),dtype=torch.float32)

        if pool:
            #self.pooling = nn.AdaptiveMaxPool1d(1)
            self.pooling = nn.AdaptiveAvgPool1d(1)
        
        self.flatten = nn.Flatten(1,-1)
        self.out_dim = self.hidden

    def forward(self, x):
        logger.debug('lstm forward {}'.format(x.size()))
        h0 = self.h0.to(x.device)
        c0 = self.c0.to(x.device)
        out, (hn,cn) = self.lstm(x,(h0,c0))
        if self.pool:
            logger.debug('lstm out {}'.format(out.size()))
            batch, seq, hidden = out.size()
            out = self.pooling(out.view(batch,hidden,seq))
        else:
            out = hn
        
        out = self.flatten(out)
        return out

class LSTM_LSTM(nn.Module):
    def __init__(self, 
        nclass, 
        nc, 
        hidden, 
        seq_len,
        batch_size,
        num_layers,
    ):
        super().__init__()
        self.nclass = nclass
        self.nc = nc
        self.hidden = hidden
        self.batch_size = batch_size
        self.num_layers = num_layers
        # input 
        self.lstm_1 = LSTM_rnn(self.hidden)
        # input 
        self.lstm_2 = LSTM_rnn(self.self.lstm_1.out_dim,self.hidden,self.num_layers,1,pool=True)
        
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
