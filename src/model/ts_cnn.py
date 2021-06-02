import torch.nn as nn
import logging

logger = logging.getLogger('model.ts_cnn')

__all__ = ['CNNTimeSeries']


class CNN_BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel, stride, padding=0, pool=True, pooling_type='maxpool', pool_fix_size=False):
        super(CNN_BasicBlock,self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=kernel, stride=stride, padding=padding)
        self.bn = nn.BatchNorm1d(planes)
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
                self.pool = nn.AvgPool1d(kernel_size=kernel, stride=stride, padding=padding)
            elif pooling_type == 'maxpool':
                self.pool = nn.MaxPool1d(kernel_size=kernel, stride=stride, padding=padding)
            
    def forward(self, x):
        print(f'cnn basic input {x.shape}')
        print(x)
        x = self.conv1(x)
        #print(f'conv1 input {x.shape}')
        x = self.bn(x)
        x = self.relu(x)
        if self.pool:
            x = self.pool(x)
            #print(f'avgpool input {x.shape}')
        #print(f'cnn basic output {x.shape}')
        return x


class CNNTimeSeries(nn.Module):
    def __init__(self, feature_d, seq_max_len, fst_p=8, dropout=True, pooling_type='maxpool'):
        super().__init__()
        nb = 0
        
        i = seq_max_len/4
        while i/4 > 1:
            i /= 4
            nb += 1
        nb = 4
        self.unflatten = nn.Unflatten(1, (feature_d,seq_max_len))  # b,1,self.feature_dim*self.max_tree_length
        
        inplanes = feature_d
        planes = fst_p
        bls = []
        factors = 1
        k_ = 3+(nb-1)*2
        # kernels 9, 7, 5, 3
        for i in range(int(nb)):
            if i == nb - 1:
                pool = False
            else:
                pool = True
            #print('CNN_ ',inplanes,planes)
            kernel, stride, pad = k_,2,(k_-1)//2
            # padding 4,3,2,1
            cnnbasic = CNN_BasicBlock(inplanes,planes,kernel,stride,pad,pool=pool,pooling_type=pooling_type)
            factors *= cnnbasic.factor
            bls += [cnnbasic]
            inplanes = planes
            planes *= 2
            k_ -= 2
        
        bls += [nn.AdaptiveAvgPool1d(1),
                nn.Flatten(1,-1)]

        if dropout:
            bls += [nn.Dropout(0.5)]

        self.time_series_layers = nn.Sequential(*bls)
        logger.debug(f'##### CNN_AVG block {nb} max_tree_len {seq_max_len} ###### ')
        self.out_dim = inplanes
    
    def forward(self, x):
        x = self.time_series_layers(x)
        return x

