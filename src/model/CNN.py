import logging
import torch.nn as nn

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
    def __init__(self, inplane, patch_size, dropout=0.5, pooling_type='maxpool'):
        super().__init__()

        num_block = 1
        i = patch_size[0]/4
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

        # bls += [nn.AdaptiveAvgPool2d(1),
        #         nn.Flatten(1,-1)]
            
        # if dropout:
        #     bls += [nn.Dropout(0.5)]
        
        self.model = nn.Sequential(*bls)
        self.out_dim = plane//2
        logger.info(f'##### CNN2D inplane {inplane} blocks {num_block} dropout {dropout} {pooling_type} outdim {self.out_dim}')

    def forward(self, X):
        X = self.model(X)
        return X

