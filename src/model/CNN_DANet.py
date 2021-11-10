import sys

from torch.nn.modules.sparse import Embedding

sys.path.append('.')
import os

print(os.getcwd())
import logging

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

logger = logging.getLogger('model.CNN_DANet')

__all__ = ['CNN_DANet']

class CNN_BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel, stride, padding=0, pool=True, pooling_type='maxpool', pool_fix_size=False):
        super(CNN_BasicBlock,self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel, stride=stride, padding=padding)
        #self.bn = nn.BatchNorm2d(planes)
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
        #x = self.bn(x)
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

class PAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        proj_query = self.query_conv(x).permute(0, 2, 1)
        proj_key = self.key_conv(x)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = self.gamma*out + x
        return out


class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """

        proj_query = x
        proj_key = x.permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x

        out = torch.bmm(attention, proj_value)
        
        out = self.gamma*out + x
        return out


class DualAttn(nn.Module):
    def __init__(self,
        in_channels, 
        out_channels,
        norm_layer,
        pool='maxpool',
        dropout=0.1,
    ):
        super().__init__()
        self.pool = pool
        inter_channels = in_channels // 4
        self.conv5a = nn.Sequential(nn.Conv1d(in_channels, inter_channels, 1, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())
        self.conv5c = nn.Sequential(nn.Conv1d(in_channels, inter_channels, 1, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())
        
        self.sa = PAM_Module(inter_channels)
        self.sc = CAM_Module(inter_channels)
        self.conv51 = nn.Sequential(nn.Conv1d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv1d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())

        self.conv6 = nn.Sequential(nn.Dropout(0.1, False), nn.Conv1d(inter_channels, out_channels, 1))
        self.conv7 = nn.Sequential(nn.Dropout(0.1, False), nn.Conv1d(inter_channels, out_channels, 1))

        self.conv8 = nn.Sequential(nn.Dropout(0.1, False), nn.Conv1d(inter_channels, out_channels, 1))
        if pool == 'maxpool':
            self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        elif pool == 'avgpool':
            self.adaptive_pool = nn.AdaptiveMaxPool1d(1)
        
        self.flatten = nn.Flatten(1)

    def forward(self, x):
        logger.debug(f'x shape {x.shape}')
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)
        sa_output = self.conv6(sa_conv)

        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)
        sc_output = self.conv7(sc_conv)

        feat_sum = sa_conv+sc_conv
        
        sasc_output = self.conv8(feat_sum)
        if self.pool:
            sasc_output = self.adaptive_pool(sasc_output)
        sasc_output = self.flatten(sasc_output)

        output = [sasc_output]
        output.append(sa_output)
        output.append(sc_output)
        return tuple(output)

class CNN_DAttn(nn.Module):
    def __init__(self, 
        nclass, 
        nc, 
        batch_size,
        attn_pool, 
        num_layers,
        patch_size=(256,256),
        slide_window=(256*256,256*256/2)
    ):
        super().__init__()
        self.nclass = nclass
        self.nc = nc
        self.batch_size = batch_size
        self.attn_pool = attn_pool
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.slide_window = slide_window
        # input C_in, W, H, output C_out, W_o, H_o
        num_block = patch_size[0]//4
        self.cnn_repre = CNN2D(inplane=nc,patch_size=patch_size)
        # input 
        self.dual_attn = DualAttn(self.cnn_repre.out_dim, out_channels=nclass, norm_layer=nn.BatchNorm1d, pool=self.attn_pool)
        
        #self.classifier = self.make_classifier(self.attn.out_dim,nclass)

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
        """
        pad pdf file to fit the grid shape
        """
        batch, c, seq_len = seq.size()
        h,w = self.patch_size
        need = h*w - seq_len % (h*w)
        logger.debug('need {}'.format(need))
        seq = F.pad(seq,(0,need))
        return seq
    
    def _view_to_grid(self, seq: torch.Tensor):
        """
        change the pdf file to grid view
        """
        seq = self._pad_to_grid(seq)
        batch, c, seq_len = seq.size()
        h,w = self.patch_size
        # ng number of patches in the grid
        ng = seq_len // (h*w)
        return seq.view(batch*ng, c, h,w), ng

    def _view_slide_window(self, seq: torch.Tensor):
        """
        change the pdf file to sliding window view
        """
        seq = self._pad_to_grid(seq)
        batch, c, seq_len = seq.size()
        h,w = self.patch_size
        window, stride = self.slide_window
        logger.debug(f'_view_slide_window {seq.shape}' )
        x = seq.unfold(dimension=2,size=int(window),step=int(stride))
        batch, c, nw, ws = x.size()
        logger.debug(f'_view_slide_window x {x.shape}' )
        x = x.reshape(batch*nw,c,h,w)
        return x, nw
    
    def forward(self, x):
        batch = len(x)
        r_outs = []
        # for each pdf file, we divide it into non-overlap patches
        for i in x:
            seq_len = i.size()[0]
            logger.debug('{}'.format(seq_len))
            xi = i.view(1,self.nc,seq_len)
            if self.slide_window is None:
                c_in, ng = self._view_to_grid(xi)
            else:
                c_in, ng = self._view_slide_window(xi)
            
            c_out = self.cnn_repre(c_in)
        
            r_in = c_out.view(1,ng,-1)
            logger.debug('in {} cout {} rin {}'.format(c_in.size(), c_out.size(), r_in.size()))
            
            r_out = self.dual_attn(r_in.permute(0,2,1))[0]
            logger.debug('r_out {}'.format(r_out.size()))
            r_outs.append(r_out)
        
        r_outs = torch.stack(r_outs).view(batch,-1)
        logger.debug('r_outs {}'.format(r_outs.size()))
        #output = self.classifier(r_outs)
        
        return r_outs

if __name__ == '__main__':
    logger = logging.getLogger()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(message)s')
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.setLevel(logging.DEBUG)
    i = torch.ones((10,2048000))
    pcnn = CNN_DAttn(2, 1,batch_size=32, attn_pool='maxpool', num_layers=3)
    #print(pcnn)
    print(pcnn(i).shape)
