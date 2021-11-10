import sys
from tracemalloc import Snapshot

from torch.nn.modules.sparse import Embedding

sys.path.append('.')
import os

print(os.getcwd())
import logging

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

logger = logging.getLogger('model.MSFCN1_CCCA1')

__all__ = ['MSFCN1_CCCA1']

def memory_usage(tensor_size):
    max_memory = torch.cuda.max_memory_allocated()
    if max_memory == 0:
        usage = 0
    else:
        usage = torch.cuda.memory_allocated() / max_memory
    logger.debug(f'[memory usage] {usage} {tensor_size}')

 
class FCN(nn.Module):
    def __init__(self, in_channels, scale):
        super().__init__()
        chan_list = [64,128,256,128]
        #chan_list = [64,128,64]
        if scale > 1:
            self.scalepool = nn.AvgPool2d(scale,scale,padding=scale//2)
        else:
            self.scalepool = None
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=chan_list[0],kernel_size=12,stride=6,padding=6,bias=True),
            nn.ReLU(),

            nn.Conv2d(in_channels=chan_list[0],out_channels=chan_list[1],kernel_size=8,stride=4,padding=4,bias=True),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=chan_list[1],out_channels=chan_list[2],kernel_size=5,stride=2,padding=2,bias=True),
            nn.ReLU(),

            nn.Conv2d(in_channels=chan_list[2],out_channels=chan_list[3],kernel_size=3,stride=1,padding=1,bias=True),
            nn.ReLU(),

        )
        #self.adptive_pool = nn.AdaptiveAvgPool2d((1,1))
        self.out_dim = chan_list[3]

    def forward(self, x):
        logger.debug(f'x shape in fcn {x.shape}')
        if self.scalepool:
            x = self.scalepool(x)
        x = self.model(x)
        logger.debug(f'model x shape in fcn {x.shape}')
        # x shape batch, channel, variable, time
        #x = self.adptive_pool(x)
        #logger.debug(f'adptive x shape in fcn {x.shape}')
        return x


def conv1x1(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,padding=dilation,
                groups=groups, bias=False, dilation=dilation)

def INF(B,D, device):
     return -torch.diag(torch.tensor(float("inf")).to(device).repeat(D),0).unsqueeze(0).repeat(B,1,1)

class ChannelCrissCrossAttention(nn.Module):
    """ Channel Criss-Cross Attention Module"""
    def __init__(self, in_dim):
        super(ChannelCrissCrossAttention,self).__init__()
        down_scale = 8
        kernel_size = 3
        padding = 1
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//down_scale, kernel_size=kernel_size,padding=padding)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//down_scale, kernel_size=kernel_size,padding=padding)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//down_scale, kernel_size=kernel_size,padding=padding)

        self.c_reconv = nn.Conv2d(in_channels=in_dim//down_scale, out_channels=in_dim, kernel_size=kernel_size,padding=padding)
        self.v_reconv = nn.Conv2d(in_channels=in_dim//down_scale, out_channels=in_dim, kernel_size=kernel_size,padding=padding)
        self.t_reconv = nn.Conv2d(in_channels=in_dim//down_scale, out_channels=in_dim, kernel_size=kernel_size,padding=padding)
        self.INF = INF
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # (B,C,V,T)
        proj_query = self.query_conv(x) # (B,C,V,T)

        B, C, V, T = proj_query.size()
        # (B,C,V,T) -> (B,V,T,C,1)
        proj_query_C = proj_query.view(B*T*V,C,1)
        # (B,C,V,T) -> (B,C,T,V,1)
        proj_query_V = proj_query.view(B*C*T,V,1)
        # (B,C,V,T) -> (B,C,V,T,1)
        proj_query_T = proj_query.view(B*C*V,T,1)
        
        
        # (B,C,V,T)->(B,C1,V,T)
        proj_key = self.key_conv(x)  
        # (B,C,V,T) -> (B,V,T,1,C)
        proj_key_C = proj_key.view(B*T*V,1,C)
        # (B,C,V,T) -> (B,C,T,1,V)
        proj_key_V = proj_key.view(B*C*T,1,V)
        # (B,C,V,T) -> (B,C,V,1,T)
        proj_key_T = proj_key.view(B*C*V,1,T)

        # (B,C,V,T)->(B,C,V,T)
        proj_value = self.value_conv(x)   
        # (B,C,V,T) -> (B,V,T,C,1)
        proj_value_C1 = proj_value.view(B,T,V,C,1).view(B*T*V,C,1)
        # (B,C,V,T) -> (B,C,T,V,1)
        proj_value_V1 = proj_value.view(B,C,T,V,1).view(B*C*T,V,1)
        # (B,C,V,T) -> (B,C,V,T,1)
        proj_value_T1 = proj_value.view(B,C,V,T,1).view(B*C*V,T,1)
        
        # (B*T*V,C,1) x (B*T*V,1,C) -> (B*T*V,C,C) -> (B,C,V,T,C)
        energy_CC = torch.bmm(proj_query_C, proj_key_C).view(B,T,V,C,C).permute(0,3,2,1,4)
        # (B*C*T,V,1) x (B*C*T,1,V) -> (B*C*T,V,V) -> (B,C,V,T,V)
        energy_VV = torch.bmm(proj_query_V, proj_key_V).view(B,C,T,V,V).permute(0,1,3,2,4)
        # (B*C*V,T,1) x (B*C*V,1,T) -> (B*C*V,T,T) -> (B,C,V,T,T)
        energy_TT = torch.bmm(proj_query_T, proj_key_T).view(B,C,V,T,T)
        
        # concate (B,C,V,T,C+V+T)
        concate = self.softmax(torch.cat([energy_CC,energy_VV,energy_TT],-1))
        logger.debug(f'concate {concate.shape}')

        # (B,C,V,T,C) -> (B,V,T,C,C) -> (B*V*T,C,C)
        att_C = concate[:,:,:,:,0:C].permute(0,2,3,1,4).contiguous().view(B*V*T,C,C)
        # (B,C,V,T,V) -> (B,C,T,V,V) -> (B*C*T,V,V)
        logger.debug(f'attV {concate[:,:,:,:,C:C+V].permute(0,1,3,2,4).shape} {B},{C},{T},{V}')
        att_V = concate[:,:,:,:,C:C+V].permute(0,1,3,2,4).contiguous().view(B*C*T,V,V)
        # (B,C,V,T,T) -> (B*C*V,T,T)
        att_T = concate[:,:,:,:,C+V:C+V+T].contiguous().view(B*C*V,T,T)

        # (B*V*T,C,C) * (B*T*V,C,1) -> (B*V*T,C,1) -> (B,C,V,T)
        out_C = torch.bmm(att_C, proj_value_C1).view(B,V,T,C).permute(0,3,1,2)
        # (B*C*T,V,V) * (B*C*T*V,1) -> (B*C*T,V,1) -> (B,C,V,T)
        out_V = torch.bmm(att_V, proj_value_V1).view(B,C,T,V).permute(0,1,3,2)
        # (B*C*V,T,T) * (B*C*V,T,1) -> (B*C*V,T,1) -> (B,C,V,T)
        out_T = torch.bmm(att_T, proj_value_T1).view(B,C,V,T)
        
        out_C = self.c_reconv(out_C)
        out_V = self.v_reconv(out_V)
        out_T = self.t_reconv(out_T)

        logger.debug(f'out shape {out_C.shape} {out_V.shape} {out_T.shape}')
        return self.gamma * (out_C+out_V+out_T) + x
        

def get_msfcn1ccca1_twoheadfcn(nclass=2, inplane=1, globalpool='maxpool', num_layers=1, patch_size=(256,256), slide_window=(256*256,256*256/2)):
    return MSFCN1_CCCA1(nclass, nc=inplane, globalpool=globalpool, num_layers=num_layers,
            multiscale='twoheadfcn', patch_size=patch_size, slide_window=slide_window)

def get_msfcn1ccca1_twoheadfcnccca(nclass=2, inplane=1, globalpool='maxpool', num_layers=1, patch_size=(256,256), slide_window=(256*256,256*256/2)):
    return MSFCN1_CCCA1(nclass, nc=inplane, globalpool=globalpool, num_layers=num_layers,
            multiscale='twoheadfcnca', patch_size=patch_size, slide_window=slide_window)

def get_msfcn1ccca1_sharedfcnccca(nclass=2, inplane=1, globalpool='maxpool', num_layers=1, patch_size=(256,256), slide_window=(256*256,256*256/2)):
    return MSFCN1_CCCA1(nclass, nc=inplane, globalpool=globalpool, num_layers=num_layers,
            multiscale='sharedfcnca', patch_size=patch_size, slide_window=slide_window)

def get_msfcn1ccca1_oneheadfcn(nclass=2, inplane=1, globalpool='maxpool', num_layers=1, patch_size=(256,256), slide_window=(256*256,256*256/2)):
    return MSFCN1_CCCA1(nclass, nc=inplane, globalpool=globalpool, num_layers=num_layers,
            multiscale='oneheadfcn', patch_size=patch_size, slide_window=slide_window)

class MSFCN1_CCCA1(nn.Module):
    def __init__(self, 
        nclass, 
        nc, 
        globalpool, 
        num_layers,
        multiscale,
        patch_size=(256,256),
        slide_window=(256*256,256*256/2)
    ):
        super().__init__()
        self.nclass = nclass
        self.nc = nc
        self.globalpool = globalpool
        self.num_layers = num_layers
        self.multiscale = multiscale
        self.patch_size = patch_size
        self.slide_window = slide_window
        # input C_in, W, H, output C_out, W_o, H_o
        num_block = patch_size[0]//4

        if self.multiscale == 'twoheadfcn' or self.multiscale == 'twoheadfcnca':
            self.cnn_repre0 = FCN(nc,1)
            self.cnn_repre1 = FCN(nc,2)
        else:
            self.cnn_repre0 = FCN(nc, 1)

        model_dim = self.cnn_repre0.out_dim
        self.dowm = nn.Sequential(
                #conv1x1(self.cnn_repre0.out_dim, down_dim),
                nn.Conv2d(self.cnn_repre0.out_dim, model_dim, kernel_size=3, stride=3, padding=1),
                nn.ReLU()
            )

        if self.multiscale == 'sharedfcnca':
            self.down_sample_input = nn.Sequential(
                nn.AvgPool2d(2,2,padding=2//2)
            )

        self.cca_attn0 = ChannelCrissCrossAttention(model_dim)
        # input 
        if self.multiscale == 'twoheadfcnca':
            self.cca_attn1 = ChannelCrissCrossAttention(model_dim)
            
        if globalpool == 'maxpool':
            self.global_pool = nn.AdaptiveMaxPool2d(1)
        elif globalpool == 'avgpool':
            self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.flatten = nn.Flatten(1)
        self.classifier = self.make_classifier(model_dim + model_dim,nclass)

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

    def _pad_resolution(self, seq, byte):
        logger.debug(f'seq {seq.shape} {seq.size(2) % byte} {byte-(seq.size(2)%byte) }')
        if seq.size(2) % byte != 0:
            seq = F.pad(seq,(2,byte-(seq.size(2)%byte)))
        
        return seq

    def _change_resolution(self, seq, byte):
        """
        change the resolution of the input
        """
        batch, nc, seq_len = seq.size()
        seq = self._pad_resolution(seq, byte)
        x = torch.div(torch.sum(seq.reshape(1,-1,byte),dim=2),byte)
        x = x.view(batch, nc, -1)
        return x
    
    def forward(self, x):
        batch = len(x)
        
        outs = []
        # for each pdf file, we divide it into non-overlap patches
        for i in x:
            seq_len = i.size()[0]
            logger.debug('i shape {}'.format(i.size()))
            xi = i.view(1,self.nc,seq_len)
            
            x_resolu_0 = xi.clone()
            #x_resolu_1 = self._change_resolution(xi, 2)
            logger.debug(f'x_resolu_0 {x_resolu_0.shape}')
            #logger.debug(f'x_resolu_1 {x_resolu_1.shape}')
            if self.slide_window is None:
                c_in_0, ng_0 = self._view_to_grid(x_resolu_0)
                #c_in_1, ng_1 = self._view_to_grid(x_resolu_1)
            else:
                c_in_0, ng_0 = self._view_slide_window(x_resolu_0)
                #c_in_1, ng_1 = self._view_slide_window(x_resolu_1)
            
            #memory_usage(c_in.size())
            logger.debug(f'c_in_0 size {c_in_0.size()}')
            #logger.debug(f'c_in_1 size {c_in_1.size()}')

            # number of windows, channels, height, width
            nw, c, h, w = c_in_0.size()
            c_in_0 = c_in_0.reshape(1,c,h*w, nw)

            # nw, c, h, w = c_in_1.size()
            # c_in_1 = c_in_1.reshape(1,c,h*w, nw)

            logger.debug(f'c_in_0 size {c_in_0.size()}')
            #logger.debug(f'c_in_1 size {c_in_1.size()}')
            
            c_out0 = self.cnn_repre0(c_in_0)
            #c_out0 = self.dowm(c_out0)

            if self.multiscale == 'twoheadfcn' or self.multiscale == 'twoheadfcnca':
                c_out1 = self.cnn_repre1(c_in_0)
                #c_out1 = self.dowm(c_out1)
            elif self.multiscale == 'sharedfcnca':
                c_in_1 = self.down_sample_input(c_in_0)
                c_out1 = self.cnn_repre0(c_in_1)
            logger.debug(f'c_out.size {c_out0.size()}')
            
            
            
            
            attn_out0 = self.cca_attn0(c_out0)
            
            if self.multiscale == 'twoheadfcn' or self.multiscale == 'sharedfcnca':
                attn_out1 = self.cca_attn0(c_out1)
            elif self.multiscale == 'twoheadfcnca':
                attn_out1 = self.cca_attn1(c_out1)
            

            logger.debug(f'attn_out0 {attn_out0.shape}')
            logger.debug(f'attn_out1 {attn_out1.shape}')

            out0 = self.global_pool(attn_out0)
            out1 = self.global_pool(attn_out1)
            out0 = self.flatten(out0)
            out1 = self.flatten(out1)

            logger.debug(f'globalpool {out0.shape}')
            logger.debug(f'globalpool {out1.shape}')

            out = torch.cat((out0,out1),1)
            logger.debug(f'out {out.shape}')
            
            outs.append(out)
        
        outs = torch.stack(outs).view(batch, -1)
        logger.debug('outs {}'.format(outs.size()))
        output = self.classifier(outs)
        
        return output

if __name__ == '__main__':
    logger = logging.getLogger()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(message)s')
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.setLevel(logging.DEBUG)
    i = torch.ones((1,4248001))
    from torch.profiler import profile, ProfilerActivity, record_function

    patch = 64
    #pcnn = get_msfcn1ccca_twoheadfcn(2,1,'maxpool',num_layers=1,patch_size=(patch,patch),slide_window=(patch*patch,patch*patch/2))
    pcnn = get_msfcn1ccca1_twoheadfcnccca(2,1,'maxpool',num_layers=1,patch_size=(patch,patch),slide_window=(patch*patch,patch*patch/2))
    #pcnn = get_msfcn1ccca_sharedfcnccca(2,1,'maxpool',num_layers=1,patch_size=(patch,patch),slide_window=(patch*patch,patch*patch/2))

    with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True,with_flops=True) as prof:
        with record_function("model_inference"):
            print(pcnn(i).shape)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))