import sys
from tracemalloc import Snapshot
from einops import rearrange

from torch.nn.modules.sparse import Embedding

sys.path.append('.')
import os

print(os.getcwd())
import logging

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

model_name = 'MSFCN1_CCCA2'
logger = logging.getLogger(f'model.{model_name}')

__all__ = [model_name]

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
        kernel_size = 1
        padding = 0
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//down_scale, kernel_size=kernel_size,padding=padding)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//down_scale, kernel_size=kernel_size,padding=padding)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//down_scale, kernel_size=kernel_size)

        self.out_reconv = nn.Conv2d(in_channels=in_dim//down_scale, out_channels=in_dim, kernel_size=kernel_size,padding=padding)

        self.INF = INF
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # (B,C,V,T)
        proj_query = self.query_conv(x) # (B,C,V,T)

        B, C, V, T = proj_query.size()
        proj_query_ct = rearrange(proj_query, 'b c v t -> (b v) c t')
        proj_query_cv = rearrange(proj_query, 'b c v t -> (b t) c v')
        proj_query_vc = rearrange(proj_query, 'b c v t -> (b t) v c')
        proj_query_vt = rearrange(proj_query, 'b c v t -> (b c) v t')
        proj_query_tc = rearrange(proj_query, 'b c v t -> (b v) t c')
        proj_query_tv = rearrange(proj_query, 'b c v t -> (b c) t v')
        
        proj_key = self.key_conv(x)  
        proj_key_tc = rearrange(proj_key, 'b c v t -> (b v) t c')
        proj_key_vc = rearrange(proj_key, 'b c v t -> (b t) v c')
        proj_key_cv = rearrange(proj_key, 'b c v t -> (b t) c v')
        proj_key_tv = rearrange(proj_key, 'b c v t -> (b c) t v')
        proj_key_ct = rearrange(proj_key, 'b c v t -> (b v) c t')
        proj_key_vt = rearrange(proj_key, 'b c v t -> (b c) v t')

        # energy
        # (b*v,c,c)
        energy_vcc = rearrange(torch.bmm(proj_query_ct, proj_key_tc)+self.INF(B*V, C, x.device), '(b v) c1 c2 -> b c1 v c2', b=B)
        energy_cvv = rearrange(torch.bmm(proj_query_vt, proj_key_tv), '(b c) v1 v2 -> b c v1 v2', b=B)
        energy_tcc = rearrange(torch.bmm(proj_query_cv, proj_key_vc)+self.INF(B*T, C, x.device), '(b t) c1 c2 -> b c1 t c2', b=B)
        energy_ctt = rearrange(torch.bmm(proj_query_tv, proj_key_vt), '(b c) t1 t2 -> b c t1 t2', b=B)
        energy_tvv = rearrange(torch.bmm(proj_query_vc, proj_key_cv)+self.INF(B*T, V, x.device), '(b t) v1 v2 -> b v1 t v2', b=B)
        energy_vtt = rearrange(torch.bmm(proj_query_tc, proj_key_ct), '(b v) t1 t2 -> b v t1 t2', b=B)
        

        # attention
        # (b c v (c v))
        concate_cv = self.softmax(torch.cat([energy_vcc,energy_cvv], -1))
        # (b c t (c t))
        concate_ct = self.softmax(torch.cat([energy_tcc,energy_ctt], -1))
        # (b v t (v t))
        concate_vt = self.softmax(torch.cat([energy_tvv,energy_vtt], -1))
        
        attn_vcc, attn_cvv = rearrange(concate_cv[:,:,:,0:C],'b c1 v c2 -> (b v) c1 c2'), rearrange(concate_cv[:,:,:,C:(C+V)],'b c v1 v2 -> (b c) v1 v2')
        attn_tcc, attn_ctt = rearrange(concate_ct[:,:,:,0:C],'b c1 t c2 -> (b t) c1 c2'), rearrange(concate_ct[:,:,:,C:(C+T)],'b c t1 t2 -> (b c) t1 t2')
        attn_tvv, attn_vtt = rearrange(concate_vt[:,:,:,0:V],'b v1 t v2 -> (b t) v1 v2'), rearrange(concate_vt[:,:,:,V:(V+T)],'b v t1 t2 -> (b v) t1 t2')

        # (B,C,V,T)->(B,C1,V,T)
        proj_value = self.value_conv(x)   
        proj_value_vt = rearrange(proj_value, 'b c v t -> (b c) v t')
        proj_value_tv = rearrange(proj_value, 'b c v t -> (b c) t v')
        proj_value_ct = rearrange(proj_value, 'b c v t -> (b v) c t')
        proj_value_tc = rearrange(proj_value, 'b c v t -> (b v) t c')
        proj_value_cv = rearrange(proj_value, 'b c v t -> (b t) c v')
        proj_value_vc = rearrange(proj_value, 'b c v t -> (b t) v c')

        # out
        out_vt = rearrange(torch.bmm(proj_value_vt, attn_ctt),'(b c) v t -> b c v t', b=B)
        out_tv = rearrange(torch.bmm(proj_value_tv, attn_cvv),'(b c) t v -> b c v t', b=B)
        out_ct = rearrange(torch.bmm(proj_value_ct, attn_vtt),'(b v) c t -> b c v t', b=B)
        out_tc = rearrange(torch.bmm(proj_value_tc, attn_vcc),'(b v) t c -> b c v t', b=B)
        out_cv = rearrange(torch.bmm(proj_value_cv, attn_tvv),'(b t) c v -> b c v t', b=B)
        out_vc = rearrange(torch.bmm(proj_value_vc, attn_tcc),'(b t) v c -> b c v t', b=B)


        logger.debug(f'{out_vt.shape} {out_tv.shape} {out_ct.shape} {out_tc.shape} {out_cv.shape} {out_vc.shape}')
        return self.gamma * (self.out_reconv(out_vt) + self.out_reconv(out_tv) + self.out_reconv(out_ct) + self.out_reconv(out_tc) + self.out_reconv(out_cv) + self.out_reconv(out_vc)) + x
        

def get_msfcn1ccca2_twoheadfcn(nclass=2, inplane=1, globalpool='maxpool', num_layers=1, patch_size=(256,256), slide_window=(256*256,256*256/2)):
    return MSFCN1_CCCA2(nclass, nc=inplane, globalpool=globalpool, num_layers=num_layers,
            multiscale='twoheadfcn', patch_size=patch_size, slide_window=slide_window)

def get_msfcn1ccca2_twoheadfcnccca(nclass=2, inplane=1, globalpool='maxpool', num_layers=1, patch_size=(256,256), slide_window=(256*256,256*256/2)):
    return MSFCN1_CCCA2(nclass, nc=inplane, globalpool=globalpool, num_layers=num_layers,
            multiscale='twoheadfcnca', patch_size=patch_size, slide_window=slide_window)

def get_msfcn1ccca2_sharedfcnccca(nclass=2, inplane=1, globalpool='maxpool', num_layers=1, patch_size=(256,256), slide_window=(256*256,256*256/2)):
    return MSFCN1_CCCA2(nclass, nc=inplane, globalpool=globalpool, num_layers=num_layers,
            multiscale='sharedfcnca', patch_size=patch_size, slide_window=slide_window)

def get_msfcn1ccca2_oneheadfcn(nclass=2, inplane=1, globalpool='maxpool', num_layers=1, patch_size=(256,256), slide_window=(256*256,256*256/2)):
    return MSFCN1_CCCA2(nclass, nc=inplane, globalpool=globalpool, num_layers=num_layers,
            multiscale='oneheadfcn', patch_size=patch_size, slide_window=slide_window)

class MSFCN1_CCCA2(nn.Module):
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
    #pcnn = get_msfcn1ccca2_twoheadfcn(2,1,'maxpool',num_layers=1,patch_size=(patch,patch),slide_window=(patch*patch,patch*patch/2))
    pcnn = get_msfcn1ccca2_twoheadfcnccca(2,1,'maxpool',num_layers=1,patch_size=(patch,patch),slide_window=(patch*patch,patch*patch/2))
    #pcnn = get_msfcn1ccca2_sharedfcnccca(2,1,'maxpool',num_layers=1,patch_size=(patch,patch),slide_window=(patch*patch,patch*patch/2))

    with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True,with_flops=True) as prof:
        with record_function("model_inference"):
            print(pcnn(i).shape)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))