import sys
from einops import rearrange
from src.module.SEAttention import SEAttention

sys.path.append('.')

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F


model_name = 'MIMSFCN_SEA'
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

def get_mimsfcnsea_twoheadfcn(nclass=2, inplane=1, globalpool='maxpool', num_layers=1, patch_size=(256,256), slide_window=(256*256,256*256/2)):
    return MIMSFCN_SEA(nclass, nc=inplane, globalpool=globalpool, num_layers=num_layers,
            multiscale='twoheadfcn', patch_size=patch_size, slide_window=slide_window)

def get_mimsfcnsea_twoheadfcnsea_para1(nclass=2, inplane=1, globalpool='maxpool', num_layers=1, patch_size=(256,256), slide_window=(256*256,256*256/2)):
    return MIMSFCN_SEA(nclass, nc=inplane, globalpool=globalpool, num_layers=num_layers,
            multiscale='twoheadfcnca', structure='parallel1', patch_size=patch_size, slide_window=slide_window)

def get_mimsfcnsea_twoheadfcnsea_para2(nclass=2, inplane=1, globalpool='maxpool', num_layers=1, patch_size=(256,256), slide_window=(256*256,256*256/2)):
    return MIMSFCN_SEA(nclass, nc=inplane, globalpool=globalpool, num_layers=num_layers,
            multiscale='twoheadfcnca',  structure='parallel2', patch_size=patch_size, slide_window=slide_window)

def get_mimsfcnsea_twoheadfcnsea(nclass=2, inplane=1, globalpool='maxpool', num_layers=1, patch_size=(256,256), slide_window=(256*256,256*256/2)):
    return MIMSFCN_SEA(nclass, nc=inplane, globalpool=globalpool, num_layers=num_layers,
            multiscale='twoheadfcnca', patch_size=patch_size, slide_window=slide_window)

def get_mimsfcnsea_sharedfcnsea(nclass=2, inplane=1, globalpool='maxpool', num_layers=1, patch_size=(256,256), slide_window=(256*256,256*256/2)):
    return MIMSFCN_SEA(nclass, nc=inplane, globalpool=globalpool, num_layers=num_layers,
            multiscale='sharedfcnca', patch_size=patch_size, slide_window=slide_window)

def get_mimsfcnsea_oneheadfcn(nclass=2, inplane=1, globalpool='maxpool', num_layers=1, patch_size=(256,256), slide_window=(256*256,256*256/2)):
    return MIMSFCN_SEA(nclass, nc=inplane, globalpool=globalpool, num_layers=num_layers,
            multiscale='oneheadfcn', patch_size=patch_size, slide_window=slide_window)

class MIMSFCN_SEA(nn.Module):
    def __init__(self, 
        nclass, 
        nc, 
        globalpool, 
        num_layers,
        multiscale,
        structure='parallel',

        patch_size=(256,256),
        slide_window=(256*256,256*256/2)
    ):
        super().__init__()
        self.nclass = nclass
        self.nc = nc
        self.globalpool = globalpool
        self.num_layers = num_layers
        self.multiscale = multiscale
        self.structure = structure
        self.patch_size = patch_size
        self.slide_window = slide_window
        # input C_in, W, H, output C_out, W_o, H_o
        num_block = patch_size[0]//4

        if self.multiscale == 'twoheadfcn' or self.multiscale == 'twoheadfcnca':
            self.cnn_repre0 = FCN(nc,1)
            self.cnn_repre1 = FCN(nc,1)
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

        self.sa_attn0 = SEAttention(model_dim,self.structure)
        # input 
        if self.multiscale == 'twoheadfcnca':
            self.sa_attn1 = SEAttention(model_dim,self.structure )
            
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
        for i in range(batch):
            x0_i = x[i][0]
            x1_i = x[i][1]

            x0_i = rearrange(x0_i, '(c t) -> 1 c t', c = self.nc)
            x1_i = rearrange(x1_i, '(c t) -> 1 c t', c = self.nc)
            logger.debug(f'x0_i {x0_i.shape} x1_i {x1_i.shape}')

            if self.slide_window is None:
                c_in_0, ng_0 = self._view_to_grid(x0_i)
                c_in_1, ng_1 = self._view_to_grid(x1_i)
            else:
                c_in_0, ng_0 = self._view_slide_window(x0_i)
                c_in_1, ng_1 = self._view_slide_window(x1_i)
            
            #memory_usage(c_in.size())

            # number of windows, channels, height, width
            c_in_0 = rearrange(c_in_0, 't c h w -> 1 c (h w) t')
            c_in_1 = rearrange(c_in_1, 't c h w -> 1 c (h w) t')
            
            c_out0 = self.cnn_repre0(c_in_0)

            if self.multiscale == 'twoheadfcn' or self.multiscale == 'twoheadfcnca':
                c_out1 = self.cnn_repre1(c_in_1)
            elif self.multiscale == 'sharedfcnca':
                c_in_1 = self.down_sample_input(c_in_1)
                c_out1 = self.cnn_repre0(c_in_1)
            logger.debug(f'c_out.size {c_out0.size()} {c_out1.size()}')
            
            attn_out0 = self.sa_attn0(c_out0)
            
            if self.multiscale == 'twoheadfcn' or self.multiscale == 'sharedfcnca':
                attn_out1 = self.sa_attn0(c_out1)
            elif self.multiscale == 'twoheadfcnca':
                attn_out1 = self.sa_attn1(c_out1)
            

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
    i = (torch.ones((1,4248001)),torch.ones((1,2124000)))
    from torch.profiler import profile, ProfilerActivity, record_function

    patch = 64
    #pcnn = get_mimsfcnsea_twoheadfcn(2,1,'maxpool',num_layers=1,patch_size=(patch,patch),slide_window=(patch*patch,patch*patch/2))
    pcnn = get_mimsfcnsea_twoheadfcnsea(2,1,'maxpool',num_layers=1,patch_size=(patch,patch),slide_window=(patch*patch,patch*patch/2))
    #pcnn = get_mimsfcnsea_sharedfcnca(2,1,'maxpool',num_layers=1,patch_size=(patch,patch),slide_window=(patch*patch,patch*patch/2))

    with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True,with_flops=True) as prof:
        with record_function("model_inference"):
            print(pcnn(i).shape)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))