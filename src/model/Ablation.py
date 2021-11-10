import sys
sys.path.append('.')
from einops import rearrange

from src.module.AblationAttention import AblationAttention
from src.module.CrissCrossAttention import CrissCrossAttention
from src.module.CrossAttention import CrossAttention
from src.module.CBAM import CBAM

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F


model_name = 'MIMSFCN_Ablation'
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
        return x


class FCNAttentionHead(nn.Module):
    def __init__(self, nc, byte, embedding, globalpool, atten_layer, atten_param):
        super().__init__()
        self.embedding = embedding
        if embedding:
            vocab_size = int(256 ** byte)
            logger.debug(f'vocab size {vocab_size}')
            embedding_dim = 8 * byte
            self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)
            nc = embedding_dim
        
        self.fcn = FCN(nc,1)
        self.atten = atten_layer(self.fcn.out_dim, *atten_param)
        if globalpool == 'maxpool':
            self.global_pool = nn.AdaptiveMaxPool2d(1)
        elif globalpool == 'avgpool':
            self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)
        self.out_dim = self.fcn.out_dim
        
    def forward(self, x):
        if self.embedding:
            x = self.embedding_layer(x)
            logger.debug(f'embedding  {x.shape}')
            x = rearrange(x, 'b c v t d -> b (c d) v t')
        x = self.fcn(x)
        x = self.atten(x)
        x = self.global_pool(x)
        x = self.flatten(x)
        return x


# ablation
def get_ablation(scale, attention, atten_param, embedding, fusion, nclass=2, inplane=1, globalpool='maxpool', 
                num_layers=1, dropout=0.5, patch_size=(256,256), slide_window=(256*256,256*256/2)):
    
    if attention == 'SEVT':
        atten_layer = AblationAttention
    elif attention == 'CCA':
        atten_layer = CrissCrossAttention
    elif attention == 'CA':
        atten_layer = CrossAttention
    elif attention == 'CBAM':
        atten_layer = CBAM
    logger.debug(f'attention {attention}')
    # ablation NO,A,VT,SE
    # structure P2
    # scale: MIMS SI
    # embedding: False True
    return AblationModel(nclass, nc=inplane, globalpool=globalpool, num_layers=num_layers,dropout=dropout, scale=scale, 
            fusion=fusion, atten_layer=atten_layer, atten_param=atten_param, embedding=embedding, patch_size=patch_size, slide_window=slide_window)

class AblationModel(nn.Module):
    def __init__(self, 
        nclass, 
        nc, 
        globalpool, 
        num_layers,
        dropout,
        scale,
        fusion,
        atten_layer=None,
        atten_param={},
        embedding=False,
        patch_size=(256,256),
        slide_window=(256*256,256*256/2)
    ):
        super().__init__()
        self.nclass = nclass
        self.nc = nc
        self.globalpool = globalpool
        self.num_layers = num_layers
        self.dropout = dropout
        self.scale = scale
        self.fusion = fusion
        self.atten_layer = atten_layer
        self.atten_param = atten_param
        self.embedding = embedding
        self.patch_size = patch_size
        self.slide_window = slide_window

        if self.scale == 'MIMS':
            self.head0 = FCNAttentionHead(nc, 1, embedding, globalpool, atten_layer, atten_param)
            self.head1 = FCNAttentionHead(nc, 2, embedding, globalpool, atten_layer, atten_param)
            if self.fusion == 'concat':
                model_dim = self.head0.out_dim + self.head1.out_dim
            elif self.fusion == 'sum':
                model_dim = self.head0.out_dim

        elif self.scale == 'SI':
            self.head0 = FCNAttentionHead(nc, 1, embedding, globalpool, atten_layer, atten_param)
            model_dim = self.head0.out_dim
  
        self.classifier = self.make_classifier(model_dim, nclass)

    def make_classifier(self, hidden_size, nclass, layer_num=1, dropout=0.5):
        layers = []
        sz = hidden_size
        
        for l in range(layer_num-1):
            layers += [nn.Linear(sz, sz//2)]
            layers += [nn.ReLU(True)]
            layers += [nn.Dropout(dropout)]
            sz //= 2
        
        layers += [nn.Dropout(dropout)]
        layers += [nn.Linear(sz, nclass)]
        return nn.Sequential(*layers)

    def _pad_to_grid(self, seq: torch.Tensor):
        """
        pad pdf file to fit the grid shape
        """
        batch, c, seq_len = seq.size()
        h,w = self.patch_size
        need = h*w - seq_len % (h*w)
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
        x = seq.unfold(dimension=2,size=int(window),step=int(stride))
        batch, c, nw, ws = x.size()
        x = x.reshape(batch*nw,c,h,w)
        return x, nw

    def forward(self, x):
        batch = len(x)
    
        outs = []
        # for each pdf file, we divide it into non-overlap patches
        for i in range(batch):
            x0_i = x[i][0]
            x0_i = rearrange(x0_i, '(c t) -> 1 c t', c = self.nc)
            c_in_0, _ = self._view_slide_window(x0_i)
            # number of windows, channels, height, width
            c_in_0 = rearrange(c_in_0, 't c h w -> 1 c (h w) t')

            if self.scale == 'MIMS':
                x1_i = x[i][1]
                x1_i = rearrange(x1_i, '(c t) -> 1 c t', c = self.nc)
                c_in_1, _ = self._view_slide_window(x1_i)
                c_in_1 = rearrange(c_in_1, 't c h w -> 1 c (h w) t')

            #memory_usage(c_in.size())
            out0 = self.head0(c_in_0)

            if self.scale == 'MIMS':
                out1 = self.head1(c_in_1)
                if self.fusion == 'sum':
                    out = out0 + out1
                elif self.fusion == 'concat':
                    out = torch.cat((out0,out1),1)
            elif self.scale == 'SI':
                out = out0

            outs.append(out)
        
        outs = torch.stack(outs).view(batch, -1)
        output = self.classifier(outs)
        
        return output

if __name__ == '__main__':
    logger = logging.getLogger()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(message)s')
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.setLevel(logging.DEBUG)
    i = [[torch.ones((4248001)),torch.ones((2124000))]]
    #i_embed = [[torch.randint(0,255,(1000,)),torch.randint(0,65535, (3000,))]]
    from torch.profiler import profile, ProfilerActivity, record_function
    #logger.debug(f'i_embed {i_embed}')
    patch = 64
    atten_param = {}
    attention = 'CBAM'
    pcnn = get_ablation('MIMS',  attention, atten_param, False,'concat',2,1,'maxpool',num_layers=1,patch_size=(patch,patch),slide_window=(patch*patch,patch*patch/2))

    with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True,with_flops=True) as prof:
        with record_function("model_inference"):
            print(pcnn(i).shape)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))