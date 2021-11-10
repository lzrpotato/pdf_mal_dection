import sys
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

sys.path.append('.')
import os

print(os.getcwd())
import logging

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

logger = logging.getLogger('model.CNN_TRAN')

__all__ = ['CNN_TRAN']

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


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class CNN_TRAN(nn.Module):
    def __init__(self, 
        nclass, 
        nc, 
        batch_size,
        pool, 
        num_layers,
        patch_size=(256,256),
        slide_window=(256*256,256*256/2)
    ):
        super().__init__()
        self.nclass = nclass
        self.nc = nc
        self.batch_size = batch_size
        self.pool = pool
        self.num_layers = num_layers
        self.patch_size = patch_size
        
        self.slide_window = slide_window
        dim = 128
        
        # input C_in, W, H, output C_out, W_o, H_o
        num_block = patch_size[0]//4
        self.cnn_repre = CNN2D(inplane=nc,patch_size=patch_size)
        
        patch_dim = self.cnn_repre.out_dim * ((patch_size[0] // 32) ** 2)
        self.to_slide_window_embedding = nn.Sequential(
            Rearrange('b c w h -> 1 (b) (w h c)'),
            nn.Linear(patch_dim, dim),
        )
        # input 
        self.tran = Transformer(dim=dim, depth=2, heads=8, dim_head=16,mlp_dim=16, dropout=0.1)
        
        #self.classifier = self.make_classifier(self.tran.out_dim,nclass)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, nclass)
        )


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
        outs = []
        # for each pdf file, we divide it into non-overlap patches
        for i in x:
            seq_len = i.size()[0]
            logger.debug('{}'.format(seq_len))
            xi = i.view(1,self.nc,seq_len)
            if self.slide_window is None:
                c_in, ng = self._view_to_grid(xi)
            else:
                c_in, ng = self._view_slide_window(xi)
            
            logger.debug(f'c_in {c_in.shape}')
            c_out = self.cnn_repre(c_in)
            logger.debug(f'c_out {c_out.shape}')
            # c_out (L, H*W)
            # (B,C,H,W) -> (1,C,H*W)
            r_in = self.to_slide_window_embedding(c_out)
            logger.debug('in {} cout {} rin {}'.format(c_in.size(), c_out.size(), r_in.size()))
            
            tran_out = self.tran(r_in)
            logger.debug('tran_out {}'.format(tran_out.size()))
            pool_out = tran_out.mean(dim=1) if self.pool == 'mean' else tran_out[:, 0]
            logger.debug('pool_out {}'.format(pool_out.size()))
            outs.append(pool_out)
        
        
        outs = torch.stack(outs)
        #outs = torch.flatten(outs,1,2)
        logger.debug('outs stack {}'.format(outs.size()))
        outs = outs[:, 0]
        logger.debug('outs {}'.format(outs.size()))
        output = self.mlp_head(outs)
        
        return output

if __name__ == '__main__':
    logger = logging.getLogger()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(message)s')
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.setLevel(logging.DEBUG)
    i = torch.ones((1,2048000))

    from torch.profiler import profile, ProfilerActivity, record_function

    patch = 256
    pcnn = CNN_TRAN(2, 1,batch_size=32, patch_size=(patch, patch), slide_window=(patch*patch,patch*patch/2), pool='mean', num_layers=3)

    with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True) as prof:
        with record_function("model_inference"):
            print(pcnn(i).shape)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))