import logging
import torch
import torch.nn as nn
from einops import rearrange


logger = logging.getLogger(f'module.CrossAttention')

def conv1x1(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,padding=dilation,
                groups=groups, bias=False, dilation=dilation)

class CrossAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.vattn = vAttn(in_dim)
        self.tattn = tAttn(in_dim)
    
    def forward(self, x):
        x = self.vattn(x)
        x = self.tattn(x)
        return x

class vAttn(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        out_channels = in_channels//8

        self.query_w = conv1x1(in_channels,out_channels)
        self.key_w = conv1x1(in_channels,out_channels)
        self.value_w = conv1x1(in_channels,in_channels//2)
        self.attn_value_w = conv1x1(in_channels//2,in_channels)

        self.softmax = nn.Softmax(-1)
        self.sigma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B = x.size(0)
        # x: (B,C,V,T)
        query_v = self.query_w(x) # (B,C,V,T)
        query_v = rearrange(query_v, 'b c v t -> (b t) v c')

        key_v = self.key_w(x)     # (B,C,V,T)
        key_v = rearrange(key_v, 'b c v t -> (b t) c v')
        
        value_v = self.value_w(x) # (B,C1,V,T)
        value_v = rearrange(value_v, 'b c1 v t -> (b t) v c1')

        # softmax at (B T V V)
        attn = self.softmax(torch.bmm(query_v, key_v))
        
        attn_value = rearrange(torch.bmm(attn, value_v), '(b t) v c1 -> b c1 v t', b=B)
        attn_value = self.attn_value_w(attn_value)  # (B,C1,V,T) -> (B,C,V,T)

        return x + self.sigma * attn_value
        

class tAttn(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        out_channels = in_channels//8

        self.query_w =conv1x1(in_channels,out_channels)
        self.key_w = conv1x1(in_channels,out_channels)
        self.value_w = conv1x1(in_channels,in_channels//2)
        self.attn_value_w = conv1x1(in_channels//2,in_channels)

        self.softmax = nn.Softmax(dim=-1)
        self.sigma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B = x.size(0)
        # x: (B,C,V,T)
        query_v = self.query_w(x)  # (B,C,V,T)
        query_v = rearrange(query_v, 'b c v t -> (b v) t c')

        key_v = self.key_w(x)  # (B,C,V,T)
        key_v = rearrange(key_v, 'b c v t -> (b v) c t')

        value_v = self.value_w(x) # (B,C1,V,T)
        value_v = rearrange(value_v, 'b c1 v t -> (b v) t c1')

        # attention softmax at (B V T T)
        attn = self.softmax(torch.bmm(query_v, key_v))

        attn_value = rearrange(torch.bmm(attn, value_v), '(b v) t c1 -> b c1 v t',b=B)
        attn_value = self.attn_value_w(attn_value) # (B C1 V T) (B,C,V,T)

        return x + self.sigma * attn_value #, attn.view(B,V,T,T)