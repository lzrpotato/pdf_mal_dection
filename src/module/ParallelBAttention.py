import sys
import torch.nn as nn
import torch
from einops import rearrange
import logging

module_name = 'ParallelBAttention'
logger = logging.getLogger(f'module.{module_name}')

def conv1x1(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,padding=dilation,
                groups=groups, bias=False, dilation=dilation)


class ParallelBAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        
        self.vattn = vAttn(in_channels)
        self.tattn = tAttn(in_channels)
        self.cattn = cAttn(in_channels)
        self.out_dim = in_channels

    def forward(self, x):
        x1 = self.vattn(x)
        x2 = self.tattn(x)
        x3 = self.cattn(x)
        x = x + x1 + x2 + x3
        logger.debug(f'x out {x.shape}')
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

        return self.sigma * attn_value
        

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

        return self.sigma * attn_value #, attn.view(B,V,T,T)

class cAttn(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query_w = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.key_w = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.value_w = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.attn_value_w = conv1x1(in_channels,in_channels)

        self.softmax = nn.Softmax(dim=-1)
        self.sigma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        B = x.size(0)
        # x: (B,C,V,T)
        query_c = self.query_w(x)  # (B,C,V,T)
        query_c = rearrange(query_c, 'b c v t -> (b v) c t')

        key_c = self.key_w(x)  # (B,C,V,T)
        key_c = rearrange(key_c, 'b c v t -> (b v) t c')

        value_c = self.value_w(x) # (B,C1,V,T)
        value_c = rearrange(value_c, 'b c v t -> (b v) c t')

        # attention softmax at (B V C C)
        attn = self.softmax(torch.bmm(query_c, key_c))

        attn_value = rearrange(torch.bmm(attn, value_c), '(b v) c t -> b c v t',b=B)
        attn_value = self.attn_value_w(attn_value) # (B C V T) (B,C,V,T)

        return self.sigma * attn_value #, attn.view(B,V,T,T)


if __name__ == '__main__':

    logger = logging.getLogger()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(message)s')
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.setLevel(logging.DEBUG)

    input_tensor = torch.randn((1,128,86,44))
    model = ParallelBAttention(128)
    
    from torch.profiler import profile, ProfilerActivity, record_function

    with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True,with_flops=True) as prof:
        with record_function("model_inference"):
            logger.debug(model(input_tensor).shape)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))