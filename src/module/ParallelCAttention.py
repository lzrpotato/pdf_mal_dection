import sys
import torch.nn as nn
import torch
from einops import rearrange
import logging

module_name = 'ParallelCAttention'
logger = logging.getLogger(f'module.{module_name}')

def conv1x1(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,padding=dilation,
                groups=groups, bias=False, dilation=dilation)


class ParallelCAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        out_dim = in_channels//4
        self.down_channel = nn.Conv2d(in_channels, out_dim, kernel_size=1,padding=1)
        self.cca = CrissCrossAttention(out_dim)
        self.cattn = cAttn(out_dim)
        self.out_dim = out_dim

    def forward(self, x):
        x = self.down_channel(x)
        x1 = self.cca(x)
        x2 = self.cattn(x)
        x = x + x1 + x2
        logger.debug(f'x out {x.shape}')
        return x


def INF(B,H,W, device):
     return -torch.diag(torch.tensor(float("inf")).to(device).repeat(H),0).unsqueeze(0).repeat(B*W,1,1)


class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""
    def __init__(self, in_dim):
        super(CrissCrossAttention,self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))


    def forward(self, x):
        # (B,C,V,T)
        B, _, V, T = x.size()

        # query
        proj_query = self.query_conv(x) # (B,C1,V,T)
        proj_query_H = rearrange(proj_query, 'b c v t -> (b t) v c')
        proj_query_W = rearrange(proj_query, 'b c v t -> (b v) t c')
        
        # key
        proj_key = self.key_conv(x)
        proj_key_H = rearrange(proj_key, 'b c v t -> (b t) c v')
        proj_key_W = rearrange(proj_key, 'b c v t -> (b v) c t')

        # value
        proj_value = self.value_conv(x)
        proj_value_H = rearrange(proj_value, 'b c v t -> (b t) c v')
        proj_value_W = rearrange(proj_value, 'b c v t -> (b v) c t')

        energy_H = rearrange((torch.bmm(proj_query_H, proj_key_H)+self.INF(B, V, T, x.device)),'(b t) v v1 -> b v t v1', b = B)
        energy_W = rearrange(torch.bmm(proj_query_W, proj_key_W), '(b v) t t1 -> b v t t1', b = B)

        # attention softmax at (B V T (V1 T1))
        concate = self.softmax(torch.cat([energy_H, energy_W], -1))

        att_H = rearrange(concate[:,:,:,0:V], 'b v t v1 -> (b t) v1 v')
        att_W = rearrange(concate[:,:,:,V:V+T], 'b v t t1 -> (b v) t1 t')

        out_H = rearrange(torch.bmm(proj_value_H, att_H),'(b t) c v -> b c v t', b = B)
        out_W = rearrange(torch.bmm(proj_value_W, att_W),'(b v) c t -> b c v t', b = B)
        
        return self.gamma*(out_H + out_W)

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
    model = ParallelCAttention(128)
    
    from torch.profiler import profile, ProfilerActivity, record_function

    with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True,with_flops=True) as prof:
        with record_function("model_inference"):
            logger.debug(model(input_tensor).shape)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))