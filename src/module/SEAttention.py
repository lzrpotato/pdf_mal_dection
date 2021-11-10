import sys
import torch
import torch.nn as nn
from einops import rearrange
import logging

module_name = 'SEAttention'
logger = logging.getLogger(f'module.{module_name}')

def INF(B,H,W, device):
     return -torch.diag(torch.tensor(float("inf")).to(device).repeat(H),0).unsqueeze(0).repeat(B*W,1,1)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16, mul_input = True):
        super(SELayer, self).__init__()
        self.mul_input = mul_input
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        if self.mul_input:
            return x * y.expand_as(x)
        else:
            return y.expand_as(x)

class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""
    def __init__(self, in_dim, add_input=True):
        super(CrissCrossAttention,self).__init__()
        self.add_input = add_input
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
        
        if self.add_input:
            return self.gamma*(out_H + out_W) + x
        else:
            return self.gamma*(out_H + out_W)

class SEAttention(nn.Module):
    def __init__(self, in_dim, structure, relu=False):
        super().__init__()
        self.out_dim = in_dim
        self.structure = structure
        if self.structure in ['parallel2','parallel1']:
            add_input = False
            mul_input = False
        else:
            add_input = True
            mul_input = True
        self.attn = CrissCrossAttention(in_dim,add_input=add_input)
        self.selayer = SELayer(in_dim,mul_input=mul_input)

        self.relu = nn.ReLU(inplace=True) if relu else None
        
    def forward(self, x):
        if self.structure == 'parallel':
            x1 = self.attn(x)
            x2 = self.selayer(x)
            x = x1 + x2
        elif self.structure == 'parallel1':
            x1 = self.attn(x)
            x2 = self.selayer(x)
            x = x1 + x * x2
        elif self.structure == 'parallel2':
            x1 = self.attn(x)
            x2 = self.selayer(x)
            x = (x + x1) * x2
        elif self.structure == 'serial1':
            x = self.selayer(x)
            x = self.attn(x)
        elif self.structure == 'serial2':
            x = self.attn(x)
            x = self.selayer(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

if __name__ == '__main__':

    logger = logging.getLogger()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(message)s')
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.setLevel(logging.DEBUG)

    input_tensor = torch.randn((1,128,86,44))
    model = SEAttention(128,'parallel')
    
    from torch.profiler import profile, ProfilerActivity, record_function

    with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True,with_flops=True) as prof:
        with record_function("model_inference"):
            logger.debug(model(input_tensor).shape)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))