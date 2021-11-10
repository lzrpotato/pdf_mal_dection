import torch
import torch.nn as nn
from einops import rearrange
import logging

logger = logging.getLogger('module.CrissCrossAttention')

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
        
        return self.gamma*(out_H + out_W) + x