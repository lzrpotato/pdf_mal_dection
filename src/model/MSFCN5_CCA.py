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

model_name = 'MSFCN5_CCA'

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

            #nn.Conv2d(in_channels=chan_list[2],out_channels=chan_list[3],kernel_size=3,stride=1,padding=1,bias=True),
            #nn.ReLU(),
            ASPP(chan_list[2],chan_list[3]),

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


class ASPP(nn.Module):

    def __init__(self, in_channels, out_channels, conv=nn.Conv2d, norm=nn.BatchNorm2d, momentum=0.0003, mult=1):
        super(ASPP, self).__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.aspp1 = conv(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.aspp2 = conv(in_channels, out_channels, kernel_size=3, stride=1,
                               dilation=int(6*mult), padding=int(6*mult),
                               bias=False)
        self.aspp3 = conv(in_channels, out_channels, kernel_size=3, stride=1,
                               dilation=int(12*mult), padding=int(12*mult),
                               bias=False)
        self.aspp4 = conv(in_channels, out_channels, kernel_size=3, stride=1,
                               dilation=int(18*mult), padding=int(18*mult),
                               bias=False)
        self.aspp5 = conv(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        #self.aspp1_bn = norm(out_channels, momentum)
        #self.aspp2_bn = norm(out_channels, momentum)
        #self.aspp3_bn = norm(out_channels, momentum)
        #self.aspp4_bn = norm(out_channels, momentum)
        #self.aspp5_bn = norm(out_channels, momentum)
        self.conv2 = conv(out_channels * 5, out_channels, kernel_size=1, stride=1,
                               bias=False)
        #self.bn2 = norm(out_channels, momentum)

    def forward(self, x):
        x1 = self.aspp1(x)
        #x1 = self.aspp1_bn(x1)
        x1 = self.relu(x1)
        x2 = self.aspp2(x)
        #x2 = self.aspp2_bn(x2)
        x2 = self.relu(x2)
        x3 = self.aspp3(x)
        #x3 = self.aspp3_bn(x3)
        x3 = self.relu(x3)
        x4 = self.aspp4(x)
        #x4 = self.aspp4_bn(x4)
        x4 = self.relu(x4)
        x5 = self.global_pooling(x)
        logger.debug(f'aspp x5 {x5.shape}')
        x5 = self.aspp5(x)
        #x5 = self.aspp5_bn(x5)
        x5 = self.relu(x5)
        x5 = nn.Upsample((x.shape[2], x.shape[3]), mode='bilinear',
                         align_corners=True)(x5)
        x = torch.cat((x1, x2, x3, x4, x5), 1)
        x = self.conv2(x)
        #x = self.bn2(x)
        x = self.relu(x)
        logger.debug(f'aspp out {x.shape}')
        return x


def conv1x1(in_channelss, out_channelss, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_channelss, out_channelss, kernel_size=3, stride=stride,padding=dilation,
                groups=groups, bias=False, dilation=dilation)

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
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x) # (B,C1,V,T)
        logger.debug(f'proj_query {proj_query.shape}')
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)  # (B,C1,V,T)->(B,T,C1,V)->(B*T,C1,V)->(B*T,V,C1)
        logger.debug(f'proj_query_h {proj_query_H.shape}')
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)  # (B,C1,V,T)->(B,V,C1,T)->(B*V,C1,T)->(B*V,T,C1)
        logger.debug(f'proj_query_W {proj_query_W.shape}')
        proj_key = self.key_conv(x)  # (B,C,V,T)->(B,C1,V,T)
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)  # (B,C1,V,T)->(B,T,C1,V)->(B*T,C1,V)
        logger.debug(f'proj_key_H {proj_key_H.shape}')
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)  # (B,C1,V,T)->(B,V,C1,T)->(B*V,C1,T)
        logger.debug(f'proj_key_W {proj_key_W.shape}')
        proj_value = self.value_conv(x)   # (B,C,V,T)->(B,C,V,T)
        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)  # (B,C,V,T)->(B,T,C,V)->(B*T,C,V)
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)  # (B,C,V,T)->(B,V,C,T)->(B*V,C,T)
        # (B*T,V,C1) * (B*T,C1,V) -> (B*T,V,V) -> (B,T,V,V) -> (B,V,T,V)
        logger.debug('INF {}'.format(self.INF(m_batchsize, height, width, x.device)[0]))
        energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width, x.device)).view(m_batchsize,width,height,height).permute(0,2,1,3)
        # (B*V,T,C1) * (B*V,C1,T) -> (B,V,T,T)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width)
        # (B,V,T,V+T)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        # (B,V,T,V+T) -> (B,V,T,V) -> (B,T,V,V) -> (B*T,V,V)
        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)
        #print(concate)
        #print(att_H) 
        # (B,V,T,V+T) -> (B,V,T,T) -> (B*V,T,T)
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)
        # (B*T,C,V) * (B*T,V,V) -> (B*T,C,V) -> (B,T,C,V) -> (B,C,V,T)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
        # (B*V,C,T) * (B*V,T,T) -> (B*V,C,T) -> (B,V,C,T) -> (B,C,V,T)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)
        #print(out_H.size(),out_W.size())
        # (B,C,V,T) + (B,C,V,T) -> (B,C,V,T)
        return self.gamma*(out_H + out_W) + x

def get_msfcn5cca_oneheadfcn(nclass=2, inplane=1, globalpool='maxpool', num_layers=1, patch_size=(256,256), slide_window=(256*256,256*256/2)):
    return MSFCN5_CCA(nclass, nc=inplane, globalpool=globalpool, num_layers=num_layers,
            multiscale='oneheadfcn', patch_size=patch_size, slide_window=slide_window)

class MSFCN5_CCA(nn.Module):
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

        self.cnn_repre0 = FCN(nc, 1)

        model_dim = self.cnn_repre0.out_dim
        
        self.cca_attn0 = CrissCrossAttention(model_dim)

        if globalpool == 'maxpool':
            self.global_pool = nn.AdaptiveMaxPool2d(1)
        elif globalpool == 'avgpool':
            self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.flatten = nn.Flatten(1)
        self.classifier = self.make_classifier(model_dim,nclass)
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(model_dim+model_dim),
            nn.Linear(model_dim+model_dim, nclass)
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
            logger.debug(f'x_resolu_0 {x_resolu_0.shape}')
            if self.slide_window is None:
                c_in_0, ng_0 = self._view_to_grid(x_resolu_0)
            else:
                c_in_0, ng_0 = self._view_slide_window(x_resolu_0)

            # number of windows, channels, height, width
            nw, c, h, w = c_in_0.size()
            c_in_0 = c_in_0.reshape(1,c,h*w, nw)
            c_out0 = self.cnn_repre0(c_in_0)

            logger.debug(f'c_out.size {c_out0.size()}')
            
            attn_out0 = self.cca_attn0(c_out0)
            
            out0 = self.global_pool(attn_out0)
            out0 = self.flatten(out0)
            
            outs.append(out0)
        
        outs = torch.stack(outs).view(batch, -1)
        logger.debug('outs {}'.format(outs.size()))
        #output = self.mlp_head(outs)
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
    #pcnn = get_msfcn5cca_twoheadfcn(2,1,'maxpool',num_layers=1,patch_size=(patch,patch),slide_window=(patch*patch,patch*patch/2))
    pcnn = get_msfcn5cca_oneheadfcn(2,1,'maxpool',num_layers=1,patch_size=(patch,patch),slide_window=(patch*patch,patch*patch/2))

    with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True,with_flops=True) as prof:
        with record_function("model_inference"):
            print(pcnn(i).shape)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))