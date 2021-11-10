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

logger = logging.getLogger('model.MSFCN1_CA')

__all__ = ['MSFCN1_CA']

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

class vAttn(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        out_channels = in_channels//8
        # self.query_w = nn.utils.spectral_norm(conv1x1(in_channels,out_channels))
        # self.key_w = nn.utils.spectral_norm(conv1x1(in_channels,out_channels))
        # self.value_w = nn.utils.spectral_norm(conv1x1(in_channels,in_channels//2))
        # self.attn_value_w = nn.utils.spectral_norm(conv1x1(in_channels//2,in_channels))

        self.query_w = conv1x1(in_channels,out_channels)
        self.key_w = conv1x1(in_channels,out_channels)
        self.value_w = conv1x1(in_channels,in_channels//2)
        self.attn_value_w = conv1x1(in_channels//2,in_channels)

        self.softmax = nn.Softmax(2)
        self.sigma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x: (B,C,V,T)
        query_v = self.query_w(x) # (B,C,V,T)
        B,C,V,T = query_v.size()
        query_v = query_v.permute(0,3,2,1) # (B,T,V,C)
        key_v = self.key_w(x)     # (B,C,V,T)
        key_v = key_v.permute(0,3,2,1) # (B,T,V,C)
        value_v = self.value_w(x) # (B,C1,V,T)
        C1 = value_v.size(1)
        value_v = value_v.permute(0,3,2,1)  # (B,T,V,C1)
        logger.debug(f'v_attn query_v {query_v.shape} key_v {key_v.shape}')

        attn = torch.bmm(query_v.view(B*T,V,C), key_v.view(B*T,V,C).permute(0,2,1))  # (B*T,V,C) * (B*T,C,V) = (B*T,V,V)
        attn = self.softmax(attn)  # (B*T,V,V)
        attn_value = torch.bmm(attn, value_v.view(B*T,V,C1))  # (B*T,V,V) * (B*T,V,C1) = (B*T,V,C1)
        attn_value = self.attn_value_w(attn_value.view(B,T,V,C1).permute(0,3,2,1))  # (B,C1,V,T) -> (B,C,V,T)

        return x + self.sigma * attn_value
        

class tAttn(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        out_channels = in_channels//8
        # self.query_w = nn.utils.spectral_norm(conv1x1(in_channels,out_channels))
        # self.key_w = nn.utils.spectral_norm(conv1x1(in_channels,out_channels))
        # self.value_w = nn.utils.spectral_norm(conv1x1(in_channels,in_channels//2))
        # self.attn_value_w = nn.utils.spectral_norm(conv1x1(in_channels//2,in_channels))

        self.query_w =conv1x1(in_channels,out_channels)
        self.key_w = conv1x1(in_channels,out_channels)
        self.value_w = conv1x1(in_channels,in_channels//2)
        self.attn_value_w = conv1x1(in_channels//2,in_channels)

        self.softmax = nn.Softmax(dim=2)
        self.sigma = nn.Parameter(torch.zeros(1))


    def forward(self, x):
        # x: (B,C,V,T)
        logger.debug(f'tAttn {x.shape} ')
        query_v = self.query_w(x)  # (B,C,V,T)
        query_v = query_v.permute(0,2,3,1)  # (B,V,T,C)
        logger.debug(f'tAttn query_v {query_v.shape} ')
        key_v = self.key_w(x)  # (B,C,V,T)
        key_v = key_v.permute(0,2,3,1)   # (B,V,T,C)
        logger.debug(f'tAttn key_v {key_v.shape} ')
        value_v = self.value_w(x) # (B,C1,V,T)
        C1 = value_v.size(1)
        value_v = value_v.permute(0,2,3,1) # （B,V,T,C1)

        logger.debug(f'tAttn value_v {value_v.shape} ')
        B,V,T,C = query_v.size()
        logger.debug(f'[bmm] query_v {query_v.view(B*V,T,C).shape} key_v {key_v.view(B*V,T,C).shape}')
        attn = torch.bmm(query_v.view(B*V,T,C), 
                    key_v.view(B*V,T,C).permute(0,2,1)) # (B*V,T,C) * (B*V,C,T) = (B*V,T,T)
        
        attn = self.softmax(attn)  # (B*V,T,T)

        logger.debug(f'[bmm] attn {attn.shape} value_v {value_v.view(B*V,T,C1).shape}')
        attn_value = torch.bmm(attn, value_v.view(B*V,T,C1)) # (B*V,T,T) * (B*V,T,C1)
        attn_value = attn_value.view(B,V,T,C1).permute(0,3,1,2) # (B,C1,V,T)
        attn_value = self.attn_value_w(attn_value) # (B,C,V,T)

        return x + self.sigma * attn_value #, attn.view(B,V,T,T)

def get_lstm(input_size,hidden_size=32,num_layers=1,batch_first=True,dropout=0.5,bidirectional=True):
    if num_layers == 1:
        dropout=0
    return nn.LSTM(input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional)

class LSTM_rnn(nn.Module):
    def __init__(self,
        in_size,
        hidden,
        bidirection,
        num_layers,
    ):
        super().__init__()
        self.in_size = in_size
        self.hidden = hidden
        self.bidirection = bidirection
        self.num_layers = num_layers
        self.lstm = get_lstm(self.in_size,self.hidden,self.num_layers,bidirectional=bidirection)
        logger.debug('in_size {} hidden {} nl {} '.format(in_size,hidden,num_layers))
        
        self.flatten = nn.Flatten(1,-1)
        self.out_dim = self.hidden * (2 if self.bidirection else 1)

    def forward(self, x):
        # (B,C,V,T)
        logger.debug(f'lstm x shape {x.shape}')
        B,C,V,T = x.size()
        x = x.permute(0,3,1,2).view(B,T,C*V)
        logger.debug(f'lstm x shape permute {x.shape}')
        out, (hn,cn) = self.lstm(x) # (B,T,C*V) -> (B,T,H)
        out = out.view(B,T,C,V).permute(0,2,3,1)

        logger.debug(f'out shape {out.shape}')

        return out


def get_msfcnlstmca_twoheadfcn(nclass=2, inplane=1, globalpool='maxpool', num_layers=1, patch_size=(256,256), slide_window=(256*256,256*256/2)):
    return MSFCN_LSTM_CA(nclass, nc=inplane, globalpool=globalpool, num_layers=num_layers,
            multiscale='twoheadfcn', patch_size=patch_size, slide_window=slide_window)

def get_msfcnlstmca_twoheadfcnca(nclass=2, inplane=1, globalpool='maxpool', num_layers=1, patch_size=(256,256), slide_window=(256*256,256*256/2)):
    return MSFCN_LSTM_CA(nclass, nc=inplane, globalpool=globalpool, num_layers=num_layers,
            multiscale='twoheadfcnca', patch_size=patch_size, slide_window=slide_window)

def get_msfcnlstmca_oneheadfcn(nclass=2, inplane=1, globalpool='maxpool', num_layers=1, patch_size=(256,256), slide_window=(256*256,256*256/2)):
    return MSFCN_LSTM_CA(nclass, nc=inplane, globalpool=globalpool, num_layers=num_layers,
            multiscale='oneheadfcn', patch_size=patch_size, slide_window=slide_window)

class MSFCN_LSTM_CA(nn.Module):
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
        
        lstm_dim0 = model_dim*(patch_size[0]*patch_size[1]//(6*4*2*1)+1)
        lstm_dim1 = model_dim*(patch_size[0]*patch_size[1]//(6*4*2*1)+1)
        self.rnn0 = LSTM_rnn(lstm_dim0, lstm_dim0, bidirection=False, num_layers=1)
        self.rnn1 = LSTM_rnn(lstm_dim1, lstm_dim1, bidirection=False, num_layers=1)

        self.ca_attn0 = nn.Sequential(
                tAttn(in_channels=model_dim),
                vAttn(in_channels=model_dim)
            )
        # input 
        if self.multiscale == 'twoheadfcnca':
            self.ca_attn1 = nn.Sequential(
                tAttn(in_channels=model_dim),
                vAttn(in_channels=model_dim)
            )

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
            else:
                pass
            logger.debug(f'c_out.size {c_out0.size()}')
            
            
            r_out0 = self.rnn0(c_out0)
            r_out1 = self.rnn1(c_out1)
            
            attn_out0 = self.ca_attn0(r_out0)
            
            if self.multiscale == 'twoheadfcn':
                attn_out1 = self.ca_attn0(r_out1)
            elif self.multiscale == 'twoheadfcnca':
                attn_out1 = self.ca_attn1(r_out1)
            

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
    i = torch.ones((2,4248001))
    from torch.profiler import profile, ProfilerActivity, record_function

    patch = 64
    #pcnn = get_msfcn1ca_twoheadfcn(2,1,'maxpool',num_layers=1,patch_size=(patch,patch),slide_window=(patch*patch,patch*patch/2))
    pcnn = get_msfcnlstmca_twoheadfcn(2,1,'maxpool',num_layers=1,patch_size=(patch,patch),slide_window=(patch*patch,patch*patch/2))

    with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True,with_flops=True) as prof:
        with record_function("model_inference"):
            print(pcnn(i).shape)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))