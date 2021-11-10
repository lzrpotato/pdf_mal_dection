import sys

sys.path.append('.')
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

logger = logging.getLogger('model.PCNN_LSTM')

__all__ = ['PCNN_LSTM']

class MobileNetV3(nn.Module):
    def __init__(self, inplane, random_weight=False):
        super().__init__()
        self.name = 'mobilenetv3'
        self.inplane = inplane
        if random_weight:
            self.random_weights()
        else:
            self.pretrained_weights()
        
        self.modify_model()

    def forward(self,x):
        x =self.model(x)
        return x
    def random_weights(self):
        self.model = models.mobilenet_v3_large(pretrained = False)
    def pretrained_weights(self):
        self.model = models.mobilenet_v3_large(pretrained = True)
    # modify the layers to fit our desired input/output
    def modify_model(self):
        self.model.features[0][0] = nn.Conv2d(self.inplane,16, kernel_size=(3,3),stride=(2,2),padding=(1,1),bias=False)
        self.model.classifier = nn.Sequential()


class PCNN(nn.Module):
    def __init__(self, cnn_model_name, inplane, dropout=0.5, freeze=False):
        super().__init__()
        self.cnn_model_name = cnn_model_name
        self.inplane = inplane
        if self.cnn_model_name == 'mobilenetv3':
            self.model = MobileNetV3(inplane)
        else:
            raise UnboundLocalError(f'self.model {self.cnn_model_name}')
        bls = [nn.Flatten(1,-1)]

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        if dropout:
            bls += [nn.Dropout(dropout)]
        
        self.cnn_out = nn.Sequential(*bls)
        self.out_dim = 960

    def forward(self, x):
        x = self.model(x)
        x = self.cnn_out(x)
        return x

def get_lstm(input_size,hidden_size=32,num_layers=1,batch_first=True,dropout=0.5,bidirectional=True):
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
        batch_size,
        pool,
    ):
        super().__init__()
        self.in_size = in_size
        self.hidden = hidden
        self.bidirection = bidirection
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.pool = pool
        self.lstm = get_lstm(self.in_size,self.hidden,self.num_layers,bidirectional=bidirection)
        if self.bidirection:
            D = 2
        else:
            D = 1
        self.h0 = torch.zeros((self.num_layers*D,self.batch_size,self.hidden),dtype=torch.float32)
        self.c0 = torch.zeros((self.num_layers*D,self.batch_size,self.hidden),dtype=torch.float32)
        logger.debug('in_size {} hidden {} nl {} bs {} pool {}'.format(in_size,hidden,num_layers,batch_size,pool))
        if pool:
            self.pooling = nn.AdaptiveAvgPool1d(1)
        
        self.flatten = nn.Flatten(1,-1)
        self.out_dim = self.hidden * (2 if self.bidirection else 1)

    def forward(self, x):
        h0 = self.h0.to(x.device)
        c0 = self.c0.to(x.device)
        logger.debug('lstm forward {} {} {}'.format(x.size(), h0.size(), c0.size()))
        out, (hn,cn) = self.lstm(x,(h0,c0))
        if self.pool:
            logger.debug('lstm out {}'.format(out.size()))
            batch, seq, hidden = out.size()
            out = self.pooling(out.view(batch,hidden,seq))
        else:
            out = hn
        
        out = self.flatten(out)
        return out

class PCNN_LSTM(nn.Module):
    def __init__(self, 
        nclass, 
        nc, 
        cnn_model_name,
        freeze,
        hidden, 
        bidirection,
        batch_size, 
        num_layers,
        patch_size=(256,256),
        slide_window=(256*256,256*256/2)
    ):
        super().__init__()
        self.nclass = nclass
        self.nc = nc
        self.cnn_model_name = cnn_model_name
        self.freeze = freeze
        self.hidden = hidden
        self.bidirection = bidirection
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.slide_window = slide_window
        # input C_in, W, H, output C_out, W_o, H_o
        num_block = patch_size[0]//4
        self.cnn_repre = PCNN(self.cnn_model_name, inplane=nc, freeze=freeze)
        # input 
        self.lstm = LSTM_rnn(self.cnn_repre.out_dim,self.hidden,self.bidirection,self.num_layers,1,pool=True)
        
        self.classifier = self.make_classifier(self.lstm.out_dim,nclass)

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

    def forward_(self, x):
        batch, c, seq_len = x.size()
        logger.debug('{} {} {}'.format(batch, c, seq_len))
        c_in, ng = self._view_to_grid(x)
        c_out = self.cnn_repre(c_in)
    
        r_in = c_out.view(batch,ng,-1)
        logger.debug('in {} cout {} rin {}'.format(c_in.size(), c_out.size(), r_in.size()))
        r_out = self.lstm(r_in)
        logger.debug('r_out {}'.format(r_out.size()))
        x = self.classifier(r_out)

        return x
    
    def forward(self, x):
        batch = len(x)
        r_outs = []
        # for each pdf file, we divide it into non-overlap patches
        for i in x:
            seq_len = i.size()[0]
            logger.debug('{}'.format(seq_len))
            xi = i.view(1,self.nc,seq_len)
            if self.slide_window is None:
                c_in, ng = self._view_to_grid(xi)
            else:
                c_in, ng = self._view_slide_window(xi)
            
            c_out = self.cnn_repre(c_in)
        
            r_in = c_out.view(1,ng,-1)
            logger.debug('in {} cout {} rin {}'.format(c_in.size(), c_out.size(), r_in.size()))
            
            r_out = self.lstm(r_in)
            logger.debug('r_out {}'.format(r_out.size()))
            r_outs.append(r_out)
        
        r_outs = torch.stack(r_outs).view(batch,-1)
        logger.debug('r_outs {}'.format(r_outs.size()))
        output = self.classifier(r_outs)
        
        return output

if __name__ == '__main__':
    logger = logging.getLogger()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(message)s')
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.setLevel(logging.DEBUG)
    i = torch.ones((1,2048000))
    pcnn = PCNN_LSTM(2, 1, 'mobilenetv3',False ,64, bidirection=True,batch_size=32, num_layers=3)
    #print(pcnn)
    print(pcnn(i).shape)
