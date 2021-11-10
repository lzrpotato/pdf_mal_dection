import torch
from torch import nn
import sys
import logging
import torch.nn.functional as F
from einops import rearrange

logger = logging.getLogger('model.ByteCNN_RF')

class CNNYSJ(nn.Module):
    def __init__(self, nclass):
        super().__init__()
        self.cnn = CNN_YSJ(nclass)
        self.padding_len = 1000

    def _pad_fixed(self, x: torch.Tensor, padding_len):
        """
        pad pdf file to fit the grid shape
        """
        seq_len = x.size()[0]
        
        need = padding_len - seq_len
        logger.debug('need {} size {}'.format(need, seq_len))
        if need < 0:
            x_padded = x.narrow(0, 0, padding_len)
        else:
            x_padded = F.pad(x,(0,need))
        return x_padded

    def forward(self, x):
        outs = []
        for seq in x:
            s0 = seq[0]
            logger.debug(f's0 {s0.shape}')
            seq_padded = self._pad_fixed(seq[0], self.padding_len).view(1,-1)
            logger.debug(f'seq_padded {seq_padded.shape}')
            out = self.cnn(seq_padded)
            outs.append(out)
        outs = torch.vstack(outs)
        return outs


class CNN_YSJ(nn.Module):
    def __init__(self, nclass):
        super().__init__()
        # E, L: E=25, 
        self.embedding = nn.Embedding(256,25)
        self.conv1 = nn.Conv2d(1,32, kernel_size=(3,25), stride=(1,25),padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32,64,kernel_size=(3,1),stride=(1,1),padding=(1,0))
        self.bn2 = nn.BatchNorm2d(64)
        self.pooling = nn.MaxPool2d(kernel_size=(100,1),stride=(100,1))
        self.flatten = nn.Flatten(1)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(640, 128)
        self.dropout = nn.Dropout(p=0.25)
        self.fc2 = nn.Linear(128,nclass)

    def forward(self, x):
        x = self.embedding(x)
        logger.debug(f'embed {x.shape}')
        x = self.relu(x)
        x = rearrange(x, 'b d l -> b 1 d l')
        x = self.conv1(x)
        x = self.bn1(x)
        logger.debug(f'conv1 {x.shape}')
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        logger.debug(f'conv2 out {x.shape}')
        x = self.pooling(x)
        logger.debug(f'pool out {x.shape}')
        x = self.relu(x)
        x = self.flatten(x)
        logger.debug(f'flatten out {x.shape}')
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    logger = logging.getLogger()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(message)s')
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.setLevel(logging.DEBUG)
    i = [[torch.randint(0,255,(4248001,)),torch.randint(0,255,(2124000,))]]

    from torch.profiler import profile, ProfilerActivity, record_function

    pcnn = CNNYSJ(2)
    with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True) as prof:
        with record_function("model_inference"):
            print(pcnn(i).shape)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))