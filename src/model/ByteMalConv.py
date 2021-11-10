import torch
from torch import nn
import sys
import logging
import torch.nn.functional as F

logger = logging.getLogger('model.MalConv')

class ByteMalConv(nn.Module):
    def __init__(self, nclass):
        super().__init__()
        self.padding_len = 2097152
        self.cnn = MalConv(nclass, self.padding_len)
        
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


class MalConv(nn.Module):
    def __init__(self, nclass, input_length=2097152):
        super().__init__()
        self.input_length = input_length
        window_size = 500
        self.embedding = nn.Embedding(256,8, padding_idx=0)
        self.conv_1 = nn.Conv1d(8,128,kernel_size=window_size,stride=window_size)
        self.conv_2 = nn.Conv1d(8,128,kernel_size=window_size,stride=window_size)
        logger.debug(f'{int(input_length/window_size)}')
        self.pooling = nn.MaxPool1d(int(input_length/window_size))
        self.fc_1 = nn.Linear(128,128)
        self.fc_2 = nn.Linear(128,nclass)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0,2,1)
        logger.debug(f'embed {x.shape}')
        cnn_value = self.conv_1(x)
        gating_weight = self.sigmoid(self.conv_2(x))
        
        x = cnn_value * gating_weight
        x = self.pooling(x)

        x = x.view(-1,128)
        x = self.fc_1(x)
        x = self.fc_2(x)
        return x


if __name__ == '__main__':
    logger = logging.getLogger()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(message)s')
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.setLevel(logging.DEBUG)
    i = [[torch.randint(0,255,(8,4248000,)),torch.randint(0,255,(2124000,))]]

    from torch.profiler import profile, ProfilerActivity, record_function

    pcnn = ByteMalConv(2)
    with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True) as prof:
        with record_function("model_inference"):
            print(pcnn(i).shape)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))