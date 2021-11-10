import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import sys

logger = logging.getLogger('model.ByteCNN_RF')

class ModelA():
    def __init__(self, in_channels, nclass):
        self.conv1 = nn.Conv1d(in_channels, 128, kernel_size=3,stride=4, padding=10)
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.linear1 = nn.Linear(128, nclass)

class ModelB(nn.Module):
    def __init__(self, in_channels, nclass):
        self.conv1 = nn.Conv1d(in_channels, 128)
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.linear1 = nn.Linear()

class ModelC(nn.Module):
    def __init__(self, nclass):
        super().__init__()
        self.padding_len = 204800
        self.embedding = nn.Embedding(256,16)
        self.feature = nn.Sequential(
            nn.Conv1d(in_channels=16,out_channels=20, kernel_size=16,stride=4,padding=4),
            nn.BatchNorm1d(20),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=16,stride=4,padding=4),
            nn.Conv1d(20, 40, kernel_size=16,stride=4,padding=4),
            nn.BatchNorm1d(40),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=16,stride=4,padding=4),
            nn.Conv1d(40,80,kernel_size=4, stride=2,padding=2),
            nn.BatchNorm1d(80),
            nn.ReLU()
        )
        self.global_max_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(80, 80),
            nn.ReLU(),
            nn.BatchNorm1d(80),
            nn.Linear(80, nclass)
        )

    def _pad_fixed(self, x: torch.Tensor):
        """
        pad pdf file to fit the grid shape
        """
        seq_len = x.size()[0]
        
        need = self.padding_len - seq_len
        logger.debug('need {} size {}'.format(need, seq_len))
        if need < 0:
            x_padded = x.narrow(0, 0, self.padding_len)
        else:
            x_padded = F.pad(x,(0,need))
        return x_padded
        

    def forward(self, data):
        batch = len(data)
        logger.debug(f'batch {batch}')
        # for each pdf file, we divide it into non-overlap patches
        pad_x = []
        for seq in data:
            seq0 = seq[0]
            logger.debug(f'seq0 {seq0.shape}')
            pad_seq = self._pad_fixed(seq0)
            logger.debug(f'pad_seq {pad_seq.shape}')
            pad_x.append(pad_seq)

        logger.debug(f'pad_x {len(seq0)}')
        x = torch.stack(pad_x)
        logger.debug(f'x {x.shape}')
        x = self.embedding(x)
        logger.debug(f'embedding out {x.shape}')
        B, L, D = x.size()
        x = x.view(B,L,D).permute(0,2,1)
        logger.debug(f'permute {x.shape}')
        x = self.feature(x)
        logger.debug(f'feature out {x.shape}')
        x = self.global_max_pool(x)
        logger.debug(f'global_max_pool out {x.shape}')
        x = self.classifier(x)
        logger.debug(f'classifier out {x.shape}')
        return x


if __name__ == '__main__':
    logger = logging.getLogger()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(message)s')
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.setLevel(logging.DEBUG)
    i = [[torch.randint(0,255,(500000,))],[torch.randint(0,255,(500000,))]]

    from torch.profiler import profile, ProfilerActivity, record_function

    pcnn = ModelC(2)
    with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True) as prof:
        with record_function("model_inference"):
            print(pcnn(i).shape)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))