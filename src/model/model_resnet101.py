"""
Tyler Nichols
New Mexico Tech 2021 REU - Lab 4
Adapted Resnet-101
"""

import torch.nn as nn
import torchvision.models as models


class Resnet101(nn.Module):
    def __init__(self, nclass):
        super().__init__()
        self.nclass = nclass

    def forward(self,x):
        x =self.model(x)
        return x
    def random_weights(self):
        self.model = models.resnet101(pretrained = False)
    def pretrained_weights(self):
        self.model = models.resnet101(pretrained = True)
    # modify the layers to fit our desired input/output
    def modify_model(self):
        self.model.conv1 = nn.Conv2d(1,64, kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=False)
        self.model.fc = nn.Linear(in_features=2048, out_features=self.nclass, bias=True)

