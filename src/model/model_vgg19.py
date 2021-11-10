"""
Tyler Nichols
New Mexico Tech 2021 REU - Lab 5
Adapted VGG 19
"""
import torch.nn as nn
import torchvision.models as models

class VGG19(nn.Module):
    def __init__(self, nclass):
        super().__init__()
        self.nclass = nclass

    def forward(self,x):
        x =self.model(x)
        return x
    def random_weights(self):
        self.model = models.vgg19(pretrained = False)
    def pretrained_weights(self):
        self.model = models.vgg19(pretrained = True)
    # modify the layers to fit our desired input/output
    def modify_model(self):
        self.model.features[0] = nn.Conv2d(1, 64, kernel_size=(3,3), stride=(1,1),padding=(1,1))
        self.model.classifier[6] = nn.Linear(in_features=4096, out_features=self.nclass, bias=True)

