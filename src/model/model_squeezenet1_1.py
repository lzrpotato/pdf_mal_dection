import torch.nn as nn
import torchvision.models as models

class SqueezeNet11(nn.Module):
    def __init__(self, nclass):
        super().__init__()
        self.nclass = nclass

    def forward(self,x):
        x =self.model(x)
        return x
    def random_weights(self):
        self.model = models.squeezenet1_1(pretrained = False)
    def pretrained_weights(self):
        self.model = models.squeezenet1_1(pretrained = True)
    # modify the layers to fit our desired input/output
    def modify_model(self):
        self.model.features[0] = nn.Conv2d(1,64,kernel_size=(3,3),stride=(2,2))
        self.model.classifier[1] = nn.Conv2d(512,self.nclass,(1,1),(1,1))
