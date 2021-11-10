import torch.nn as nn
import torchvision.models as models

class MobileNetV3(nn.Module):
    def __init__(self, nclass):
        super().__init__()
        self.nclass = nclass

    def forward(self,x):
        x =self.model(x)
        return x

    def random_weights(self):
        self.model = models.mobilenet_v3_large(pretrained = False)
    
    def pretrained_weights(self):
        self.model = models.mobilenet_v3_large(pretrained = True)
    
    def modify_model(self):
        # modify the layers to fit our desired input/output
        self.model.features[0][0] = nn.Conv2d(1,16, kernel_size=(3,3),stride=(2,2),padding=(1,1),bias=False)
        self.model.classifier[3] = nn.Linear(1280, self.nclass)

    def hook_attention(self):
        outputs = {'feature': None}
        def activation_hook(model, input, output):
            outputs['feature'] = output
        
        self.model.features[16].register_forward_hook(activation_hook)
        return outputs