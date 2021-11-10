import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import sys
import math
sys.path.append('.')

from src.model.model_mobilenetV3 import MobileNetV3
from src.model.model_resnet101 import Resnet101
from src.model.model_squeezenet1_1 import SqueezeNet11
from src.model.model_vgg19 import VGG19

import logging

logger = logging.getLogger('model.cnn_tl')


def get_cnn_tl_model(model, nclass, image_size):
    if model == 'MBNET3':
        model = MobileNetV3(nclass)
    elif model == 'RESNET101':
        model = Resnet101(nclass)
    elif model == 'SZNET11':
        model = SqueezeNet11(nclass)
    elif model == 'VGG19':
        model = VGG19(nclass)
    model.pretrained_weights()
    model.modify_model()
    return CNN_TranLearn(model, image_size=image_size)


class CNN_TranLearn(nn.Module):
    def __init__(self, model, image_size):
        super().__init__()
        self.model = model
        self.image_size = image_size

    def _resize(self, seq):
        padded_seq, h = self._pad_square(seq)
        image = rearrange(padded_seq, '(h1 h2) ->1 1 h1 h2', h1=h, h2=h)
        logger.debug(f'image {image.shape} image_size {self.image_size}')
        return F.interpolate(image, size=(self.image_size))

    def _pad_square(self, seq):
        seq_len = seq.size(0)
        h = math.ceil(math.sqrt(seq_len))
        need = (h*h) - seq_len
        if need > 0:
            return F.pad(seq, (0,need)), h
        return seq, h

    def forward(self, x):
        images = []
        for seq in x:
            s0 = seq[0]
            img = self._resize(s0)
            logger.debug(f'img shape {img.shape}')
            images.append(img)
        
        img = torch.vstack(images)
        logger.debug(f'img {img.shape}')
        return self.model(img)


if __name__ == '__main__':
    import sys
    logger = logging.getLogger()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(message)s')
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.setLevel(logging.DEBUG)
    i = [[torch.randn((4248001)),torch.randn((2124000))],[torch.randn(31232)]]
    #i_embed = [[torch.randint(0,255,(1000,)),torch.randint(0,65535, (3000,))]]
    from torch.profiler import profile, ProfilerActivity, record_function
    #logger.debug(f'i_embed {i_embed}')
    patch = 64
    
    pcnn = get_cnn_tl_model('MBNET3',2,(256,256))

    with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True,with_flops=True) as prof:
        with record_function("model_inference"):
            print(pcnn(i).shape)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))