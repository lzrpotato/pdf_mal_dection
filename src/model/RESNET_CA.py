import sys
from typing import Optional, Callable, Type
from torch import Tensor

sys.path.append('.')
import os

print(os.getcwd())
import logging

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

logger = logging.getLogger('model.Resnet_CA')

__all__ = ['Resnet_CA']

def memory_usage(tensor_size):
    max_memory = torch.cuda.max_memory_allocated()
    if max_memory == 0:
        usage = 0
    else:
        usage = torch.cuda.memory_allocated() / max_memory
    logger.debug(f'[memory usage] {usage} {tensor_size}')

###########################################
def resnet18(num_classes):
    return Resnet(BasicBlock, layers=[2,2,2,2], num_classes=num_classes, norm_layer=nn.GroupNorm)

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.norm1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.norm2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Resnet(nn.Module):
    def __init__(self, 
        block: Type[BasicBlock], 
        layers=[2, 2, 2, 2], 
        num_classes=2, 
        zero_init_residual=False, 
        groups=1, 
        width_per_group=64,
        norm_layer=None
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        
        self.inplanes = 64
        self.dilation = 1
        replace_stride_with_dilation = [False, False, False]

        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3,
                bias=False)
        self.norm1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1],stride=2,
                                        dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2],stride=2,
                                        dilate=replace_stride_with_dilation[1])
        #self.layer4 = self._make_layer(block, 512, layers[3],stride=2,
        #                                 dilate=replace_stride_with_dilation[2])
        #self.avgpool = nn.AdaptiveAvgPool2d(1)
        #self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

        self.out_dim = 256

    def _make_layer(self, block, planes, blocks, stride = 1, dilate = False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                #norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        #x = self.layer4(x)

        #logger.debug(f'avgpool {x.shape}')
        #x = self.avgpool(x)
        
        x# = torch.flatten(x, 1)
        x# = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        # batch, fea = x.size()
        # x = x.reshape(batch, 1, fea)
        return self._forward_impl(x)



def vconv1x1(in_planes, out_planes, stride=1, groups=1, dilation=1):
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

        self.query_w = vconv1x1(in_channels,out_channels)
        self.key_w = vconv1x1(in_channels,out_channels)
        self.value_w = vconv1x1(in_channels,in_channels//2)
        self.attn_value_w = vconv1x1(in_channels//2,in_channels)

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

        return x + self.sigma * attn_value, attn
        

class tAttn(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        out_channels = in_channels//8
        # self.query_w = nn.utils.spectral_norm(conv1x1(in_channels,out_channels))
        # self.key_w = nn.utils.spectral_norm(conv1x1(in_channels,out_channels))
        # self.value_w = nn.utils.spectral_norm(conv1x1(in_channels,in_channels//2))
        # self.attn_value_w = nn.utils.spectral_norm(conv1x1(in_channels//2,in_channels))

        self.query_w = vconv1x1(in_channels,out_channels)
        self.key_w = vconv1x1(in_channels,out_channels)
        self.value_w = vconv1x1(in_channels,in_channels//2)
        self.attn_value_w = vconv1x1(in_channels//2,in_channels)

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
        
        return x + self.sigma * attn_value, attn.view(B,V,T,T)

class Resnet_CA(nn.Module):
    def __init__(self, 
        nclass, 
        nc, 
        batch_size,
        globalpool, 
        num_layers,
        patch_size=(256,256),
        slide_window=(256*256,256*256/2)
    ):
        super().__init__()
        self.nclass = nclass
        self.nc = nc
        self.batch_size = batch_size
        self.globalpool = globalpool
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.slide_window = slide_window
        # input C_in, W, H, output C_out, W_o, H_o
        num_block = patch_size[0]//4
        self.cnn_repre = resnet18(nclass)
        # input 
        self.t_attn = tAttn(in_channels=self.cnn_repre.out_dim)
        self.v_attn = vAttn(in_channels=self.cnn_repre.out_dim)
        
        if globalpool == 'maxpool':
            self.global_pool = nn.AdaptiveMaxPool2d(1)
        elif globalpool == 'avgpool':
            self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.flatten = nn.Flatten(1)
        self.classifier = self.make_classifier(self.cnn_repre.out_dim,nclass)

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
    
    def forward(self, x):
        batch = len(x)
        
        outs = []
        # for each pdf file, we divide it into non-overlap patches
        for i in x:
            seq_len = i.size()[0]
            logger.debug('i shape {}'.format(i.size()))
            xi = i.view(1,self.nc,seq_len)
            if self.slide_window is None:
                c_in, ng = self._view_to_grid(xi)
            else:
                c_in, ng = self._view_slide_window(xi)
            
            #memory_usage(c_in.size())
            logger.debug(f'c_in size {c_in.size()}')
            # number of windows, channels, height, width
            nw, c, h, w = c_in.size()
            # 
            c_in = c_in.reshape(1,c,h*w, nw)
            logger.debug(f'c_in size {c_in.size()}')
            c_out = self.cnn_repre(c_in)
            logger.debug(f'c_out.size {c_out.size()}')
            #T, C, H, W = c_out.size()
            #c_out = c_out.view(1,T,C,H*W).permute(0,2,1,3)
            #logger.debug(f'c_out.size {c_out.size()}')

            t_attn_out, t_attn = self.t_attn(c_out)
            logger.debug(f't_attn out {t_attn_out.shape} {t_attn.shape}')
            v_attn_out, v_attn = self.v_attn(c_out)
            logger.debug(f'v_attn out {v_attn_out.shape} {v_attn.shape}')
            out = self.global_pool(v_attn_out)
            out = self.flatten(out)
            logger.debug(f'globalpool {out.shape}')
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
    i = torch.ones((1,4248000))
    from torch.profiler import profile, ProfilerActivity, record_function

    patch = 64
    pcnn = Resnet_CA(2,1,32,'maxpool',num_layers=1,patch_size=(patch,patch),slide_window=(patch*patch,patch*patch/2))
    
    with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True) as prof:
        with record_function("model_inference"):
            print(pcnn(i).shape)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))