import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import sys

logger = logging.getLogger('model.UNET_CA')


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        #self.outc = OutConv(64, n_classes)

        self.out_dim = 64

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        #logits = self.outc(x)
        return x


def memory_usage(tensor_size):
    max_memory = torch.cuda.max_memory_allocated()
    if max_memory == 0:
        usage = 0
    else:
        usage = torch.cuda.memory_allocated() / max_memory
    logger.debug(f'[memory usage] {usage} {tensor_size}')



def conv1x1(in_planes, out_planes, stride=1, groups=1, dilation=1):
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

        self.query_w = conv1x1(in_channels,out_channels)
        self.key_w = conv1x1(in_channels,out_channels)
        self.value_w = conv1x1(in_channels,in_channels//2)
        self.attn_value_w = conv1x1(in_channels//2,in_channels)

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

        return x + self.sigma * attn_value
        

class tAttn(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        out_channels = in_channels//8
        # self.query_w = nn.utils.spectral_norm(conv1x1(in_channels,out_channels))
        # self.key_w = nn.utils.spectral_norm(conv1x1(in_channels,out_channels))
        # self.value_w = nn.utils.spectral_norm(conv1x1(in_channels,in_channels//2))
        # self.attn_value_w = nn.utils.spectral_norm(conv1x1(in_channels//2,in_channels))

        self.query_w =conv1x1(in_channels,out_channels)
        self.key_w = conv1x1(in_channels,out_channels)
        self.value_w = conv1x1(in_channels,in_channels//2)
        self.attn_value_w = conv1x1(in_channels//2,in_channels)

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

        return x + self.sigma * attn_value #, attn.view(B,V,T,T)



class UNet_CA(nn.Module):
    def __init__(self, 
        nclass, 
        nc, 
        globalpool, 
        num_layers,
        patch_size=(256,256),
        slide_window=(256*256,256*256/2)
    ):
        super().__init__()
        self.nclass = nclass
        self.nc = nc
        self.globalpool = globalpool
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.slide_window = slide_window
        # input C_in, W, H, output C_out, W_o, H_o
        num_block = patch_size[0]//4


        self.cnn_repre0 = UNet(nc,nclass)


        self.ca_attn0 = nn.Sequential(
            tAttn(in_channels=self.cnn_repre0.out_dim),
            vAttn(in_channels=self.cnn_repre0.out_dim)
        )
    
        
        if globalpool == 'maxpool':
            self.global_pool = nn.AdaptiveMaxPool2d(1)
        elif globalpool == 'avgpool':
            self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.flatten = nn.Flatten(1)
        self.classifier = self.make_classifier(self.cnn_repre0.out_dim,nclass)

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

    def _pad_resolution(self, seq, byte):
        logger.debug(f'seq {seq.shape} {seq.size(2) % byte} {byte-(seq.size(2)%byte) }')
        if seq.size(2) % byte != 0:
            seq = F.pad(seq,(2,byte-(seq.size(2)%byte)))
        
        return seq

    def _change_resolution(self, seq, byte):
        """
        change the resolution of the input
        """
        batch, nc, seq_len = seq.size()
        seq = self._pad_resolution(seq, byte)
        x = torch.div(torch.sum(seq.reshape(1,-1,byte),dim=2),byte)
        x = x.view(batch, nc, -1)
        return x
    
    def forward(self, x):
        batch = len(x)
        
        outs = []
        # for each pdf file, we divide it into non-overlap patches
        for i in x:
            seq_len = i.size()[0]
            logger.debug('i shape {}'.format(i.size()))
            xi = i.view(1,self.nc,seq_len)
            
            x_resolu_0 = xi.clone()
            x_resolu_1 = self._change_resolution(xi, 2)
            logger.debug(f'x_resolu_0 {x_resolu_0.shape}')
            logger.debug(f'x_resolu_1 {x_resolu_1.shape}')
            if self.slide_window is None:
                c_in_0, ng_0 = self._view_to_grid(x_resolu_0)
                c_in_1, ng_1 = self._view_to_grid(x_resolu_1)
            else:
                c_in_0, ng_0 = self._view_slide_window(x_resolu_0)
                c_in_1, ng_1 = self._view_slide_window(x_resolu_1)
            
            #memory_usage(c_in.size())
            logger.debug(f'c_in_0 size {c_in_0.size()}')
            logger.debug(f'c_in_1 size {c_in_1.size()}')

            # number of windows, channels, height, width
            nw, c, h, w = c_in_0.size()
            c_in_0 = c_in_0.reshape(1,c,h*w, nw)

            nw, c, h, w = c_in_1.size()
            c_in_1 = c_in_1.reshape(1,c,h*w, nw)

            logger.debug(f'c_in_0 size {c_in_0.size()}')
            logger.debug(f'c_in_1 size {c_in_1.size()}')
            
            c_out0 = self.cnn_repre0(c_in_0)

            logger.debug(f'c_out.size {c_out0.size()}')
            
            
            attn_out0 = self.ca_attn0(c_out0)
            
            

            logger.debug(f'attn_out0 {attn_out0.shape}')
            #logger.debug(f'attn_out1 {attn_out1.shape}')

            out0 = self.global_pool(attn_out0)
            #out1 = self.global_pool(attn_out1)
            out0 = self.flatten(out0)
            #out1 = self.flatten(out1)

            logger.debug(f'globalpool {out0.shape}')
            #logger.debug(f'globalpool {out1.shape}')

            #out = torch.cat((out0,out1),1)
            logger.debug(f'out {out0.shape}')
            
            outs.append(out0)
        
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
    i = torch.ones((1,4248001))
    from torch.profiler import profile, ProfilerActivity, record_function

    patch = 64
    pcnn = UNet_CA(2,1,globalpool='maxpool',num_layers=1,patch_size=(patch,patch),slide_window=(patch*patch,patch*patch/2))
    #pcnn = get_msfcncs_twoheadfcn(2,1,'maxpool',num_layers=1,patch_size=(patch,patch),slide_window=(patch*patch,patch*patch/2))
    #pcnn = get_msfcncs_twoheadfcnca(2,1,'maxpool',num_layers=1,patch_size=(patch,patch),slide_window=(patch*patch,patch*patch/2))

    with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True,with_flops=True) as prof:
        with record_function("model_inference"):
            print(pcnn(i).shape)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))