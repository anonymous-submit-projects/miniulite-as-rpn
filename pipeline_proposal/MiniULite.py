import torch
import torch.nn as nn
      

class AdjustChannels(nn.Module):
    """
    Adjusts the number of channels of a tensor:
    -If in_ch < out_ch: repeat channels until reaching out_ch
    -If in_ch > out_ch: keep (out_ch//2) channels and reduce the rest with a conv1x1 to (out_ch -keep)
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        if in_ch > out_ch:
            self.keep = out_ch // 2
            self.reduced_out = out_ch - self.keep
            self.reduce = nn.Conv2d(in_ch - self.keep, self.reduced_out, kernel_size=1)
        else:
            self.reduce = None

    def forward(self, x):
        b, c, h, w = x.shape

        if c == self.out_ch:
            return x

        elif c < self.out_ch:
            # Replaces x.repeat with a more explicit and ONNX-friendly concatenation
            num_repeats = (self.out_ch + c - 1) // c  # Equivalente a ceil(out_ch / c)
            # Creates a list of tensors 'x' and concatenates them into the channel dimension
            x_repeated = torch.cat([x] * num_repeats, dim=1)
            # Slice the resulting tensor to get the exact number of channels
            return x_repeated[:, :self.out_ch]

        else:  # c > out_ch
            part1 = x[:, :self.keep]
            excess = x[:, self.keep:]
            reduced = self.reduce(excess)
            return torch.cat([part1, reduced], dim=1)

class AxialConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, dilation=1, groups=1, bias=True, padding='same'):
        super().__init__()
        self.adjust = AdjustChannels(in_channels, out_channels)
        
        self.groups       = groups
        self.out_channels = out_channels
        if groups == out_channels: #Dw
            self.dw_h   = nn.Conv2d(out_channels, out_channels, kernel_size=(kernel_size, 1), padding=(padding, 0), groups=groups, dilation=dilation, bias=bias)
            self.dw_w   = nn.Conv2d(out_channels, out_channels, kernel_size=(1, kernel_size), padding=(0, padding), groups=groups, dilation=dilation, bias=bias)
        else:    
            self.dw_h   = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(padding, 0), groups=groups, dilation=dilation, bias=bias)
            self.dw_w   = nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size), padding=(0, padding), groups=groups, dilation=dilation, bias=bias)

    def forward(self, x):
        if self.groups == self.out_channels:
            # (If it is DepthWise)
            x = self.adjust(x)
            x = x + self.dw_h(x) + self.dw_w(x)
        else:
            x = self.adjust(x) + self.dw_h(x) + self.dw_w(x)
        return x

class AxialEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        if in_channels == 3:
            groups = 1
        else:
            groups = in_channels
        self.dw   = AxialConv(in_channels, out_channels, kernel_size=kernel_size, padding=2, groups=groups)
        self.bn   = nn.BatchNorm2d(out_channels)
        self.down = nn.MaxPool2d((2,2))
        self.act  = nn.GELU()

    def forward(self, x):
        skip = self.bn(self.dw(x))
        x = self.act(self.down(skip))
        return x, skip

class AxialDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.pw = nn.Conv2d(in_channels,    out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.dw = AxialConv(out_channels,   out_channels, kernel_size=kernel_size, padding=2, groups=out_channels)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(out_channels,  out_channels, kernel_size=1)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.act(self.pw2(self.dw(self.bn(self.pw(x)))))
        return x
    
class BottleNeckBlockV1(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        middle_channels = in_channels//4
        self.pw1 = nn.Conv2d(in_channels, middle_channels, kernel_size=1)

        self.dw1 = nn.Conv2d(middle_channels, middle_channels,  kernel_size=3, dilation=1, padding=1)
        self.dw2 = nn.Conv2d(middle_channels, middle_channels,  kernel_size=3, dilation=2, padding=2)
        self.dw3 = nn.Conv2d(middle_channels, middle_channels,  kernel_size=3, dilation=3, padding=3)

        self.bn = nn.BatchNorm2d(4*middle_channels)
        self.pw2 = nn.Conv2d(4*middle_channels, in_channels, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.pw1(x)
        x = torch.cat([x, self.dw1(x), self.dw2(x), self.dw3(x)], 1)
        x = self.act(self.pw2(self.bn(x)))
        return x
    

class MiniULite(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        layer1, layer2, layer3, bn = 16, 32, 64, 128
        
        self.down = nn.MaxPool2d((4,4))
        self.up = nn.Upsample(scale_factor=4)

        self.e1 = AxialEncoderBlock(in_channels, layer2)
        self.e2 = AxialEncoderBlock(layer2,      layer3)
        self.e3 = AxialEncoderBlock(layer3,      bn)

        self.bn = BottleNeckBlockV1(bn)

        self.dec3 = AxialDecoderBlock(bn*2,      layer3)
        self.dec2 = AxialDecoderBlock(layer3*2,  layer2)
        self.dec1 = AxialDecoderBlock(layer2*2,  layer1)
        self.conv_out = nn.Conv2d(layer1, out_channels, kernel_size=1)

    def forward(self, x):
        x        = self.down(x)    #512
        x, skip1 = self.e1(x)      #128
        x, skip2 = self.e2(x)      #64
        x, skip3 = self.e3(x)      #32

        x = self.bn(x)             #32

        x = self.dec3(x, skip3)    #64
        x = self.dec2(x, skip2)    #128
        x = self.dec1(x, skip1)    #256
        x = self.conv_out(x)       #256
        x = self.up(x)             #512
        return x
    
