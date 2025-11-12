# Para este script funcionar primeiro é necessário clonar o repositório original da Attention U-Net utilizada neste trabalho
# git clone https://github.com/LeeJunHyun/Image_Segmentation/
# adicione no sys.path.append o caminho para o repositorio baixado
import torch
from network import *

class MiniAttU_Net_5Layers(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(MiniAttU_Net_5Layers,self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        L1, L2, L3, L4, L5 = 32, 64, 128, 256, 512

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=L1)
        self.Conv2 = conv_block(ch_in=L1,ch_out=L2)
        self.Conv3 = conv_block(ch_in=L2,ch_out=L3)
        self.Conv4 = conv_block(ch_in=L3,ch_out=L4)
        self.Conv5 = conv_block(ch_in=L4,ch_out=L5)

        self.Up5 = up_conv(ch_in=L5,ch_out=L4)
        self.Att5 = Attention_block(F_g=L4,F_l=L4,F_int=L3)
        self.Up_conv5 = conv_block(ch_in=L5, ch_out=L4)

        self.Up4 = up_conv(ch_in=L4,ch_out=L3)
        self.Att4 = Attention_block(F_g=L3,F_l=L3,F_int=L2)
        self.Up_conv4 = conv_block(ch_in=L4, ch_out=L3)

        self.Up3 = up_conv(ch_in=L3,ch_out=L2)
        self.Att3 = Attention_block(F_g=L2,F_l=L2,F_int=L1)
        self.Up_conv3 = conv_block(ch_in=L3, ch_out=L2)

        self.Up2 = up_conv(ch_in=L2,ch_out=L1)
        self.Att2 = Attention_block(F_g=L1,F_l=L1,F_int=32)
        self.Up_conv2 = conv_block(ch_in=L2, ch_out=L1)

        self.Conv_1x1 = nn.Conv2d(L1,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class MiniAttU_Net_3Layers(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(MiniAttU_Net_3Layers,self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        L1, L2, L3, L4, L5 = 32, 64, 128, 256, 512

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=L1)
        self.Conv2 = conv_block(ch_in=L1,ch_out=L2)
        self.Conv3 = conv_block(ch_in=L2,ch_out=L3)
        
        self.Up3 = up_conv(ch_in=L3,ch_out=L2)
        self.Att3 = Attention_block(F_g=L2,F_l=L2,F_int=L1)
        self.Up_conv3 = conv_block(ch_in=L3, ch_out=L2)

        self.Up2 = up_conv(ch_in=L2,ch_out=L1)
        self.Att2 = Attention_block(F_g=L1,F_l=L1,F_int=32)
        self.Up_conv2 = conv_block(ch_in=L2, ch_out=L1)

        self.Conv_1x1 = nn.Conv2d(L1,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x) #64

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        d3 = self.Up3(x3)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class MiniAttU_Net_4Layers(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(MiniAttU_Net_4Layers,self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        L1, L2, L3, L4, L5 = 32, 64, 128, 256, 512

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=L1)
        self.Conv2 = conv_block(ch_in=L1,ch_out=L2)
        self.Conv3 = conv_block(ch_in=L2,ch_out=L3)
        self.Conv4 = conv_block(ch_in=L3,ch_out=L4)
        
        self.Up4 = up_conv(ch_in=L4,ch_out=L3)
        self.Att4 = Attention_block(F_g=L3,F_l=L3,F_int=L2)
        self.Up_conv4 = conv_block(ch_in=L4, ch_out=L3)

        self.Up3 = up_conv(ch_in=L3,ch_out=L2)
        self.Att3 = Attention_block(F_g=L2,F_l=L2,F_int=L1)
        self.Up_conv3 = conv_block(ch_in=L3, ch_out=L2)

        self.Up2 = up_conv(ch_in=L2,ch_out=L1)
        self.Att2 = Attention_block(F_g=L1,F_l=L1,F_int=32)
        self.Up_conv2 = conv_block(ch_in=L2, ch_out=L1)

        self.Conv_1x1 = nn.Conv2d(L1,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        d4 = self.Up4(x4)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1
