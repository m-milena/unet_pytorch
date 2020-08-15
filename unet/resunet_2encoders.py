import torch
import torch.nn as nn
import unet_parts as net
import numpy as np
from torchsummary import summary


class ResUNet_2encoders(nn.Module):
    ''' UNet architecture with residual blocks
    
    Arguments:
        - in_size (int): input image channels number (e.g. RGB image has 3 channels)
        - out_size (int): output image channels number
        - n_size (int): number of neurons in first layer (default: 64)
        
    Example: 
            net = ResUNet(3, 1, n_size=16)
    '''
    def __init__(self, in_size, out_size, n_size=64):
        super(ResUNet_2encoders, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.n_size = n_size
        
        # Encoder 1
        self.convD11 = net.ResBlock(in_size, n_size, batch_norm=True)
        self.convD12 = net.Down(n_size, n_size*2, residual=True, batch_norm=True)
        self.convD13 = net.Down(n_size*2, n_size*4, residual=True, batch_norm=True)
        self.convD14 = net.Down(n_size*4, n_size*8, residual=True, batch_norm=True)

        #self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Encoder 2
        self.convD21 = net.ResBlock(in_size, n_size, batch_norm=True)
        self.convD22 = net.Down(n_size, n_size*2, residual=True, batch_norm=True)
        self.convD23 = net.Down(n_size*2, n_size*4, residual=True, batch_norm=True)
        self.convD24 = net.Down(n_size*4, n_size*8, residual=True, batch_norm=True)

        #self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.convM = net.Down(n_size*8, n_size*16, residual=True, batch_norm=True)
        
        self.convU4 = net.Up(n_size*16 + n_size*8, n_size*8, residual=True, batch_norm=True)
        self.convU3 = net.Up(n_size*8 + n_size*4, n_size*4, residual=True, batch_norm=True)
        self.convU2 = net.Up(n_size*4 + n_size*2, n_size*2, residual=True, batch_norm=True)
        self.convU1 = net.Up(n_size*2 + n_size, n_size, residual=True, batch_norm=True)
        
        self.output = net.ConvOut(n_size, out_size)
    
    
    def forward(self, x1, x2):
        # Down encoder 1
        x11 = self.convD11(x1)
        x12 = self.convD12(x11)
        x13 = self.convD13(x12)
        x14 = self.convD14(x13)

        # Down encoder 2
        x21 = self.convD21(x2)
        x22 = self.convD22(x21)
        x23 = self.convD23(x22)
        x24 = self.convD24(x23)

        x = torch.cat([x14, x24], dim=0)
    

        # Middle
        x = self.convM(x)

        # Up
        x4 = torch.cat([x14, x24], dim=0)
        x = self.convU4(x, x4)
        x3 = torch.cat([x13, x23], dim=0)
        x = self.convU3(x, x3)
        x2 = torch.cat([x12, x22], dim=0)
        x = self.convU2(x, x2)
        x1 = torch.cat([x11, x21], dim=0)
        x = self.convU1(x, x1)

        return self.output(x)
        
        
if __name__ == '__main__':
    net = ResUNet_2encoders(1, 1, n_size=64)
    x = torch.rand(1, 1, 128, 128)
    y = net(x, x)
    print(y[0].size())
