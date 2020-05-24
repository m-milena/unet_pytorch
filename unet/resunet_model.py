import torch
import torch.nn as nn
import unet_parts as net
from torchsummary import summary

class ResUNet(nn.Module):
    ''' UNet architecture with residual blocks
    
    Arguments:
        - in_size (int): input image channels number (e.g. RGB image has 3 channels)
        - out_size (int): output image channels number
        - n_size (int): number of neurons in first layer (default: 64)
        
    Example: 
            net = ResUNet(3, 1, n_size=16)
    '''
    def __init__(self, in_size, out_size, n_size=64):
        super(ResUNet, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.n_size = n_size
        
        self.convD1 = net.ResBlock(in_size, n_size)
        self.convD2 = net.Down(n_size, n_size*2, residual=True)
        self.convD3 = net.Down(n_size*2, n_size*4, residual=True)
        self.convD4 = net.Down(n_size*4, n_size*8, residual=True)
        
        self.convM = net.Down(n_size*8, n_size*16, residual=True)
        
        self.convU4 = net.Up(n_size*16 + n_size*8, n_size*8, residual=True)
        self.convU3 = net.Up(n_size*8 + n_size*4, n_size*4, residual=True)
        self.convU2 = net.Up(n_size*4 + n_size*2, n_size*2, residual=True)
        self.convU1 = net.Up(n_size*2 + n_size, n_size, residual=True)
        
        self.output = net.ConvOut(n_size, out_size)
    
    
    def forward(self, x):
        # Down
        print(x.shape)
        x1 = self.convD1(x)
        print(x1.shape)
        x2 = self.convD2(x1)
        print(x2.shape)
        x3 = self.convD3(x2)
        print(x3.shape)
        x4 = self.convD4(x3)
        print(x4.shape)
        # Middle
        x = self.convM(x4)
        # Up
        x = self.convU4(x, x4)
        x = self.convU3(x, x3)
        x = self.convU2(x, x2)
        x = self.convU1(x, x1)

        return self.output(x)
        
        
if __name__ == '__main__':
    net = ResUNet(1, 1, n_size=64)
    summary(net, (1, 256, 256))
