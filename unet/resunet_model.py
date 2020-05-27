import torch
import torch.nn as nn
import unet.unet_parts as net
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
        
        self.convD1 = net.ResBlock(in_size, n_size, batch_norm=True)
        self.convD2 = net.Down(n_size, n_size*2, residual=True, batch_norm=True)
        self.convD3 = net.Down(n_size*2, n_size*4, residual=True, batch_norm=True)
        self.convD4 = net.Down(n_size*4, n_size*8, residual=True, batch_norm=True)
        
        self.convM = net.Down(n_size*8, n_size*16, residual=True, batch_norm=True)
        
        self.convU4 = net.Up(n_size*16 + n_size*8, n_size*8, residual=True, batch_norm=True)
        self.convU3 = net.Up(n_size*8 + n_size*4, n_size*4, residual=True, batch_norm=True)
        self.convU2 = net.Up(n_size*4 + n_size*2, n_size*2, residual=True, batch_norm=True)
        self.convU1 = net.Up(n_size*2 + n_size, n_size, residual=True, batch_norm=True)
        
        self.output = net.ConvOut(n_size, out_size)
    
    
    def forward(self, x):
        # Down
        x1 = self.convD1(x)
        x2 = self.convD2(x1)
        x3 = self.convD3(x2)
        x4 = self.convD4(x3)
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
