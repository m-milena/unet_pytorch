import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    ''' Double convolution layer for UNet architecture
    
    Arguments:
        - in_size (int): input neurons size (output size of previous layer)
        - out_size (int): output neurons size
        - batch_norm (bool): option to make batch normalization (default: False)
        
    Example: 
            layer = DoubleConv(4, 8, batch_norm=True)
    '''
    def __init__(self, in_size, out_size, batch_norm=False):
        super(DoubleConv, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.is_batch_norm = batch_norm
        
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.batch_norm_1 = nn.BatchNorm2d(out_size)

        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.batch_norm_2 = nn.BatchNorm2d(out_size)

            
    def forward(self, x):
        x = self.conv1(x)
        if self.is_batch_norm:
            x = self.batch_norm_1(x)
        x = nn.ReLU()(x)

        x = self.conv2(x)
        if self.is_batch_norm:
            x = self.batch_norm_2(x)
        x = nn.ReLU()(x)
        return x
        
        
class ResBlock(nn.Module):
    ''' Residual Block for UNet
    
    Arguments:
        - in_size (int): input neurons size (output size of previous layer)
        - out_size (int): output neurons size
        - batch_norm (bool): option to make batch normalization (default: False)
        
    Example: 
            block = ResBlock(16, 64, batch_norm=True)
    '''
    def __init__(self, in_size, out_size, batch_norm=False):
        super(ResBlock, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.is_batch_norm = batch_norm

        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.batch_norm_1 = nn.BatchNorm2d(out_size)

        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.batch_norm_2 = nn.BatchNorm2d(out_size)

        self.conv1x1 = nn.Conv2d(in_size, out_size, kernel_size=1, padding=0)
        self.batch_norm_1x1 = nn.BatchNorm2d(out_size)

        
    def forward(self, x):
        x1 = x

        x2 = self.conv1(x)
        if self.is_batch_norm:
            x2 = self.batch_norm_1(x2)
        x2 = nn.ReLU()(x2)

        x2 = self.conv2(x2)
        if self.is_batch_norm:
            x2 = self.batch_norm_2(x2)
        x2 = nn.ReLU()(x2)

        if self.in_size != self.out_size:
            x1 = self.conv1x1(x1)
            if self.is_batch_norm: x1 = self.batch_norm_1x1(x1)
            x1 = nn.ReLU()(x1)

        x2 += x1
        x = nn.ReLU()(x2)
        return x
        
        
class Down(nn.Module):
    ''' UNet decoder part
    
    Arguments:
        - in_size (int): input neurons size (output size of previous layer)
        - out_size (int): output neurons size
        - batch_norm (bool): option to make batch normalization (default: False)
        - residual (bool): option to make residual block
        
    Example: 
            layer = Down(4, 8)
    '''
    def __init__(self, in_size, out_size, batch_norm=False, residual=False):
        super(Down, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.residual = residual
        self.is_batch_norm = batch_norm
        
        if residual:
            self.conv = ResBlock(in_size, out_size, batch_norm)
        else:
            self.conv = DoubleConv(in_size, out_size, batch_norm)
    
    def forward(self, x):
        x = nn.MaxPool2d(2)(x)
        x = self.conv(x)
        return x
        
        
class Up(nn.Module):
    ''' UNet encoder part
    
    Arguments:
        - in_size (int): input neurons size (output size of previous layer)
        - out_size (int): output neurons size
        - batch_norm (bool): option to make batch normalization (default: False)
        - residual (bool): option to make residual block
        
    Example: 
            layer = Up(4, 8)
    '''
    def __init__(self, in_size, out_size, batch_norm=False, residual=False):
        super(Up, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.is_batch_norm = batch_norm
        self.residual = residual
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        if residual:
            self.conv = ResBlock(in_size, out_size, batch_norm)
        else:
            self.conv = DoubleConv(in_size, out_size, batch_norm)
        
    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)
        
        
class ConvOut(nn.Module):
    ''' Convolutional output
    
    Arguments:
        - in_size (int): input filter size (previous conv layer out filter)
        - out_size (int): output filter size
        
    Example: 
            layer = ConvOut(16, 2)
    '''
    def __init__(self, in_size, out_size):
        super(ConvOut, self).__init__()
        self.conv = nn.Conv2d(in_size, out_size, kernel_size=1)
    
    def forward(self, x):
        self.conv(x)
        return nn.Sigmoid()(x)
        
