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
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        
    
    def batch_norm(self, x):
        if self.is_batch_norm:
            return nn.BatchNorm2d(self.out_size)(x)
        else:
            return x
            
    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.conv2(x)
        x = self.batch_norm(x)
        x = nn.ReLU(inplace=True)(x)
        return x
        
        
class Down(nn.Module):
    ''' UNet decoder part
    
    Arguments:
        - in_size (int): input neurons size (output size of previous layer)
        - out_size (int): output neurons size
        
    Example: 
            layer = Down(4, 8)
    '''
    def __init__(self, in_size, out_size):
        super(Down, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
    
    def forward(self, x):
        x = nn.MaxPool2d(2)(x)
        x = DoubleConv(self.in_size, self.out_size)(x)
        return x
        
        
class Up(nn.Module):
    ''' UNet encoder part
    
    Arguments:
        - in_size (int): input neurons size (output size of previous layer)
        - out_size (int): output neurons size
        
    Example: 
            layer = Up(4, 8)
    '''
    def __init__(self, in_size, out_size):
        super(Up, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_size, out_size)
        
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
        super(Out, self).__init__()
        self.conv = nn.Conv2d(in_size, out_size, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)
        
        
