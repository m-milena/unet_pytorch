import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_size, out_size, batch_norm=False):
        super(DoubleConv, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.batch_norm = batch_norm
        
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        
    def batch_norm(self, x):
        if self.batch_norm:
            return nn.BatchNorm2d(self.out_size)(x)
        else:
            return x
            
    def forward(self, x):
        x = self.conv1(x)
        x = batch_norm(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = batch_norm(x)
        x = nn.ReLU()(x)
        return x
        
        
class Down(nn.Module):
    def __init__(self, in_size, out_size):
        super(Down, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
    
    def forward(self, x):
        x = DoubleConv(self.in_size, self.out_size)(x)
        x = nn.MaxPool2d(2)(x)
        return x
        
class Up(nn.Module):
    def __init__(self, in_size, out_size):
        super(Up, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(self.in_size, out_size)
        
    def forward(self, x1, x2):
        x = self.conv(x1)
        x = self.upsample(x)
        x = torch.cat([x, x2], dim=1)
        return x
        
class Out(nn.Module):
    def __init__(self, in_size, out_size):
        super(Out, self).__init__()
        self.conv = nn.Conv2d(in_size, out_size, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)
        
        
