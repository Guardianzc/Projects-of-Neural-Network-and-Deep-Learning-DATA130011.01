import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(ResidualBlock, self).__init__()
        self.pooling = True
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1)
        )
        self.shortcut = nn.Sequential()
        if stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride, padding = 0),
                nn.BatchNorm2d(out_channels))
                   
    def forward(self, x):
        out = self.block(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet18_t(nn.Module):
    def __init__(self, ResidualBlock, use_dropout, classes = 10):
        super(ResNet18_t, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = False)
        )
        self.block1 = nn.Sequential(
            ResidualBlock(in_channels = 64, out_channels = 64),
            ResidualBlock(in_channels = 64, out_channels = 64)
        )
        self.block2 = nn.Sequential(
            ResidualBlock(in_channels = 64, out_channels = 128, stride = 2),
            ResidualBlock(in_channels = 128, out_channels = 128)
        )
        self.block3 = nn.Sequential(
            ResidualBlock(in_channels = 128, out_channels = 256, stride = 2),
            ResidualBlock(in_channels = 256, out_channels = 256)
        )
        self.block4 = nn.Sequential(
            ResidualBlock(in_channels = 256, out_channels = 512, stride = 2),
            ResidualBlock(in_channels = 512, out_channels = 512)
        )
        self.linear = nn.Sequential( nn.Linear(512, classes))
        if use_dropout:
            self.linear = nn.Sequential( nn.Linear(512, classes), nn.Dropout(0.5))

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.block1(x1)
        x3 = self.block2(x2)
        x4 = self.block3(x3)
        x5 = self.block4(x4)
        x_avg = F.avg_pool2d(x5, 4)
        x_linear = x_avg.view(x_avg.size(0), -1)
        out = self.linear(x_linear)
        return out

def ResNet_Transform(use_dropout = False):
    return ResNet18_t(ResidualBlock, use_dropout)


class ResNet18_pooling(nn.Module):
    def __init__(self, ResidualBlock, use_dropout, classes = 10):
        super(ResNet18_pooling, self).__init__()
        self.pooling1 = nn.MaxPool2d(kernel_size = 8, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        self.pooling2 = nn.MaxPool2d(kernel_size = 4, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        self.pooling3 = nn.MaxPool2d(kernel_size = 2, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = False)
        )
        self.block1 = nn.Sequential(
            ResidualBlock(in_channels = 64, out_channels = 64),
            ResidualBlock(in_channels = 64, out_channels = 64)
        )
        self.block2 = nn.Sequential(
            ResidualBlock(in_channels = 64, out_channels = 128, stride = 2),
            ResidualBlock(in_channels = 128, out_channels = 128)
        )
        self.block3 = nn.Sequential(
            ResidualBlock(in_channels = 128, out_channels = 256, stride = 2),
            ResidualBlock(in_channels = 256, out_channels = 256)
        )
        self.block4 = nn.Sequential(
            ResidualBlock(in_channels = 256, out_channels = 512, stride = 2),
            ResidualBlock(in_channels = 512, out_channels = 512)
        )
        self.linear512 = nn.Sequential( nn.Linear(512, classes))
        self.linear64 = nn.Sequential( nn.Linear(64, classes))
        self.linear128 = nn.Sequential( nn.Linear(128, classes))
        self.linear256 = nn.Sequential( nn.Linear(256, classes))
        self.linear50 = nn.Sequential( nn.Linear(50, classes))
        if use_dropout:
            self.linear512 = nn.Sequential( nn.Linear(512, classes), nn.Dropout(0.5))
            self.linear64 = nn.Sequential( nn.Linear(64, classes), nn.Dropout(0.5))
            self.linear128 = nn.Sequential( nn.Linear(128, classes), nn.Dropout(0.5))
            self.linear256 = nn.Sequential( nn.Linear(256, classes), nn.Dropout(0.5))
            self.linear50 = nn.Sequential( nn.Linear(50, classes), nn.Dropout(0.5))

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.block1(x1)
        x3 = self.block2(x2)
        x4 = self.block3(x3)
        x5 = self.block4(x4)
        x5_avg = F.avg_pool2d(x5, 4)
        x5_linear = x5_avg.view(x5_avg.size(0), -1)
        
        x1_pool = self.pooling1(x1)
        x1_avg = F.avg_pool2d(x1_pool, 4)
        x1_linear = x1_avg.view(x1_avg.size(0), -1)

        x2_pool = self.pooling1(x2)
        x2_avg = F.avg_pool2d(x2_pool, 4)
        x2_linear = x2_avg.view(x2_avg.size(0), -1)  

        x3_pool = self.pooling2(x3)
        x3_avg = F.avg_pool2d(x3_pool, 4)
        x3_linear = x3_avg.view(x3_avg.size(0), -1)

        x4_pool = self.pooling3(x4)
        x4_avg = F.avg_pool2d(x4_pool, 4)
        x4_linear = x4_avg.view(x4_avg.size(0), -1)
        
        out5 = self.linear512(x5_linear)
        out1 = self.linear64(x1_linear)
        out2 = self.linear64(x2_linear)
        out3 = self.linear128(x3_linear)
        out4 = self.linear256(x4_linear)
        out_all = torch.cat((out1, out2, out3, out4, out5), 1)
        out = self.linear50(out_all)

        
        return out

def ResNet_pooling(use_dropout = False):
    return ResNet18_pooling(ResidualBlock, use_dropout)
    
