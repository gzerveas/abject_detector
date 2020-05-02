from __future__ import absolute_import

import torch
import torch.nn as nn
import math

from .DCNv2.dcn_v2 import DCN
from torch import nn

BN_MOMENTUM = 0.1


def fill_up_weights(layer):
    """Custom function for initializing upsampling layers"""
    w = layer.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :] 

def fill_fc_weights(layers):
    """Custom function for initializing output layers"""
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            # torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class ResBlock(nn.Module):
    """Basic ResNet block"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        
        if stride != 1 or in_channels != out_channels: # transform input to match residual dimensions
            self.downsample = nn.Sequential(
                            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                            nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM))
        else:
            self.downsample = None
        

    def forward(self, x):
        if self.downsample is not None:
            inp = self.downsample(x)
        else:
            inp = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += inp
        out = self.relu(out)

        return out


class TranspConvBlock(nn.Module):
    """Transposed convolution block. Used for upsampling.
    Has 2 components: 1) Makes use of deformable convolutional layer. """
    def __init__(self, in_channels, out_channels, upsample_factor=2, bias=False):
        super(TranspConvBlock, self).__init__()

        self.fc = DCN(in_channels, out_channels, kernel_size=(3,3), stride=1, padding=1, dilation=1, deformable_groups=1)
        # fc = nn.Conv2d(self.inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        # fill_fc_weights(fc)
        up = nn.ConvTranspose2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=4,
                stride=upsample_factor,
                padding=1,
                output_padding=0,
                bias=bias)
        fill_up_weights(up)

        self.bn1 = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)


        def forward(self, x):
            
            out = self.fc(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.up(out)
            out = self.bn2(out)
            out = self.relu(out)

            return out


class DCN_Detector(nn.Module):

    def __init__(self, layers, heads, head_conv):
        self.in_channels = 64  # number of output channels produced by the entry block
        self.heads = heads
        self.deconv_with_bias = False

        super(DCN_Detector, self).__init__()
        
        # Entry block
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks
        self.block1 = ResBlock(self.in_channels, 64, stride=1)
        self.block2 = ResBlock(64, 128, stride=2)
        self.block3 = ResBlock(128, 256, stride=2)

        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.upsample1 = TranspConvBlock(256, )
        self.deconv_layers = self._make_deconv_layer(3, [256, 128, 64], [4, 4, 4])

        # Add one output layer per target
        for head in self.heads:
            classes = self.heads[head]
            out_layer = nn.Conv2d(64, classes, kernel_size=1, stride=1, padding=0, bias=True)
            if 'hm' in head:
                out_layer.bias.data.fill_(-2.19)  # magic value for heatmap background
            else:
                fill_fc_weights(out_layer)
            self.__setattr__(head, out_layer)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.deconv_layers(x)
        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(x)
        return [ret]


