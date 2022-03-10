## Resnet Model

## -------------------
## --- Third-party ---
## -------------------
import torch as t
import torch.nn as nn
from typing import List, Tuple



## --- Functions ---
def conv1x1(inplanes: int, outplanes: int, padding=0, stride=1):
    """1x1 convolution for Bottlenneck (residualblock)"""
    return nn.Conv1d(inplanes, outplanes, kernel_size= 1, stride= stride,
                     padding= padding, bias= False)

def conv1d(inplanes: int, outplanes: int, kernel_size: int =3, stride=1, padding: int = 1, dilation: int = 1):
    """1d convolution with padding
    (N, Cin, L)
    (N, Cout, Lout)
    N: batch size
    C: a number of channels
    L: a length of signal sequence
    """
    return nn.Conv1d(inplanes, outplanes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, bias=True)

## --- Classes ---
class ResBlock(nn.Module):
    def __init__(self, ch_in: int, ch_out: int, kernel_size: int, stride: int):
        super(ResBlock, self).__init__()
        padding = kernel_size//2
        self.conv1 = conv1d(ch_in, ch_out, kernel_size=kernel_size, padding= int(padding),
                            stride=stride)
        self.bn1 = nn.BatchNorm1d(ch_out)
        self.relu1 = nn.ReLU()

        self.conv2 = conv1d(ch_out, ch_out, kernel_size=kernel_size, padding=int(padding))

        ## for the input added
        self.conv1x1 = conv1x1(ch_in, ch_out, stride=stride)

        self.bn2 = nn.BatchNorm1d(ch_out)
        self.relu2 = nn.ReLU()

        self.block = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu1,
            self.conv2
        )

    def forward(self, x):
        input = x
        x = self.block(x)
        x += self.conv1x1(input)
        x = self.bn2(x)
        x = self.relu2(x)
        return x

    def model_name(self):
        return "ResidualBlock"

class ResNet(nn.Module):
    def __init__(self, ch_in: int, ch_out: List, num_classes: int,
                 kernel_size: List, stride: List):
        super(ResNet, self).__init__()
        if len(ch_out) != len(kernel_size) + 1:
            raise ValueError("The size of ch_out should be one more larger than the size of kernel_size")
        if len(stride) != len(kernel_size):
            raise ValueError("The size of stride should be the same as the size of kernel_size")
        if len(ch_out) != len(stride) + 1:
            raise ValueError("The size of ch_out should be one more larger than the size of stride")
        self.ch_out = ch_out
        self.layers = []

        self.conv1 = conv1d(ch_in, ch_out[0], kernel_size=7, stride=2)
        self.bn1 = nn.BatchNorm1d(ch_out[0])
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2)

        for i in range(len(ch_out) - 1):
            self.layers += [ResBlock(ch_in=ch_out[i], ch_out=ch_out[i+1], kernel_size=kernel_size[i], stride=stride[i])]

        self.gap = nn.AdaptiveAvgPool1d(1) ## [N, F, L] --> [N, F, 1]
        self.fc = nn.Linear(ch_out[-1], num_classes) ## require (N, L, C) C is features

        self.block = nn.Sequential(
            *self.layers
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.block(x)

        x = self.gap(x)
        x = x.transpose(1, 2)
        x = self.fc(x)
        x = x.transpose(1, 2)  ## return back [N, F, L]
        return x

    def model_name(self):
        return "ResNet_blocks_{}".format(len(self.ch_out) - 1)