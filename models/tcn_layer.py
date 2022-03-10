## TCN layer ( using Residual Block )
"""
tcn is a Temporal Convolutional Network, using residual blocks and fully convolutional layers
refer to: https://github.com/samyakmshah/pytorchTCN
paper: https://arxiv.org/pdf/1803.01271.pdf
"""
## -------------------
## --- Third-party ---
## -------------------
import torch as t
import torch.nn as nn
from torch.nn.utils import weight_norm
from typing import List, Tuple
from collections import OrderedDict

## --- Functions ---
def conv1d(inplanes: int, outplanes: int, kernel_size: int =3, stride=1, padding: int = 0, dilation: int = 1):
    """1d convolution with padding
    (N, Cin, L)
    (N, Cout, Lout)
    N: batch size
    C: a number of channels
    L: a length of signal sequence
    """
    return nn.Conv1d(inplanes, outplanes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, bias=True)

def conv1x1(inplanes: int, outplanes: int, padding=0, stride=1):
    """1x1 convolution for Bottlenneck (residualblock)"""
    return nn.Conv1d(inplanes, outplanes, kernel_size= 1, stride= stride,
                     padding= padding, bias= False)

## --- Classes ---
class ResidualBlock(nn.Module): ## (K, d) --> kernel size, dilation
    def __init__(self, ch_in: int, ch_out: int, kernel_size: int, stride: int =1,
                 left_padding: int = 0,
                 dilation: int = 1,
                 dropout= 0.2):
        super(ResidualBlock, self).__init__()
        self.zero_pad1 = nn.ZeroPad2d((left_padding, 0, 0, 0))
        self.conv1 = weight_norm(conv1d(ch_in, ch_out, kernel_size= kernel_size, stride= stride, padding= 0, dilation= dilation))
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.zero_pad2 = nn.ZeroPad2d((left_padding, 0, 0, 0))
        self.conv2 = weight_norm(conv1d(ch_out, ch_out, kernel_size= kernel_size, stride= stride, padding= 0, dilation= dilation))
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.sequential = nn.Sequential(
            self.zero_pad1,
            self.conv1,
            self.relu1,
            self.dropout1,
            self.zero_pad2,
            self.conv2,
            self.relu2,
            self.dropout2
        )

        # 1x1 conv to match the shapes (channel dimension)
        self.conv1x1 = conv1x1(ch_in, ch_out) if ch_in != ch_out else None
        self.relu = nn.ReLU()
        # self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.conv1x1 is not None:
            self.conv1x1.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.sequential(x)
        res = x if self.conv1x1 is None else self.conv1x1(x)
        out += res
        return self.relu(out)

class TCN_layer(nn.Module):
    def __init__(self, ch_in, ch_out: List = [64, 64, 128], kernel_size=2, stride: int = 1, dilation: List = [1, 2, 4], dropout= 0.2):
        super(TCN_layer, self).__init__()
        """ensure ch_out has the same size as dilation"""
        self.layers = []
        for i in range(len(ch_out)):
            ch_in = ch_in if i == 0 else ch_out[i-1]
            self.layers += [ResidualBlock(ch_in, ch_out[i], kernel_size= kernel_size, stride= stride,
                                          dilation= dilation[i], left_padding=(kernel_size - 1)*dilation[i], dropout= dropout)]

        self.tcn = nn.Sequential(*self.layers) ## to unpack the array

    def forward(self, x):
        return self.tcn(x)


