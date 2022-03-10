"""
Conv1d Block and Gap softmax block
TD conv1d Block (in future)
"""
## -------------------
## --- Third-party ---
## -------------------
import functools
import operator
import torch as t
import torch.nn as nn
from torch.nn.utils import weight_norm
from typing import List, Tuple

## --- Functions ---
def conv1d(inplanes: int, outplanes: int, kernel_size: int =3, stride=1, padding: Tuple = (0, 0)):
    """1d convolution with padding
    (N, Cin, L)
    (N, Cout, Lout)
    N: batch size
    C: a number of channels
    L: a length of signal sequence
    """
    return nn.Conv1d(inplanes, outplanes, kernel_size=kernel_size, stride=stride,
                     padding=padding, bias=True)

## --- Classes ---
class LastStepModule(nn.Module):
    def __init__(self):
        super(LastStepModule, self).__init__()
    def forward(self, x):
        return x[:, :, -1] ## return last step of time

## --- Classes ---
class Conv1d_block(nn.Module):
    def __init__(self, ch_in: int, ch_out: int, kernel_size: int, stride: int = 1, dropout: float = 0.2,
                 padding: Tuple[int] = 0,
                 name: str = 'ConvBlock', use_dropout: bool = True):
        """
        A Convolutional block with Conv1D, Dropout, BatchNorm and ReLu activation

        Parameters
        ----------
        ch_in : Integer, the dimensionality of the input space
                (i.e. the number of input filters in the convolution).
        ch_out: Integer, the dimensionality of the output space
                (i.e. the number of output filters in the convolution).
        kernel_size: An integer or tuple/list of a single integer, specifying the length of the 1D convolution window.
        stride: An integer or tuple/list of a single integer, specifying the stride length of the convolution.
                Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1
        dropout: Float between 0 and 1. Fraction of the input units to drop.
        padding: Shape Tuplee
        name: str for printing
        """
        super(Conv1d_block, self).__init__()
        self.name = name
        self.conv1 = conv1d(ch_in, ch_out, kernel_size= kernel_size, stride= stride, padding= padding)
        self.dropout = nn.Dropout(dropout) if use_dropout else None
        self.batchnorm = nn.BatchNorm1d(num_features=ch_out)
        self.relu = nn.ReLU()

        if use_dropout:
            self.block = nn.Sequential(
                self.conv1,
                self.dropout,
                self.batchnorm,
                self.relu
            )
        else:
            self.block = nn.Sequential(
                self.conv1,
                self.batchnorm,
                self.relu
            )
    def forward(self, x):
        return self.block(x)

class Gmp_softmax_block(nn.Module):
    def __init__(self, ch_in: int, ch_out: int, dropout: float = 0.2,
                 name: str = 'Gap_softmax_block',
                 use_full: bool = True,
                 use_fc: bool = True,
                 use_pooling: bool = True,
                 input_dim: tuple = (1, 1, 150),
                 num_classes: int = 2):
        """
        Final block of Fully Convolutional Network
        Including: GlobalAveragePooling (GAP) followed by Softmax  (LogSoftmax, if CrossEntropyLoss not used)

        Parameters
        ----------
        ch_in : Integer, the dimensionality of the input space
                (i.e. the number of input filters in the convolution).
        ch_out: Integer, the dimensionality of the output space
                (i.e. the number of output filters in the convolution).
        dropout: Float between 0 and 1. Fraction of the input units to drop.
        use_full (bool) : True, use Conv1d 1x1, dropout and batchnorm
                          False, only AdaptiveMaxPool1d
        use_fc (bool) : True, use Fully Connected layer as the last layer of model
        use_pooling (bool): True, use the Global Maxpooling layer,
                            False, not use it
        name: str for printing
        input_dim (tuple) : the dimensions of a input sample
        num_classes (int) : the number of classes
                            default: 2 (binary classification)
        """
        super(Gmp_softmax_block, self).__init__()
        self.name = name

        self.conv1 = nn.Conv1d(ch_in, ch_out, kernel_size= 1, stride= 1, padding= 0)
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.zeros_(self.conv1.bias)

        self.dropout = nn.Dropout(dropout)
        self.batchnorm = nn.BatchNorm1d(num_features=ch_out)
        self.relu = nn.ReLU()

        self.use_pooling = use_pooling
        if use_pooling:
            self.gap = nn.AdaptiveMaxPool1d(1) ## (N, C, L) --> (N, C, 1)

        self.use_fc = use_fc
        if use_fc:
            # self.flatten = nn.Flatten(start_dim=1)
            # num_features_before_fcnn = functools.reduce(operator.mul,
            #                                             list(self.conv1(t.rand(input_dim)).shape))
            ## require (N, L, C) C is features
            ## or (N, C) C is features after flatten
            self.fc = nn.Linear(ch_out,
                                num_classes)


        if use_full and use_pooling:
            self.block = nn.Sequential(
                self.conv1,
                self.dropout,
                self.batchnorm,
                self.gap
            )
        elif not use_full and use_pooling:
            self.block = nn.Sequential(
                self.gap
            )
        elif use_full and not use_pooling:
            self.block = nn.Sequential(
                self.conv1,
                self.dropout,
                self.batchnorm
                # self.relu
            )

    def forward(self, x):
        if self.use_fc and not self.use_pooling:
            x = self.block(x)
            x = x[:, :, -1]             ## return [N, C]
            # x = self.flatten(x)
            # x = x.transpose(1, 2)        ## return [N, L, C]
            x = self.fc(x)              ## return Now [N, num_cls]
            x = x.unsqueeze(-1)         ## return Now [N,num_cls,1]
        elif not self.use_fc and self.use_pooling:
            x = self.block(x)            ## without FC
        elif not self.use_fc and not self.use_pooling:
            x = self.block(x)
            x = x[:, :, -1]              ## return the last step [N, num_cls]
            x = x.unsqueeze(-1)          ## return Now [N,num_cls,1]
        return x


