## FCN and TCN model

## -------------------
## --- Third-party ---
## -------------------
import sys
sys.path.append('..')
import torch as t
import torch.nn as nn
import numpy as np
from typing import List
import functools
import operator

## -----------
## --- own ---
## -----------
from models.basic_blocks import Conv1d_block, Gmp_softmax_block
from models.tcn_layer import TCN_layer
from models.tcn import TemporalConvNet


class TCN(nn.Module):
    """For one Matrix including all input features [acc, gyr, mag, mic]
        TCN is designed to produce the output for the Classification ( with the help from Global Maxpooling)
        -> for windowed labeling
    """
    def __init__(self, ch_in: int, ch_out: List = [64, 128, 128, 64],
                 kernel_size: List = [2, 2, 2, 2],
                 dropout_rate: float = 0.2,
                 use_full: bool = True,
                 use_fc: bool = True,
                 use_pooling: bool = True,
                 num_classes: int = 2,
                 input_dim: tuple = (1, 1, 150)):
        """
        Parameters
        ----------
        ch_in (int) : Input Channel (Sensor Features)
        ch_out (List) : Channel Size for each layer
        kernel_size (List): Kernel size for each convolutional layer
                            Should have the same length as ch_out
        dropout_rate (float) : dropout rate for the TCN (Temporal Convnet)
        use_full (bool) : True, use Conv1d 1x1, dropout and batchnorm in Gmp_softmax_block
                          False, only AdaptiveMaxPool1d in Gmp_softmax_block
        use_fc (bool) : True, use Fully Connected layer as the last layer of model
        use_pooling (bool) : True, use Global Max pooling in Softmax Class
                            False, not use it
        num_classes (int) : the number of classes
        input_dim (tuple): the dimensions of a input sample
                            default: (1, 1, 150)
        """
        super(TCN, self).__init__()
        self.layers = []

        if use_fc:
            channels4tcn = ch_out[:-1]
            ch_in4gap_softmax = channels4tcn[-1]
            ch_out4gap_softmax = ch_out[-1]
        else:
            channels4tcn = ch_out
            ch_in4gap_softmax = ch_out[-1]
            ch_out4gap_softmax = num_classes
        ## TCN
        self.tcn_layers = TemporalConvNet(num_inputs=ch_in,
                                          num_channels=channels4tcn,
                                          kernel_size=kernel_size,
                                          dropout=dropout_rate)

        ## get the feature dim from tcn_layers
        features_dim = self.tcn_layers(t.rand(input_dim)).shape

        ## Global Map Pooling
        self.gap_softmax = Gmp_softmax_block(ch_in=ch_in4gap_softmax,
                                             ch_out=ch_out4gap_softmax,
                                             dropout=0,
                                             name="Final_Softmax",
                                             use_full=use_full,
                                             use_fc=use_fc,
                                             use_pooling=use_pooling,
                                             input_dim=features_dim,
                                             num_classes=num_classes)
        ## sequential
        self.layers += [self.tcn_layers]
        self.layers += [self.gap_softmax]
        self.tcn = nn.Sequential(*self.layers)

        # self.log_softmax = nn.LogSoftmax(dim=1)  ## dim 1 for the feature length
        self.device = t.device("cuda" if t.cuda.is_available() else "cpu")

    def forward(self, x):
        x = self.tcn(x)
        ## we use nn.CrossEntropyLoss
        # if not self.use_gradcam:
        #     x = self.log_softmax(x)
        return x

    def model_name(self):
        return "TCN"


class TCN_dense(nn.Module):
    """For one Matrix including all input features [acc, gyr, mag, mic]
        TCN_dense is designed to produce the output as the same size as input (The length t)
        -> Useful for Densely Labeling
    """
    def __init__(self, ch_in: int, ch_out: List = [64, 128, 128, 64],
                 kernel_size: List = [2, 2, 2, 2],
                 dropout_rate: float = 0.2,
                 num_classes: int = 2):
        """
        Parameters
        ----------
        ch_in (int) : Input Channel (Sensor Features)
        ch_out
        kernel_size
        dilation
        dropout_rate (float) : dropout rate for the Gap Softmax Block
        use_full (bool) : True, use Conv1d 1x1, dropout and batchnorm in Gmp_softmax_block
                          False, only AdaptiveMaxPool1d in Gmp_softmax_block
        use_fc (bool) : True, use Fully Connected layer as the last layer of model
        num_classes (int) : the number of classes
        """
        super(TCN_dense, self).__init__()
        self.layers = []

        ## TCN
        self.tcn_layers = TemporalConvNet(num_inputs=ch_in,
                                          num_channels=ch_out,
                                          kernel_size=kernel_size,
                                          dropout=dropout_rate)
        self.seg_classifier = nn.Sequential(
            # nn.AvgPool2d(kernel_size=(1, self.epoch_length * self.sampling_frequency)),
            # nn.Flatten(start_dim=2),
            # nn.ConstantPad1d(padding=(self.padding, self.padding), value=0),
            nn.Conv1d(in_channels=ch_out[-1], out_channels=num_classes, kernel_size=1, stride=1, padding=0),
            # nn.Softmax(dim=1),
        )
        nn.init.xavier_uniform_(self.seg_classifier[0].weight)
        nn.init.zeros_(self.seg_classifier[0].bias)
        ## sequential
        self.layers += [self.tcn_layers]
        self.layers += [self.seg_classifier]
        self.tcn = nn.Sequential(*self.layers)

        # self.log_softmax = nn.LogSoftmax(dim=1)  ## dim 1 for the feature length
        self.device = t.device("cuda" if t.cuda.is_available() else "cpu")

    def forward(self, x):
        x = self.tcn(x)
        ## we use nn.CrossEntropyLoss
        # if not self.use_gradcam:
        #     x = self.log_softmax(x)
        return x

    def model_name(self):
        return "TCN_dense"

class TCN_4base(nn.Module):
    """For mag and imu, with aud"""
    def __init__(self, model1: nn.Module, model2: nn.Module, model3: nn.Module, model4: nn.Module, dropout_rate: float, num_classes: int, use_cam: bool = True):
        super(TCN_4base, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.model4 = model4
        self.convblock1 = Conv1d_block(ch_in=10, ch_out=64, kernel_size=1, stride=1,
                                       dropout=dropout_rate, name='ConvBlock_1')
        self.convblock2 = Conv1d_block(ch_in=64, ch_out=64, kernel_size=1, stride=1,
                                       dropout=dropout_rate, name='ConvBlock_2')
        self.gap_softmax = Gmp_softmax_block(ch_in=64, ch_out=num_classes, dropout=dropout_rate, use_cam=use_cam)
        self.log_softmax = nn.LogSoftmax(dim=1)  ## dim 1 for the feature length
        self.device = t.device('cuda' if t.cuda.is_available() else 'cpu')

    def forward(self, x1, x2, x3, x4):
        x1 = self.model1(x1)
        x2 = self.model2(x2)
        x3 = self.model1(x3)
        x4 = self.model2(x4)
        ## zero padding to the same length in each sample
        max_len = np.max((x1.shape[2], x2.shape[2], x3.shape[2], x4.shape[2]))
        x1_pad = t.zeros((x1.shape[0], x1.shape[1], max_len))
        x1_pad[:, :, :x1.shape[2]] = x1
        x2_pad = t.zeros((x2.shape[0], x2.shape[1], max_len))
        x2_pad[:, :, :x2.shape[2]] = x2
        x3_pad = t.zeros((x3.shape[0], x3.shape[1], max_len))
        x3_pad[:, :, :x3.shape[2]] = x3
        x4_pad = t.zeros((x4.shape[0], x4.shape[1], max_len))
        x4_pad[:, :, :x4.shape[2]] = x4
        # concat inputs
        x = t.cat((x1_pad, x2_pad, x3_pad, x4_pad), dim=1).to(self.device)
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.gap_softmax(x)
        # x = self.log_softmax(x)
        return x


class TCN_3base(nn.Module):
    """For mag, imu without aud"""
    def __init__(self, model1: nn.Module, model2: nn.Module, model3: nn.Module, dropout_rate: float, num_classes: int, use_cam: bool = True):
        super(TCN_3base, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.convblock1 = Conv1d_block(ch_in=64*3, ch_out=64, kernel_size=1, stride=1,
                                       dropout=dropout_rate, name='ConvBlock_1')
        self.convblock2 = Conv1d_block(ch_in=64, ch_out=64, kernel_size=1, stride=1,
                                       dropout=dropout_rate, name='ConvBlock_2')
        self.gap_softmax = Gmp_softmax_block(ch_in=64, ch_out=num_classes, dropout=dropout_rate, use_cam=use_cam)
        self.log_softmax = nn.LogSoftmax(dim=1)  ## dim 1 for the feature length
        self.device = t.device('cuda' if t.cuda.is_available() else 'cpu')

    def forward(self, x1, x2, x3):
        x1 = self.model1(x1)
        x2 = self.model2(x2)
        x3 = self.model3(x3)
        ## zero padding to the same length in each sample
        max_len = np.max((x1.shape[2], x2.shape[2], x3.shape[2]))
        x1_pad = t.zeros((x1.shape[0], x1.shape[1], max_len))
        x1_pad[:, :, :x1.shape[2]] = x1
        x2_pad = t.zeros((x2.shape[0], x2.shape[1], max_len))
        x2_pad[:, :, :x2.shape[2]] = x2
        x3_pad = t.zeros((x3.shape[0], x3.shape[1], max_len))
        x3_pad[:, :, :x3.shape[2]] = x3
        # concat inputs
        x = t.cat((x1_pad, x2_pad, x3_pad), dim=1).to(self.device)
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.gap_softmax(x)
        # x = self.log_softmax(x)
        return x

class FCN(nn.Module):
    """FCN for one matrix including all input features [acc,gyr,mag,mic]
        Input shape should be [Bn, Features, Seq_len]
        Bn: Number of Batch
        Features: the Feature dimension
        Seq_Len: the Sequence Length (Time steps)
    """
    def __init__(self, ch_in: int, ch_out: List = [64, 128, 128, 64],
                 kernel_size: List = [7, 5, 3, 3],
                 dropout_rate: float = 0.2,
                 use_full: bool = True,
                 use_fc: bool = True,
                 use_pooling: bool = True,
                 num_classes: int = 2,
                 input_dim: tuple = (1, 1, 150)):
        """
        Parameters
        ----------
        ch_in (int) : Input Channel (Sensor Features)
        ch_out (List) : Output Channels of each Convolutional Layer
        dropout_rate (float) : dropout rate for the Gap Softmax Block
        kernel_size (List) : kernel size of each Convolutional Layer,
                            This should have the same size as ch_out
        use_full (bool) : True, use Conv1d 1x1, dropout and batchnorm in Gmp_softmax_block
                          False, only AdaptiveMaxPool1d in Gmp_softmax_block
        use_fc (bool) : True, use Fully Connected layer as the last layer of model
        use_pooling (bool) : True, use Global Max pooling in Softmax Class
                            False, not use it
        num_classes (int) : the number of classes
        input_dim (tuple): the dimensions of a input sample,
                            default: (1, 1, 150)
        """
        super(FCN, self).__init__()
        if len(kernel_size) != len(ch_out) and not use_fc:
            raise ValueError(f"The Length of Kernel Size {kernel_size} is not same"
                             f" as Channel Out {ch_out}")
        elif len(kernel_size) + 1 != len(ch_out) and use_fc:
            raise ValueError(f"The Length of Kernel Size {kernel_size} should be one size "
                             f"smaller than Channel Out {ch_out}")

        if use_fc:
            channels4fcn = ch_out[:-1]
            ch_in4gap_softmax = channels4fcn[-1]
            ch_out4gap_softmax = ch_out[-1]
        else:
            channels4fcn = ch_out
            ch_in4gap_softmax = ch_out[-1]
            ch_out4gap_softmax = num_classes

        self.layers = []
        ## Conv Block
        for i in range(len(channels4fcn)):
            ch_in = ch_in if i == 0 else channels4fcn[i-1]
            self.layers += [Conv1d_block(ch_in=ch_in, ch_out=channels4fcn[i],
                                         kernel_size=kernel_size[i],
                                         stride=1,
                                         dropout=dropout_rate,
                                         name=f"ConvBlock_{i}",
                                         use_dropout=True)]

        self.conv_block = nn.Sequential(*self.layers)

        ## get the feature dim from Fcn_layers
        features_dim = self.conv_block(t.rand(input_dim)).shape
        ## Global Pooling
        self.gap_softmax = Gmp_softmax_block(ch_in=ch_in4gap_softmax,
                                             ch_out=ch_out4gap_softmax,
                                             dropout=dropout_rate,
                                             name='Final_Softmax',
                                             use_full=use_full,
                                             use_fc=use_fc,
                                             use_pooling=use_pooling,
                                             input_dim=features_dim)

        self.layers += [self.gap_softmax]
        self.fcn = nn.Sequential(*self.layers)
        # self.log_softmax = nn.LogSoftmax(dim=1) ## dim 1 for the feature length
        self.device = t.device('cuda' if t.cuda.is_available() else 'cpu')

    def forward(self, x):
        ## we use nn.CrossEntropyLoss
        return self.fcn(x)

    def model_name(self):
        return "FCN"


class FCN_3baseline(nn.Module):
    """BaselineFCN, For Mag, Imu without Aud"""
    def __init__(self, in_mag: int = 3, in_acc: int = 3, in_gyr: int = 3, num_classes: int = 4, use_cam: bool = True):
        super(FCN_3baseline, self).__init__()
        ## Magnetic Field
        self.mag_convblock1 = Conv1d_block(ch_in=in_mag, ch_out=128, kernel_size=8, stride=1, dropout=0,
                                           name='Mag_ConvBlock0', use_dropout=False)
        self.mag_convblock2 = Conv1d_block(ch_in=128, ch_out=256, kernel_size=5, stride=1, dropout=0,
                                           name='Mag_ConvBlock1', use_dropout=False)
        self.mag_convblock3 = Conv1d_block(ch_in=256, ch_out=128, kernel_size=3, stride=1, dropout=0,
                                           name='Mag_ConvBlock2', use_dropout=False)
        ## ACC
        self.acc_convblock1 = Conv1d_block(ch_in=in_acc, ch_out=128, kernel_size=8, stride=1, dropout=0,
                                           name='IMU_ConvBlock0', use_dropout=False)
        self.acc_convblock2 = Conv1d_block(ch_in=128, ch_out=256, kernel_size=5, stride=1, dropout=0,
                                           name='IMU_ConvBlock1', use_dropout=False)
        self.acc_convblock3 = Conv1d_block(ch_in=256, ch_out=128, kernel_size=3, stride=1, dropout=0,
                                           name='IMU_ConvBlock2', use_dropout=False)
        ## GYR
        self.gyr_convblock1 = Conv1d_block(ch_in=in_gyr, ch_out=128, kernel_size=8, stride=1, dropout=0,
                                           name='IMU_ConvBlock0', use_dropout=False)
        self.gyr_convblock2 = Conv1d_block(ch_in=128, ch_out=256, kernel_size=5, stride=1, dropout=0,
                                           name='IMU_ConvBlock1', use_dropout=False)
        self.gyr_convblock3 = Conv1d_block(ch_in=256, ch_out=128, kernel_size=3, stride=1, dropout=0,
                                           name='IMU_ConvBlock2', use_dropout=False)

        self.gap_softmax = Gmp_softmax_block(ch_in=128*3, ch_out=num_classes, dropout=0.2, name='Final_Softmax',
                                             use_full=True, use_cam=use_cam)
        self.log_softmax = nn.LogSoftmax(dim=1)  ## dim 1 for the feature length
        self.device = t.device('cuda' if t.cuda.is_available() else 'cpu')

    def forward(self, x1, x2, x3):
        x1 = self.mag_convblock1(x1)
        x1 = self.mag_convblock2(x1)
        x1 = self.mag_convblock3(x1)

        x2 = self.acc_convblock1(x2)
        x2 = self.acc_convblock2(x2)
        x2 = self.acc_convblock3(x2)

        x3 = self.gyr_convblock1(x3)
        x3 = self.gyr_convblock2(x3)
        x3 = self.gyr_convblock3(x3)
        ## zero padding to the same length in each sample
        max_len = np.max((x1.shape[2], x2.shape[2], x3.shape[2]))
        x1_pad = t.zeros((x1.shape[0], x1.shape[1], max_len))
        x1_pad[:, :, :x1.shape[2]] = x1
        x2_pad = t.zeros((x2.shape[0], x2.shape[1], max_len))
        x2_pad[:, :, :x2.shape[2]] = x2
        x3_pad = t.zeros((x3.shape[0], x3.shape[1], max_len))
        x3_pad[:, :, :x3.shape[2]] = x3
        # concat inputs
        x = t.cat((x1_pad, x2_pad, x3_pad), dim=1).to(self.device)
        output = self.gap_softmax(x)
        # output = self.log_softmax(output)
        return output

class FCN_4baseline(nn.Module):
    """BaselineFcn, For Mag, Imu and Aud"""
    def __init__(self, in_mag: int = 3, in_acc: int = 3, in_gyr: int = 3, in_aud: int = 1, num_classes: int = 4, use_cam: bool = True):
        super(FCN_4baseline, self).__init__()
        ## Magnetic Field
        self.mag_convblock1 = Conv1d_block(ch_in=in_mag, ch_out=128, kernel_size=8, stride=1, dropout=0,
                                           name='Mag_ConvBlock0', use_dropout=False)
        self.mag_convblock2 = Conv1d_block(ch_in=128, ch_out=256, kernel_size=5, stride=1, dropout=0,
                                           name='Mag_ConvBlock1', use_dropout=False)
        self.mag_convblock3 = Conv1d_block(ch_in=256, ch_out=128, kernel_size=3, stride=1, dropout=0,
                                           name='Mag_ConvBlock2', use_dropout=False)
        ## ACC
        self.acc_convblock1 = Conv1d_block(ch_in=in_acc, ch_out=128, kernel_size=8, stride=1, dropout=0,
                                           name='Acc_ConvBlock0', use_dropout=False)
        self.acc_convblock2 = Conv1d_block(ch_in=128, ch_out=256, kernel_size=5, stride=1, dropout=0,
                                           name='Acc_ConvBlock1', use_dropout=False)
        self.acc_convblock3 = Conv1d_block(ch_in=256, ch_out=128, kernel_size=3, stride=1, dropout=0,
                                           name='Acc_ConvBlock2', use_dropout=False)
        ## GYR
        self.gyr_convblock1 = Conv1d_block(ch_in=in_gyr, ch_out=128, kernel_size=8, stride=1, dropout=0,
                                           name='Gyr_ConvBlock0', use_dropout=False)
        self.gyr_convblock2 = Conv1d_block(ch_in=128, ch_out=256, kernel_size=5, stride=1, dropout=0,
                                           name='Gyr_ConvBlock1', use_dropout=False)
        self.gyr_convblock3 = Conv1d_block(ch_in=256, ch_out=128, kernel_size=3, stride=1, dropout=0,
                                           name='Gyr_ConvBlock2', use_dropout=False)
        ## Audio
        self.aud_convblock1 = Conv1d_block(ch_in=in_aud, ch_out=128, kernel_size=8, stride=1, dropout=0,
                                           name='Aud_ConvBlock0', use_dropout=False)
        self.aud_convblock2 = Conv1d_block(ch_in=128, ch_out=256, kernel_size=5, stride=1, dropout=0,
                                           name='Aud_ConvBlock1', use_dropout=False)
        self.aud_convblock3 = Conv1d_block(ch_in=256, ch_out=128, kernel_size=3, stride=1, dropout=0,
                                           name='Aud_ConvBlock2', use_dropout=False)

        self.gap_softmax = Gmp_softmax_block(ch_in=128*4, ch_out=num_classes, dropout=0.2, name='Final_Softmax',
                                             use_full=True, use_cam=use_cam)

        self.log_softmax = nn.LogSoftmax(dim=1)  ## dim 1 for the feature length

        self.device = t.device('cuda' if t.cuda.is_available() else 'cpu')

    def forward(self, x1, x2, x3, x4):
        x3 = self.mag_convblock1(x3)
        x3 = self.mag_convblock2(x3)
        x3 = self.mag_convblock3(x3)

        x1 = self.acc_convblock1(x1)
        x1 = self.acc_convblock2(x1)
        x1 = self.acc_convblock3(x1)

        x2 = self.gyr_convblock1(x2)
        x2 = self.gyr_convblock2(x2)
        x2 = self.gyr_convblock3(x2)

        x4 = self.aud_convblock1(x4)
        x4 = self.aud_convblock2(x4)
        x4 = self.aud_convblock3(x4)

        ## zero padding to the same length in each sample
        max_len = np.max((x1.shape[2], x2.shape[2], x3.shape[2], x4.shape[2]))
        x1_pad = t.zeros((x1.shape[0], x1.shape[1], max_len))
        x1_pad[:, :, :x1.shape[2]] = x1
        x2_pad = t.zeros((x2.shape[0], x2.shape[1], max_len))
        x2_pad[:, :, :x2.shape[2]] = x2
        x3_pad = t.zeros((x3.shape[0], x3.shape[1], max_len))
        x3_pad[:, :, :x3.shape[2]] = x3
        x4_pad = t.zeros((x4.shape[0], x4.shape[1], max_len))
        x4_pad[:, :, :x4.shape[2]] = x4
        # concat inputs
        x = t.cat((x1_pad, x2_pad, x3_pad, x4_pad), dim=1).to(self.device)
        output = self.gap_softmax(x)
        # output = self.log_softmax(output)
        return output




