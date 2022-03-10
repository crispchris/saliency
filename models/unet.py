## Unet for Time series (Utime)
## refer to: Origin paper https://github.com/perslev/U-Time/blob/master/utime/models/utime.py
## and for pytorch: https://github.com/neergaard/utime-pytorch/blob/main/models/utime.py

## -------------------
## --- Third-party ---
## -------------------
import sys
sys.path.append('..')
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from typing import List

class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.interp = F.interpolate
        self.size = size
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
        return x

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels=5, out_channels=5, kernel_size=3,
                 dilation=1, activation="relu"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.activation = activation
        self.padding = (self.dilation * (self.kernel_size - 1)) // 2
        # self.padding = (self.kernel_size + (self.kernel_size - 1) * (self.dilation - 1) - 1) // 2

        self.layers = nn.Sequential(
            # nn.ConstantPad1d(padding=(self.padding, self.padding), value=0),
            nn.Conv1d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                padding=self.padding,
                bias=True,
            ),
            nn.ReLU(),
            nn.BatchNorm1d(self.out_channels),
        )
        nn.init.xavier_uniform_(self.layers[0].weight)
        nn.init.zeros_(self.layers[0].bias)

    def forward(self, x):
        return self.layers(x)

class Encoder(nn.Module):
    def __init__(self, filters=[16, 32, 64, 128], in_channels=5,
                 maxpool_kernels=[10, 8, 6, 4], kernel_size=5, dilation=2):
        super().__init__()
        self.filters = filters
        self.in_channels = in_channels
        self.maxpool_kernels = maxpool_kernels
        self.kernel_size = kernel_size
        self.dilation = dilation
        assert len(self.filters) == len(
            self.maxpool_kernels
        ), f"Number of filters ({len(self.filters)}) does not equal number of supplied maxpool kernels ({len(self.maxpool_kernels)})!"

        self.depth = len(self.filters)

        # fmt: off
        self.blocks = nn.ModuleList([nn.Sequential(
            ConvBNReLU(
                in_channels=self.in_channels if k == 0 else self.filters[k - 1],
                out_channels=self.filters[k],
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                activation="relu",
            ),
            ConvBNReLU(
                in_channels=self.filters[k],
                out_channels=self.filters[k],
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                activation="relu",
            ),
        ) for k in range(self.depth)])
        # fmt: on

        self.maxpools = nn.ModuleList([nn.MaxPool1d(self.maxpool_kernels[k]) for k in range(self.depth)])

        self.bottom = nn.Sequential(
            ConvBNReLU(
                in_channels=self.filters[-1],
                out_channels=self.filters[-1] * 2,
                kernel_size=self.kernel_size,
            ),
            ConvBNReLU(
                in_channels=self.filters[-1] * 2,
                out_channels=self.filters[-1] * 2,
                kernel_size=self.kernel_size
            ),
        )

    def forward(self, x):
        shortcuts = []
        for layer, maxpool in zip(self.blocks, self.maxpools):
            z = layer(x)
            shortcuts.append(z)
            x = maxpool(z)

        # Bottom part
        encoded = self.bottom(x)

        return encoded, shortcuts

class Decoder(nn.Module):
    def __init__(self, filters=[128, 64, 32, 16], upsample_kernels=[4, 6, 8, 10],
                 in_channels=256, dilation=5, kernel_size=5):
        super().__init__()
        self.filters = filters
        self.upsample_kernels = upsample_kernels
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        # self.out_channels = out_channels
        assert len(self.filters) == len(
            self.upsample_kernels
        ), f"Number of filters ({len(self.filters)}) does not equal number of supplied upsample kernels ({len(self.upsample_kernels)})!"
        self.depth = len(self.filters)

        # fmt: off
        self.upsamples = nn.ModuleList([nn.Sequential(
            # nn.Upsample(scale_factor=self.upsample_kernels[k]),
            # nn.ConvTranspose1d(
            #     in_channels=self.in_channels if k == 0 else self.filters[k - 1],
            #     out_channels=self.in_channels if k == 0 else self.filters[k - 1],
            #     kernel_size=self.kernel_size,
            #     dilation=1,
            #     padding=1
            # ),
            ConvBNReLU(
                in_channels=self.in_channels if k == 0 else self.filters[k - 1],
                out_channels=self.filters[k],
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                activation='relu',
            )
        ) for k in range(self.depth)])

        self.blocks = nn.ModuleList([nn.Sequential(
            ConvBNReLU(
                in_channels=self.in_channels if k == 0 else self.filters[k - 1],
                out_channels=self.filters[k],
                kernel_size=self.kernel_size,
            ),
            ConvBNReLU(
                in_channels=self.filters[k],
                out_channels=self.filters[k],
                kernel_size=self.kernel_size,
            ),
        ) for k in range(self.depth)])
        # fmt: off

    def forward(self, z, shortcuts):
        for upsample, block, shortcut in zip(self.upsamples, self.blocks, shortcuts[::-1]):
            z = F.upsample(z, size=[shortcut.shape[-1]], mode="nearest")
            z = upsample(z)
            z = t.cat([shortcut, z], dim=1)
            z = block(z)

        return z ## return [Batch, Ch_out, len]

class SegmentClassifier(nn.Module):
    def __init__(self, ch_in, num_classes=5):
        super().__init__()
        self.num_classes = num_classes
        self.layers = nn.Sequential(
            # nn.AvgPool2d(kernel_size=(1, self.epoch_length * self.sampling_frequency)),
            # nn.Flatten(start_dim=2),
            # nn.ConstantPad1d(padding=(self.padding, self.padding), value=0),
            nn.Conv1d(in_channels=ch_in, out_channels=self.num_classes, kernel_size=1),
            nn.ReLU()
            # nn.Softmax(dim=1),
        )
        nn.init.xavier_uniform_(self.layers[0].weight)
        nn.init.zeros_(self.layers[0].bias)

    def forward(self, x):
        # batch_size, num_classes, n_samples = x.shape
        # z = x.reshape((batch_size, num_classes, -1, self.epoch_length * self.sampling_frequency))
        return self.layers(x)

class Utime(nn.Module):
    """For one Matrix including all input features [acc, gyr, mag]
    Unet is designed to produce the output as the same size as input (The length t) -> Useful for Densely Labeling
    """
    def __init__(self, ch_in: int = None,
                 ch_out: List = None,
                 maxpool_kernels: List = None,
                 kernel_size=None,
                 dilation=None,
                 num_classes=None):
        super(Utime, self).__init__()
        self.encoder = Encoder(
            filters=ch_out,
            in_channels=ch_in,
            maxpool_kernels=maxpool_kernels,
            kernel_size=kernel_size,
            dilation=dilation
        )
        self.decoder = Decoder(
            filters=ch_out[::-1],
            upsample_kernels=maxpool_kernels[::-1],
            in_channels=ch_out[-1] * 2,
            kernel_size=kernel_size,
            dilation=dilation
        )
        self.segmet_classifier = SegmentClassifier(
            ch_in=ch_out[0],
            num_classes=num_classes
        )

    def forward(self, x):
        ## Run through Encoder
        z, shorcuts = self.encoder(x)

        ## Run through Decoder
        z = self.decoder(z, shorcuts)

        ## classifier
        z = self.segmet_classifier(z)
        return z








