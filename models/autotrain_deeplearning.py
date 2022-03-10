"""
build the deep learning model
refer to: autotrain_deep_learning.py from fraunhofer tool tracking project
"""
## -------------------
## --- third-party ---
## -------------------
import torch.nn.functional as F
import torch as t
import torch.nn as nn
from sklearn.metrics import classification_report
import optuna
import numpy as np
import json
from typing import List, Dict, Tuple
import copy
import tarfile
import shutil
import os

## -----------
## --- own ---
## -----------
from .tcn_layer import TCN_layer
from .basic_blocks import Conv1d_block, Gap_softmax_block
from .models import TCN_3base, TCN_4base, FCN_4baseline, FCN_3baseline
from tool_tracking.tool_utils import Dataset

def get_implemented_networks() -> List[str]:
    """
    The network type that are implemented

    Returns
    -------
    ['str',...,'str] list of strings, i.e., LSTM, FCN, TCN
    """
    return ['TDLSTM', 'FCN', 'TCN', 'BaselineFCN']

def are_implemented_networks(nets: List[str]) -> bool:
    """
    Check whether the neural networks are implemented
    Parameters
    ----------
    nets is a list of strings

    Returns
    -------
    Boolean False if any element in nets is not defined
    """
    for n in nets:
        if n not in get_implemented_networks():
            return False
    return True

def init_torch():
    """
    Check whether CUDA is currently available.
    Returns
    -------
    Boolean
    """
    return t.cuda.is_available()

class ANN:
    """
    Class that contains a neural networks common properties:
    - batchsize, num of classes, num of epochs
    - fit(), evaluate
    - constructors for basic blocks (Conv-block, GAP-Sofmax, ..)
    """

    def __init__(self, data: Dataset, num_classes: int, optuna_search_space: Dict):
        """

        Parameters
        ----------
        data that contains has_audio, has_imu, has_mag, can be replaced with simpler data-structure in the future
        num_classes, for constructing the softmax classification dimension
        optuna_search_space, for parameterizing optuna
        """
        self.data = data
        self.num_classes = num_classes
        self.optuna_search_space = optuna_search_space
        # some fixed values for training
        self.batch_size = 32
        self.epochs = 100
        # internal helpers
        self.model_ = None  ## the current model
        self.params_ = {}   ## current trial parameters

    def fit(self, TrainData: Dataset, ValData: Dataset, trial: optuna.trial):
        """
        Trains an ANN based on the parameters given in trial.

        Training uses Early Stopping
        - Torch monitors val_loss
        - Trial pruning (Optuna's Early Stopping between trials)

        Parameters
        ----------
        TrainData
        ValData
        trial: Optuna Parameters

        Returns
        -------

        """
        pass

    def evaluate(self, X, y, verbose: int = 0) -> List:
        """
        Evaluate the trained model on the data

        Returns
        -------
        Returns the loss value & metrics values for the model in test model
        """
        pass

    def _conv1d_block(self, ch_in: int, ch_out: int, kernel_size: int, strides: int, dropout: float,
                     padding: Tuple[int] = (0, 0),
                     name: str = 'ConvBlock', use_dropout: bool = True):
        """
        A Convolutional block with Conv1D, Dropout, BatchNorm and ReLU activation
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

        Returns
        -------
        A Torch nn Module of Conv1d block
        """
        return Conv1d_block(ch_in=ch_in, ch_out=ch_out, kernel_size= kernel_size, stride=strides,
                            dropout=dropout, padding=padding, name=name, use_dropout=use_dropout)

    def _gap_softmax_block(self, ch_in: int, ch_out: int, dropout: float = 0.2,
                 name: str = 'Gap_softmax_block', use_full: bool = True):
        """
        Final block of Fully Convolutional Network
        Including: GlobalAveragePooling (GAP) followed by Softmax
        Parameters
        ----------
        ch_in : Integer, the dimensionality of the input space
                (i.e. the number of input filters in the convolution).
        ch_out: Integer, the dimensionality of the output space
                (i.e. the number of output filters in the convolution).
        dropout: Float between 0 and 1. Fraction of the input units to drop.
        name: str for printing

        Returns
        -------
        A Torch Model of GAP block with SoftMax
        """
        return Gap_softmax_block(ch_in=ch_in, ch_out=ch_out, dropout=dropout, name=name,
                                 use_full=use_full)

    def _td_con1d_block(self):
        pass

class TCN(ANN):
    """
    Temporal Convolutional Network: Fully Convolutional Network that uses "Stacked Dilations" in order to
    learn temporal and causal properties of the data.

    Implementation supports
    - skip connections
    - stacking
    - padding either (same) or causal
    - batch and layer norm

    Paper: https://arxiv.org/pdf/1803.01271.pdf
    """

    def __init__(self, data: Dataset, num_classes: int, optuna_search_space: Dict):
        super().__init__(data=data, num_classes=num_classes, optuna_search_space=optuna_search_space)
        self.name_ = "TCN"

    def suggest(self, trial: optuna.trial):
        """
        The parameters of the current trial are automatically selected by Optuna.
        Parameters
        ----------
        trial Optuna trial
        """
        self.params_["Dropout Rate"] = trial.suggest_uniform('Dropout Rate',
                                                             self.optuna_search_space['Dropout Rate'][0],
                                                             self.optuna_search_space['Dropout Rate'][1])
        self.params_["Learning Rate"] = trial.suggest_uniform('Learning Rate',
                                                             self.optuna_search_space['Learning Rate'][0],
                                                             self.optuna_search_space['Learning Rate'][1])

    def construct(self):
        # Fully Convolutional inputs
        ## three input (mag imu aud)
        tcn_mag = TCN_layer(ch_in=3, dilation=[1, 2, 4])
        tcn_imu = TCN_layer(ch_in=6, dilation=[1, 2, 4])

        if self.data.has_audio():
            tcn_aud = TCN_layer(ch_in=1, dilation=[1, 2, 4])
            # concat inputs
            tcn = TCN_4base(tcn_imu, tcn_mag, tcn_aud, dropout_rate=self.params_['Dropout Rate'], num_classes=self.num_classes)
        else:
            tcn = TCN_3base(tcn_imu, tcn_mag, dropout_rate=self.params_['Dropout Rate'], num_classes=self.num_classes)
        self.model_ = tcn

class BaselineFCN(ANN):
    """
    This is the baseline FCN published in literature, adapted for three inputs
    paper: https://arxiv.org/pdf/1611.06455.pdf
    """
    def __init__(self, data: Dataset, num_classes: int, optuna_search_space: Dict):
        super().__init__(data=data, num_classes=num_classes, optuna_search_space=optuna_search_space)
        self.name_ = "BaselineFCN"

    def suggest(self, trial: optuna.trial):
        """
        The parameters of the current trial are automatically selected by Optuna.
        Parameters
        ----------
        trial Optuna trial
        """
        self.params_["Learning Rate"] = trial.suggest_uniform('Learning Rate',
                                                             self.optuna_search_space['Learning Rate'][0],
                                                             self.optuna_search_space['Learning Rate'][1])

    def construct(self):
        """
        Builds a 3-headed, as in three inputs, Fully Convolutional Neural Network, similarly to [1].
        The three heads encoded features are concatenaded, then 1x1 Conv and Dense (softmax) finish.

        [1] "Time Series Classification from Scratch with Deep Neural Networks: A Strong Baseline"
        URL: https://arxiv.org/pdf/1611.06455.pdf
        """
        # Fully Convolutional inputs
        if self.data.has_audio():
            baselineFCN = FCN_4baseline(in_mag=3, in_imu=6, in_aud=1, num_classes=self.num_classes)
        else:
            baselineFCN = FCN_3baseline(in_mag=3, in_imu=6, num_classes=self.num_classes)
        self.model_ = baselineFCN


class FCN(ANN):
    pass