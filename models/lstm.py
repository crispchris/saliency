## -------------------
## --- Third-Party ---
## -------------------
import sys
sys.path.append('..')
import numpy as np
import torch as t
import torch.nn as nn


## --- LSTM ---
class LSTM(nn.Module):
    def __init__(self, ch_in: int, hidden_size: int, num_layers: int, dropout: float,
                 bidirectional: bool = False,
                 num_classes : int = 2):
        """
        LSTM Network

        Parameters
        ----------
        ch_in (int) : The number of expected features in the input
        hidden_size (int) : The number of features in the hidden state h
        num_layers (int) : The number of recurrent layers (Stack together)
        dropout (float) : If non-zero, introduces a Dropout layer on the outputs of each LSTM layer
                        However, dropout doesn't work that good in recurent neural network

        bidirectional (bool) : if True, a bidirectional LSTM
                                This preserve information from past and future
        num_classes (int) : The number of classes, which should be classified
        """
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1

        self.rnn = nn.LSTM(input_size=ch_in,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           batch_first=True,
                           dropout=dropout,
                           bidirectional=bidirectional)
        self.fc = nn.Linear(in_features=hidden_size, out_features=num_classes)

        self.device = t.device("cuda" if t.cuda.is_available() else "cpu")

    def forward(self, x):
        x = x.transpose(1, 2) ## get shape [B, Seq, input_size]

        ## Set Initial States
        ## shape [B, n_layer*n_directions, hidden_size]
        h_0 = t.zeros(self.num_layers*self.num_directions, x.shape[0],  self.hidden_size).to(self.device)
        c_0 = t.zeros(self.num_layers*self.num_directions, x.shape[0], self.hidden_size).to(self.device)

        output, (h_n, c_n) = self.rnn(x, (h_0, c_0))
        output = output[:, -1, :]  ## get only the last time Tn, output shape : [B, T, hidden_size*direction]
        output = output.unsqueeze(1)
        output = self.fc(output).transpose(1, 2)

        return output ## shape [B, Hidden(Class), 1(last seq)]

    def model_name(self):
        return "LSTM"

## --- LSTM for densely labeling ---
class LSTM_dense(nn.Module):
    def __init__(self, ch_in: int, hidden_size: int, num_layers: int, dropout: float,
                 bidirectional: bool = False,
                 num_classes : int = 2):
        """
        LSTM Network

        Parameters
        ----------
        ch_in (int) : The number of expected features in the input
        hidden_size (int) : The number of features in the hidden state h
        num_layers (int) : The number of recurrent layers (Stack together)
        dropout (float) : If non-zero, introduces a Dropout layer on the outputs of each LSTM layer
                        However, dropout doesn't work that good in recurent neural network

        bidirectional (bool) : if True, a bidirectional LSTM
                                This preserve information from past and future
        num_classes (int) : The number of classes, which should be classified
        """
        super(LSTM_dense, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1

        self.rnn = nn.LSTM(input_size=ch_in,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           batch_first=True,
                           dropout=dropout,
                           bidirectional=bidirectional)
        self.fc = nn.Linear(in_features=hidden_size * 2, out_features=num_classes)
        self.device = t.device("cuda" if t.cuda.is_available() else "cpu")

    def forward(self, x):
        x = x.transpose(1, 2) ## get shape [B, Seq, input_size]

        ## Set Initial States
        ## shape [B, n_layer*n_directions, hidden_size]
        h_0 = t.zeros(self.num_layers*self.num_directions, x.shape[0],  self.hidden_size).to(self.device)
        c_0 = t.zeros(self.num_layers*self.num_directions, x.shape[0], self.hidden_size).to(self.device)

        output, (h_n, c_n) = self.rnn(x, (h_0, c_0))
        ## get the whole time steps, output shape : [B, T, hidden_size*direction]
        output = self.fc(output)
        output = output.transpose(1, 2)  # output shape : [B, Class, T]
        return output ## shape [B, Hidden(Class), T]

    def model_name(self):
        return "LSTM_dense"