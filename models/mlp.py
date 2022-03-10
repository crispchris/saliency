## MLP Model MultiLayer perceptrons
## For 5 Layers

## -------------------
## --- Third-party ---
## -------------------
import torch as t
import torch.nn as nn
import numpy as np
from typing import List

class Linearblock(nn.Module):
    def __init__(self, ch_in, ch_out, dropout: float=0.1):
        super(Linearblock, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.dense_layer = nn.Linear(ch_in, ch_out)
        self.relu = nn.ReLU()

        self.sequential = nn.Sequential(
            self.dropout,
            self.dense_layer,
            self.relu
        )
    def forward(self, x):
        out = self.sequential(x)
        return out
    
class Unsqueeze(nn.Module):
    def __init__(self):
        super(Unsqueeze, self).__init__()
        
    def forward(self, x):
        out = x.unsqueeze(-1)
        return out
    
    
class MLP(nn.Module):
    def __init__(self, ch_in: int, ch_out: list, dropout_rate: list, num_classes: int):
        super(MLP, self).__init__()
        
        self.layers = []
        self.layers += [nn.Flatten()]
        for i in range(len(ch_out)):
            in_channels = ch_in if i == 0 else ch_out[i-1]
            out_channels = ch_out[i]
            self.layers += [Linearblock(in_channels, 
                                   out_channels,
                                   dropout_rate[i])]
        self.layers += [nn.Linear(ch_out[-1], num_classes)]
        #self.layers += [nn.Relu()]
        self.layers += [Unsqueeze()]
        self.mlp = nn.Sequential(*self.layers)

    def forward(self, x):
        #x = t.reshape(x, (x.shape[0], -1))  ## return [N, L*C]
        x = self.mlp(x)        ## return [N, Classes]
        #x = x.transpose(1, 2)       ## return [N, C, L]
        #x = x.unsqueeze(-1)
        return x
    
    def model_name(self):
        return "MLP"

# class MLP(nn.Module):
#     """For one Matrix, including all input features
#         input combines all features
#     """
#     def __init__(self, ch_in: int, ch_out: list, dropout_rate: list, num_classes: int):
#         super(MLP, self).__init__()
#         # the input shape for the linear layer (MLP) should be [N, *, Features]

#         self.layer1 = nn.Linear(ch_in, ch_out[0])
#         self.relu1 = nn.ReLU()
#         self.dropout1 = nn.Dropout(dropout_rate[0])

#         self.layer2 = nn.Linear(ch_out[0], ch_out[1])
#         self.relu2 = nn.ReLU()
#         self.dropout2 = nn.Dropout(dropout_rate[1])

#         self.layer3 = nn.Linear(ch_out[1], ch_out[2])
#         self.relu3 = nn.ReLU()
#         self.dropout3 = nn.Dropout(dropout_rate[2])

#         self.layer4 = nn.Linear(ch_out[2], ch_out[3])
#         self.relu4 = nn.ReLU()
#         self.dropout4 = nn.Dropout(dropout_rate[3])

#         self.layer5 = nn.Linear(ch_out[3], num_classes)
#         ## if we use nn.CrossEntropyLoss, we don't need LogSoftmax (CrossEntropyLoss combines Logsoftmax and NLLLoss)
#         # self.log_softmax = nn.LogSoftmax(dim=-1) ## dim -1 for the feature length

#         self.device = t.device('cuda' if t.cuda.is_available() else 'cpu')
#         self.sequence = nn.Sequential(
#             self.layer1,
#             self.relu1,
#             self.dropout1,
#             self.layer2,
#             self.relu2,
#             self.dropout2,
#             self.layer3,
#             self.relu3,
#             self.dropout3,
#             self.layer4,
#             self.relu4,
#             self.dropout4,
#             self.layer5
#         )


#     def forward(self, x):           ## input x [N, F, L]
#         #x = x.transpose(1, 2)
#         x = t.reshape(x, (x.shape[0], -1))
#         x = self.sequence(x)        ## return [N, L, C]
#         #x = x.transpose(1, 2)       ## return [N, C, L]
#         x = x.unsqueeze(-1)

#         return x

#     def model_name(self):
#         return "MLP"