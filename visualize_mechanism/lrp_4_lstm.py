"""
References: https://github.com/ArrasL/LRP_for_LSTM
Adapt: LRP Method for LSTM layer
"""
'''
@author: Leila Arras
@maintainer: Leila Arras
@date: 21.06.2017
@version: 1.0+
@copyright: Copyright (c) 2017, Leila Arras, Gregoire Montavon, Klaus-Robert Mueller, Wojciech Samek
@license: see LICENSE file in repository root
'''

import numpy as np
import pickle
from numpy import newaxis as na
from visualize_mechanism.lrp_linear_layer import *
from torch import Tensor

class LSTM_bidi:
    def __init__(self, model,
                 device):
        """
        Load trained model and set the weights and bias.
        For bi-direction
        (Only for one layer LSTM right now)
        """
        ## Pytorch model
        self.model = model
        self.device = device

        # LSTM left encoder
        self.Wxh_Left = self.model.rnn.weight_ih_l0.data.cpu().detach().numpy()  # shape 4d*e [4*Hidden, Input]
        self.bxh_Left = self.model.rnn.bias_ih_l0.data.cpu().detach().numpy()  # shape 4d
        self.Whh_Left = self.model.rnn.weight_hh_l0.data.cpu().detach().numpy()  # shape 4d*d [4*Hidden, Hidden]
        self.bhh_Left = self.model.rnn.bias_hh_l0.data.cpu().detach().numpy()  # shape 4d
        # # LSTM right encoder
        self.Wxh_Right = self.model.rnn.weight_ih_l0_reverse.data.cpu().detach().numpy()  # shape 4d*e [4*Hidden, Input]
        self.bxh_Right = self.model.rnn.bias_ih_l0_reverse.data.cpu().detach().numpy()  # shape 4d
        self.Whh_Right = self.model.rnn.weight_hh_l0_reverse.data.cpu().detach().numpy()  # shape 4d*d [4*Hidden, Hidden]
        self.bhh_Right = self.model.rnn.bias_hh_l0_reverse.data.cpu().detach().numpy()  # shape 4d

        # linear output layer
        fc_weight_shape = self.model.fc.weight.data.shape
        self.Why = self.model.fc.weight.data.cpu().detach().numpy()
        self.bhy = self.model.fc.bias.data.cpu().detach().numpy()
        self.Why_Left = self.model.fc.weight.data[:, :int(fc_weight_shape[-1]/2)].cpu().detach().numpy()  # shape C*d [Class, Hidden]
        self.bhy_Left = self.model.fc.bias.data.cpu().detach().numpy() # Shape C [Class]
        self.Why_Right = self.model.fc.weight.data[:, int(fc_weight_shape[-1]/2):].cpu().detach().numpy()  # shape C*d [Class, Hidden]

    def set_input(self, sample: Tensor,
                  delete_pos=None):
        """
        Build the numerical input sequence x/x_rev from the word indices(Sample) w (+ initialize hidden layers h, c).
        Optionally delete words at positions delete_pos.
        """

        T = sample.shape[-1]  # sequence length
        d = int(self.Wxh_Left.shape[0] / 4)  # hidden layer dimension
        e = sample.shape[1]  ## input sequence channel (feature sensor)
        # x = np.zeros((T, e))
        x = sample.cpu().detach().numpy()[0].T

        if delete_pos is not None:
            x[delete_pos, :] = np.zeros((len(delete_pos), e))

        self.w = sample
        self.x = x
        self.x_rev = x[::-1, :].copy()

        self.h_Left = np.zeros((T + 1, d))
        self.c_Left = np.zeros((T + 1, d))
        self.h_Right = np.zeros((T + 1, d))
        self.c_Right = np.zeros((T + 1, d))

    def forward(self):
        """
        Standard forward pass.
        Compute the hidden layer values (assuming input x/x_rev was previously set)
        """
        T = self.w.shape[-1]
        d = int(self.Wxh_Left.shape[0] / 4) ## hidden size
        # gate indices (assuming the gate ordering in the LSTM weights is i,f,g,o):
        idx = np.hstack((np.arange(0, 2 * d), np.arange(3 * d, 4 * d))).astype(int)  # indices of gates i,f,o together
        idx_i, idx_f, idx_g, idx_o = np.arange(0, d), np.arange(d, 2 * d), np.arange(2 * d, 3 * d), np.arange(3 * d,
                                                                                                              4 * d)  # indices of gates i,f,g,o separately

        # initialize
        self.gates_xh_Left = np.zeros((T, 4 * d))
        self.gates_hh_Left = np.zeros((T, 4 * d))
        self.gates_pre_Left = np.zeros((T, 4 * d))  # gates pre-activation
        self.gates_Left = np.zeros((T, 4 * d))  # gates activation

        self.gates_xh_Right = np.zeros((T, 4 * d))
        self.gates_hh_Right = np.zeros((T, 4 * d))
        self.gates_pre_Right = np.zeros((T, 4 * d))
        self.gates_Right = np.zeros((T, 4 * d))

        for t in range(T):
            self.gates_xh_Left[t] = np.dot(self.Wxh_Left, self.x[t])
            self.gates_hh_Left[t] = np.dot(self.Whh_Left, self.h_Left[t - 1])
            self.gates_pre_Left[t] = self.gates_xh_Left[t] + self.gates_hh_Left[t] + self.bxh_Left + self.bhh_Left
            self.gates_Left[t, idx] = 1.0 / (1.0 + np.exp(- self.gates_pre_Left[t, idx]))
            self.gates_Left[t, idx_g] = np.tanh(self.gates_pre_Left[t, idx_g])
            self.c_Left[t] = self.gates_Left[t, idx_f] * self.c_Left[t - 1] + self.gates_Left[t, idx_i] * \
                             self.gates_Left[t, idx_g]
            self.h_Left[t] = self.gates_Left[t, idx_o] * np.tanh(self.c_Left[t])

            self.gates_xh_Right[t] = np.dot(self.Wxh_Right, self.x_rev[t])
            self.gates_hh_Right[t] = np.dot(self.Whh_Right, self.h_Right[t - 1])
            self.gates_pre_Right[t] = self.gates_xh_Right[t] + self.gates_hh_Right[t] + self.bxh_Right + self.bhh_Right
            self.gates_Right[t, idx] = 1.0 / (1.0 + np.exp(- self.gates_pre_Right[t, idx]))
            self.gates_Right[t, idx_g] = np.tanh(self.gates_pre_Right[t, idx_g])
            self.c_Right[t] = self.gates_Right[t, idx_f] * self.c_Right[t - 1] + self.gates_Right[t, idx_i] * \
                              self.gates_Right[t, idx_g]
            self.h_Right[t] = self.gates_Right[t, idx_o] * np.tanh(self.c_Right[t])

        ## left should take the last time step
        ## right should take the first time step
        self.h = np.concatenate((self.h_Left[T - 1], self.h_Right[0]), axis=0)
        self.y = np.dot(self.Why, self.h) + self.bhy
        # self.y_Left = np.dot(self.Why_Left, self.h_Left[T - 1]) + self.bhy_Left
        # self.y_Right = np.dot(self.Why_Right, self.h_Right[T - 1])
        # self.s = self.y_Left + self.y_Right
        self.s = self.y

        return self.s.copy()  # prediction scores

    def backward(self, w,
                 sensitivity_class):
        """
        Standard gradient backpropagation backward pass.
        Compute the hidden layer gradients by backpropagating a gradient of 1.0 for the class sensitivity_class

        w: sample
        sensitivity_class: the label (the class to be explained)
        """
        # forward pass
        self.set_input(w)
        self.forward()

        T = w.shape[-1]
        d = int(self.Wxh_Left.shape[0] / 4) ## hidden size
        C = self.Why_Left.shape[0]  # number of classes
        # gate indices (assuming the gate ordering in the LSTM weights is i,f,g,o):
        idx = np.hstack((np.arange(0, 2 * d), np.arange(3 * d, 4 * d))).astype(int)  # indices of gates i,f,o together
        idx_i, idx_f, idx_g, idx_o = np.arange(0, d), np.arange(d, 2 * d), np.arange(2 * d, 3 * d), np.arange(3 * d,
                                                                                                              4 * d)  # indices of gates i,f,g,o separately

        # initialize
        self.dx = np.zeros(self.x.shape)
        self.dx_rev = np.zeros(self.x.shape)

        self.dh_Left = np.zeros((T + 1, d))
        self.dc_Left = np.zeros((T + 1, d))
        self.dgates_pre_Left = np.zeros((T, 4 * d))  # gates pre-activation
        self.dgates_Left = np.zeros((T, 4 * d))  # gates activation

        self.dh_Right = np.zeros((T + 1, d))
        self.dc_Right = np.zeros((T + 1, d))
        self.dgates_pre_Right = np.zeros((T, 4 * d))
        self.dgates_Right = np.zeros((T, 4 * d))

        ds = np.zeros((C))
        ds[sensitivity_class] = 1.0
        dy_Left = ds.copy()
        dy_Right = ds.copy()

        self.dh_Left[T - 1] = np.dot(self.Why_Left.T, dy_Left)
        self.dh_Right[T - 1] = np.dot(self.Why_Right.T, dy_Right)

        for t in reversed(range(T)):
            self.dgates_Left[t, idx_o] = self.dh_Left[t] * np.tanh(self.c_Left[t])  # do[t]
            self.dc_Left[t] += self.dh_Left[t] * self.gates_Left[t, idx_o] * (
                        1. - (np.tanh(self.c_Left[t])) ** 2)  # dc[t]
            self.dgates_Left[t, idx_f] = self.dc_Left[t] * self.c_Left[t - 1]  # df[t]
            self.dc_Left[t - 1] = self.dc_Left[t] * self.gates_Left[t, idx_f]  # dc[t-1]
            self.dgates_Left[t, idx_i] = self.dc_Left[t] * self.gates_Left[t, idx_g]  # di[t]
            self.dgates_Left[t, idx_g] = self.dc_Left[t] * self.gates_Left[t, idx_i]  # dg[t]
            self.dgates_pre_Left[t, idx] = self.dgates_Left[t, idx] * self.gates_Left[t, idx] * (
                        1.0 - self.gates_Left[t, idx])  # d ifo pre[t]
            self.dgates_pre_Left[t, idx_g] = self.dgates_Left[t, idx_g] * (
                        1. - (self.gates_Left[t, idx_g]) ** 2)  # d g pre[t]
            self.dh_Left[t - 1] = np.dot(self.Whh_Left.T, self.dgates_pre_Left[t])
            self.dx[t] = np.dot(self.Wxh_Left.T, self.dgates_pre_Left[t])

            self.dgates_Right[t, idx_o] = self.dh_Right[t] * np.tanh(self.c_Right[t])
            self.dc_Right[t] += self.dh_Right[t] * self.gates_Right[t, idx_o] * (1. - (np.tanh(self.c_Right[t])) ** 2)
            self.dgates_Right[t, idx_f] = self.dc_Right[t] * self.c_Right[t - 1]
            self.dc_Right[t - 1] = self.dc_Right[t] * self.gates_Right[t, idx_f]
            self.dgates_Right[t, idx_i] = self.dc_Right[t] * self.gates_Right[t, idx_g]
            self.dgates_Right[t, idx_g] = self.dc_Right[t] * self.gates_Right[t, idx_i]
            self.dgates_pre_Right[t, idx] = self.dgates_Right[t, idx] * self.gates_Right[t, idx] * (
                        1.0 - self.gates_Right[t, idx])
            self.dgates_pre_Right[t, idx_g] = self.dgates_Right[t, idx_g] * (1. - (self.gates_Right[t, idx_g]) ** 2)
            self.dh_Right[t - 1] = np.dot(self.Whh_Right.T, self.dgates_pre_Right[t])
            self.dx_rev[t] = np.dot(self.Wxh_Right.T, self.dgates_pre_Right[t])

        return self.dx.copy(), self.dx_rev[::-1, :].copy()
        # return self.dx.copy()

    def lrp(self, sample, LRP_class, eps=0.001, bias_factor=0.0):
        """
        Layer-wise Relevance Propagation (LRP) backward pass.
        Compute the hidden layer relevances by performing LRP for the target class LRP_class
        (according to the papers:
            - https://doi.org/10.1371/journal.pone.0130140
            - https://doi.org/10.18653/v1/W17-5221 )
        """
        # forward pass
        self.set_input(sample)
        self.forward()

        T = sample.shape[-1]
        d = int(self.Wxh_Left.shape[0] / 4)
        e = int(sample.shape[1])
        C = self.Why_Left.shape[0]  # number of classes

        # gate indices (assuming the gate ordering in the LSTM weights is i,f,g,o):
        idx = np.hstack((np.arange(0, 2 * d), np.arange(3 * d, 4 * d))).astype(int)  # indices of gates i,f,o together
        idx_i, idx_f, idx_g, idx_o = np.arange(0, d), np.arange(d, 2 * d), np.arange(2 * d, 3 * d), np.arange(3 * d,
                                                                                                              4 * d)  # indices of gates i,f,g,o separately

        # initialize
        Rx = np.zeros(self.x.shape)
        Rx_rev = np.zeros(self.x.shape)

        Rh_Left = np.zeros((T + 1, d))
        Rc_Left = np.zeros((T + 1, d))
        Rg_Left = np.zeros((T, d))  # gate g only
        Rh_Right = np.zeros((T + 1, d))
        Rc_Right = np.zeros((T + 1, d))
        Rg_Right = np.zeros((T, d))  # gate g only

        Rout_mask = np.zeros((C))
        Rout_mask[LRP_class] = 1.0

        # format reminder: lrp_linear(hin, w, b, hout, Rout, bias_nb_units, eps, bias_factor)
        Rh = lrp_linear(self.h, self.Why.T, self.bhy, self.s, self.s * Rout_mask,
                        2 * d, eps, bias_factor, debug=False)
        Rh_Left[T - 1] = Rh[:d]
        Rh_Right[T - 1] = Rh[d:]
        # Rh_Left[T - 1] = lrp_linear(self.h_Left[T - 1], self.Why_Left.T, self.bhy_Left, self.s, self.s * Rout_mask,
        #                             2 * d, eps, bias_factor, debug=False)
        # Rh_Right[T - 1] = lrp_linear(self.h_Right[T - 1], self.Why_Right.T, self.bhy_Left, self.s, self.s * Rout_mask,
        #                              2 * d, eps, bias_factor, debug=False)

        for t in reversed(range(T)):
            Rc_Left[t] += Rh_Left[t]
            Rc_Left[t - 1] = lrp_linear(self.gates_Left[t, idx_f] * self.c_Left[t - 1], np.identity(d), np.zeros((d)),
                                        self.c_Left[t], Rc_Left[t], 2 * d, eps, bias_factor, debug=False)
            Rg_Left[t] = lrp_linear(self.gates_Left[t, idx_i] * self.gates_Left[t, idx_g], np.identity(d),
                                    np.zeros((d)), self.c_Left[t], Rc_Left[t], 2 * d, eps, bias_factor, debug=False)
            Rx[t] = lrp_linear(self.x[t], self.Wxh_Left[idx_g].T, self.bxh_Left[idx_g] + self.bhh_Left[idx_g],
                               self.gates_pre_Left[t, idx_g], Rg_Left[t], d + e, eps, bias_factor, debug=False)
            Rh_Left[t - 1] = lrp_linear(self.h_Left[t - 1], self.Whh_Left[idx_g].T,
                                        self.bxh_Left[idx_g] + self.bhh_Left[idx_g], self.gates_pre_Left[t, idx_g],
                                        Rg_Left[t], d + e, eps, bias_factor, debug=False)

            Rc_Right[t] += Rh_Right[t]
            Rc_Right[t - 1] = lrp_linear(self.gates_Right[t, idx_f] * self.c_Right[t - 1], np.identity(d),
                                         np.zeros((d)), self.c_Right[t], Rc_Right[t], 2 * d, eps, bias_factor,
                                         debug=False)
            Rg_Right[t] = lrp_linear(self.gates_Right[t, idx_i] * self.gates_Right[t, idx_g], np.identity(d),
                                     np.zeros((d)), self.c_Right[t], Rc_Right[t], 2 * d, eps, bias_factor, debug=False)
            Rx_rev[t] = lrp_linear(self.x_rev[t], self.Wxh_Right[idx_g].T,
                                   self.bxh_Right[idx_g] + self.bhh_Right[idx_g], self.gates_pre_Right[t, idx_g],
                                   Rg_Right[t], d + e, eps, bias_factor, debug=False)
            Rh_Right[t - 1] = lrp_linear(self.h_Right[t - 1], self.Whh_Right[idx_g].T,
                                         self.bxh_Right[idx_g] + self.bhh_Right[idx_g], self.gates_pre_Right[t, idx_g],
                                         Rg_Right[t], d + e, eps, bias_factor, debug=False)

        return Rx, Rx_rev[::-1, :], Rh_Left[-1].sum()+Rc_Left[-1].sum()+Rh_Right[-1].sum()+Rc_Right[-1].sum()


class LSTM_unodi:
    def __init__(self, model,
                 device):
        """
        Load trained model and set the weights and bias.
        For one direction
        (Only for one layer LSTM right now)
        """
        ## Pytorch model
        self.model = model
        self.device = device

        # LSTM left encoder
        self.Wxh_Left = self.model.rnn.weight_ih_l0.data.cpu().detach().numpy()  # shape 4d*e [4*Hidden, Input]
        self.bxh_Left = self.model.rnn.bias_ih_l0.data.cpu().detach().numpy()  # shape 4d
        self.Whh_Left = self.model.rnn.weight_hh_l0.data.cpu().detach().numpy()  # shape 4d*d [4*Hidden, Hidden]
        self.bhh_Left = self.model.rnn.bias_hh_l0.data.cpu().detach().numpy()  # shape 4d
        # # LSTM right encoder
        # self.Wxh_Right = self.model.rnn.weight_ih_l0_reverse.data.cpu().detach().numpy()  # shape 4d*e [4*Hidden, Input]
        # self.bxh_Right = self.model.rnn.bias_ih_l0_reverse.data.cpu().detach().numpy()  # shape 4d
        # self.Whh_Right = self.model.rnn.weight_hh_l0_reverse.data.cpu().detach().numpy()  # shape 4d*d [4*Hidden, Hidden]
        # self.bhh_Right = self.model.rnn.bias_hh_l0_reverse.data.cpu().detach().numpy()  # shape 4d

        # linear output layer
        self.Why_Left = self.model.fc.weight.data.cpu().detach().numpy()  # shape C*d [Class, Hidden]
        self.bhy_Left = self.model.fc.bias.data.cpu().detach().numpy() # Shape C [Class]
        # self.Why_Right = model["Why_Right"]  # shape C*d

    def set_input(self, sample: Tensor,
                  delete_pos=None):
        """
        Build the numerical input sequence x/x_rev from the word indices(Sample) w (+ initialize hidden layers h, c).
        Optionally delete words at positions delete_pos.
        """

        T = sample.shape[-1]  # sequence length
        d = int(self.Wxh_Left.shape[0] / 4)  # hidden layer dimension
        e = sample.shape[1]  ## input sequence channel (feature sensor)
        # x = np.zeros((T, e))
        x = sample.cpu().detach().numpy()[0].T

        if delete_pos is not None:
            x[delete_pos, :] = np.zeros((len(delete_pos), e))

        self.w = sample
        self.x = x
        # self.x_rev = x[::-1, :].copy()

        self.h_Left = np.zeros((T + 1, d))
        self.c_Left = np.zeros((T + 1, d))
        # self.h_Right = np.zeros((T + 1, d))
        # self.c_Right = np.zeros((T + 1, d))

    def forward(self):
        """
        Standard forward pass.
        Compute the hidden layer values (assuming input x/x_rev was previously set)
        """
        T = self.w.shape[-1]
        d = int(self.Wxh_Left.shape[0] / 4) ## hidden size
        # gate indices (assuming the gate ordering in the LSTM weights is i,f,g,o):
        idx = np.hstack((np.arange(0, 2 * d), np.arange(3 * d, 4 * d))).astype(int)  # indices of gates i,f,o together
        idx_i, idx_f, idx_g, idx_o = np.arange(0, d), np.arange(d, 2 * d), np.arange(2 * d, 3 * d), np.arange(3 * d,
                                                                                                              4 * d)  # indices of gates i,f,g,o separately

        # initialize
        self.gates_xh_Left = np.zeros((T, 4 * d))
        self.gates_hh_Left = np.zeros((T, 4 * d))
        self.gates_pre_Left = np.zeros((T, 4 * d))  # gates pre-activation
        self.gates_Left = np.zeros((T, 4 * d))  # gates activation

        # self.gates_xh_Right = np.zeros((T, 4 * d))
        # self.gates_hh_Right = np.zeros((T, 4 * d))
        # self.gates_pre_Right = np.zeros((T, 4 * d))
        # self.gates_Right = np.zeros((T, 4 * d))

        for t in range(T):
            self.gates_xh_Left[t] = np.dot(self.Wxh_Left, self.x[t])
            self.gates_hh_Left[t] = np.dot(self.Whh_Left, self.h_Left[t - 1])
            self.gates_pre_Left[t] = self.gates_xh_Left[t] + self.gates_hh_Left[t] + self.bxh_Left + self.bhh_Left
            self.gates_Left[t, idx] = 1.0 / (1.0 + np.exp(- self.gates_pre_Left[t, idx]))
            self.gates_Left[t, idx_g] = np.tanh(self.gates_pre_Left[t, idx_g])
            self.c_Left[t] = self.gates_Left[t, idx_f] * self.c_Left[t - 1] + self.gates_Left[t, idx_i] * \
                             self.gates_Left[t, idx_g]
            self.h_Left[t] = self.gates_Left[t, idx_o] * np.tanh(self.c_Left[t])

            # self.gates_xh_Right[t] = np.dot(self.Wxh_Right, self.x_rev[t])
            # self.gates_hh_Right[t] = np.dot(self.Whh_Right, self.h_Right[t - 1])
            # self.gates_pre_Right[t] = self.gates_xh_Right[t] + self.gates_hh_Right[t] + self.bxh_Right + self.bhh_Right
            # self.gates_Right[t, idx] = 1.0 / (1.0 + np.exp(- self.gates_pre_Right[t, idx]))
            # self.gates_Right[t, idx_g] = np.tanh(self.gates_pre_Right[t, idx_g])
            # self.c_Right[t] = self.gates_Right[t, idx_f] * self.c_Right[t - 1] + self.gates_Right[t, idx_i] * \
            #                   self.gates_Right[t, idx_g]
            # self.h_Right[t] = self.gates_Right[t, idx_o] * np.tanh(self.c_Right[t])

        self.y_Left = np.dot(self.Why_Left, self.h_Left[T - 1]) + self.bhy_Left
        # self.y_Right = np.dot(self.Why_Right, self.h_Right[T - 1])
        # self.s = self.y_Left + self.y_Right
        self.s = self.y_Left

        return self.s.copy()  # prediction scores

    def backward(self, w,
                 sensitivity_class):
        """
        Standard gradient backpropagation backward pass.
        Compute the hidden layer gradients by backpropagating a gradient of 1.0 for the class sensitivity_class

        w: sample
        sensitivity_class: the label (the class to be explained)
        """
        # forward pass
        self.set_input(w)
        self.forward()

        T = w.shape[-1]
        d = int(self.Wxh_Left.shape[0] / 4) ## hidden size
        C = self.Why_Left.shape[0]  # number of classes
        # gate indices (assuming the gate ordering in the LSTM weights is i,f,g,o):
        idx = np.hstack((np.arange(0, 2 * d), np.arange(3 * d, 4 * d))).astype(int)  # indices of gates i,f,o together
        idx_i, idx_f, idx_g, idx_o = np.arange(0, d), np.arange(d, 2 * d), np.arange(2 * d, 3 * d), np.arange(3 * d,
                                                                                                              4 * d)  # indices of gates i,f,g,o separately

        # initialize
        self.dx = np.zeros(self.x.shape)
        # self.dx_rev = np.zeros(self.x.shape)

        self.dh_Left = np.zeros((T + 1, d))
        self.dc_Left = np.zeros((T + 1, d))
        self.dgates_pre_Left = np.zeros((T, 4 * d))  # gates pre-activation
        self.dgates_Left = np.zeros((T, 4 * d))  # gates activation

        # self.dh_Right = np.zeros((T + 1, d))
        # self.dc_Right = np.zeros((T + 1, d))
        # self.dgates_pre_Right = np.zeros((T, 4 * d))
        # self.dgates_Right = np.zeros((T, 4 * d))

        ds = np.zeros((C))
        ds[sensitivity_class] = 1.0
        dy_Left = ds.copy()
        # dy_Right = ds.copy()

        self.dh_Left[T - 1] = np.dot(self.Why_Left.T, dy_Left)
        # self.dh_Right[T - 1] = np.dot(self.Why_Right.T, dy_Right)

        for t in reversed(range(T)):
            self.dgates_Left[t, idx_o] = self.dh_Left[t] * np.tanh(self.c_Left[t])  # do[t]
            self.dc_Left[t] += self.dh_Left[t] * self.gates_Left[t, idx_o] * (
                        1. - (np.tanh(self.c_Left[t])) ** 2)  # dc[t]
            self.dgates_Left[t, idx_f] = self.dc_Left[t] * self.c_Left[t - 1]  # df[t]
            self.dc_Left[t - 1] = self.dc_Left[t] * self.gates_Left[t, idx_f]  # dc[t-1]
            self.dgates_Left[t, idx_i] = self.dc_Left[t] * self.gates_Left[t, idx_g]  # di[t]
            self.dgates_Left[t, idx_g] = self.dc_Left[t] * self.gates_Left[t, idx_i]  # dg[t]
            self.dgates_pre_Left[t, idx] = self.dgates_Left[t, idx] * self.gates_Left[t, idx] * (
                        1.0 - self.gates_Left[t, idx])  # d ifo pre[t]
            self.dgates_pre_Left[t, idx_g] = self.dgates_Left[t, idx_g] * (
                        1. - (self.gates_Left[t, idx_g]) ** 2)  # d g pre[t]
            self.dh_Left[t - 1] = np.dot(self.Whh_Left.T, self.dgates_pre_Left[t])
            self.dx[t] = np.dot(self.Wxh_Left.T, self.dgates_pre_Left[t])

            # self.dgates_Right[t, idx_o] = self.dh_Right[t] * np.tanh(self.c_Right[t])
            # self.dc_Right[t] += self.dh_Right[t] * self.gates_Right[t, idx_o] * (1. - (np.tanh(self.c_Right[t])) ** 2)
            # self.dgates_Right[t, idx_f] = self.dc_Right[t] * self.c_Right[t - 1]
            # self.dc_Right[t - 1] = self.dc_Right[t] * self.gates_Right[t, idx_f]
            # self.dgates_Right[t, idx_i] = self.dc_Right[t] * self.gates_Right[t, idx_g]
            # self.dgates_Right[t, idx_g] = self.dc_Right[t] * self.gates_Right[t, idx_i]
            # self.dgates_pre_Right[t, idx] = self.dgates_Right[t, idx] * self.gates_Right[t, idx] * (
            #             1.0 - self.gates_Right[t, idx])
            # self.dgates_pre_Right[t, idx_g] = self.dgates_Right[t, idx_g] * (1. - (self.gates_Right[t, idx_g]) ** 2)
            # self.dh_Right[t - 1] = np.dot(self.Whh_Right.T, self.dgates_pre_Right[t])
            # self.dx_rev[t] = np.dot(self.Wxh_Right.T, self.dgates_pre_Right[t])

        # return self.dx.copy(), self.dx_rev[::-1, :].copy()
        return self.dx.copy()

    def lrp(self, sample, LRP_class, eps=0.001, bias_factor=0.0):
        """
        Layer-wise Relevance Propagation (LRP) backward pass.
        Compute the hidden layer relevances by performing LRP for the target class LRP_class
        (according to the papers:
            - https://doi.org/10.1371/journal.pone.0130140
            - https://doi.org/10.18653/v1/W17-5221 )
        """
        # forward pass
        self.set_input(sample)
        self.forward()

        T = sample.shape[-1]
        d = int(self.Wxh_Left.shape[0] / 4)
        e = int(sample.shape[1])
        C = self.Why_Left.shape[0]  # number of classes

        # gate indices (assuming the gate ordering in the LSTM weights is i,f,g,o):
        idx = np.hstack((np.arange(0, 2 * d), np.arange(3 * d, 4 * d))).astype(int)  # indices of gates i,f,o together
        idx_i, idx_f, idx_g, idx_o = np.arange(0, d), np.arange(d, 2 * d), np.arange(2 * d, 3 * d), np.arange(3 * d,
                                                                                                              4 * d)  # indices of gates i,f,g,o separately

        # initialize
        Rx = np.zeros(self.x.shape)
        # Rx_rev = np.zeros(self.x.shape)

        Rh_Left = np.zeros((T + 1, d))
        Rc_Left = np.zeros((T + 1, d))
        Rg_Left = np.zeros((T, d))  # gate g only
        # Rh_Right = np.zeros((T + 1, d))
        # Rc_Right = np.zeros((T + 1, d))
        # Rg_Right = np.zeros((T, d))  # gate g only

        Rout_mask = np.zeros((C))
        Rout_mask[LRP_class] = 1.0

        # format reminder: lrp_linear(hin, w, b, hout, Rout, bias_nb_units, eps, bias_factor)
        Rh_Left[T - 1] = lrp_linear(self.h_Left[T - 1], self.Why_Left.T, self.bhy_Left, self.s, self.s * Rout_mask,
                                    d, eps, bias_factor, debug=False)
        # Rh_Right[T - 1] = lrp_linear(self.h_Right[T - 1], self.Why_Right.T, np.zeros((C)), self.s, self.s * Rout_mask,
        #                              d, eps, bias_factor, debug=False)

        for t in reversed(range(T)):
            Rc_Left[t] += Rh_Left[t]
            Rc_Left[t - 1] = lrp_linear(self.gates_Left[t, idx_f] * self.c_Left[t - 1], np.identity(d), np.zeros((d)),
                                        self.c_Left[t], Rc_Left[t], 2 * d, eps, bias_factor, debug=False)
            Rg_Left[t] = lrp_linear(self.gates_Left[t, idx_i] * self.gates_Left[t, idx_g], np.identity(d),
                                    np.zeros((d)), self.c_Left[t], Rc_Left[t], 2 * d, eps, bias_factor, debug=False)
            Rx[t] = lrp_linear(self.x[t], self.Wxh_Left[idx_g].T, self.bxh_Left[idx_g] + self.bhh_Left[idx_g],
                               self.gates_pre_Left[t, idx_g], Rg_Left[t], d + e, eps, bias_factor, debug=False)
            Rh_Left[t - 1] = lrp_linear(self.h_Left[t - 1], self.Whh_Left[idx_g].T,
                                        self.bxh_Left[idx_g] + self.bhh_Left[idx_g], self.gates_pre_Left[t, idx_g],
                                        Rg_Left[t], d + e, eps, bias_factor, debug=False)

            # Rc_Right[t] += Rh_Right[t]
            # Rc_Right[t - 1] = lrp_linear(self.gates_Right[t, idx_f] * self.c_Right[t - 1], np.identity(d),
            #                              np.zeros((d)), self.c_Right[t], Rc_Right[t], 2 * d, eps, bias_factor,
            #                              debug=False)
            # Rg_Right[t] = lrp_linear(self.gates_Right[t, idx_i] * self.gates_Right[t, idx_g], np.identity(d),
            #                          np.zeros((d)), self.c_Right[t], Rc_Right[t], 2 * d, eps, bias_factor, debug=False)
            # Rx_rev[t] = lrp_linear(self.x_rev[t], self.Wxh_Right[idx_g].T,
            #                        self.bxh_Right[idx_g] + self.bhh_Right[idx_g], self.gates_pre_Right[t, idx_g],
            #                        Rg_Right[t], d + e, eps, bias_factor, debug=False)
            # Rh_Right[t - 1] = lrp_linear(self.h_Right[t - 1], self.Whh_Right[idx_g].T,
            #                              self.bxh_Right[idx_g] + self.bhh_Right[idx_g], self.gates_pre_Right[t, idx_g],
            #                              Rg_Right[t], d + e, eps, bias_factor, debug=False)

        return Rx, Rh_Left[-1].sum() + Rc_Left[-1].sum()

        # return Rx, Rx_rev[::-1, :], Rh_Left[-1].sum() + Rc_Left[-1].sum() + Rh_Right[-1].sum() + Rc_Right[-1].sum()
