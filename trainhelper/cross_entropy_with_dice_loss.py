import os
import sys
sys.path.append('..')
from trainhelper.dice_loss import GDiceLossV2
import torch
from torch.nn import CrossEntropyLoss
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropy_GDice_Loss(nn.Module):
    def __init__(self, apply_nonlin=None, smooth=1e-5, cross_en_weight: float =0.5):
        super(CrossEntropy_GDice_Loss, self).__init__()

        self.gdice = GDiceLossV2(apply_nonlin=apply_nonlin,
                                 smooth=smooth)
        self.cross_entropy = CrossEntropyLoss()
        self.cross_en_weight = cross_en_weight

    def forward(self, net_output, gt):
        cross_ent_loss = self.cross_entropy(net_output, gt)

        pred = F.softmax(net_output, dim=1)
        gdice_loss = self.gdice(pred, gt)

        loss = cross_ent_loss * self.cross_en_weight + gdice_loss * (1 - self.cross_en_weight)
        return loss