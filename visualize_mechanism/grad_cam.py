"""
"Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization
https://arxiv.org/pdf/1610.02391.pdf
also refer to: https://github.com/kazuto1011/grad-cam-pytorch/blob/fd10ff7fc85ae064938531235a5dd3889ca46fed/grad_cam.py#L94

"""

## -------------------
## --- Third-party ---
## -------------------
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

## Device Configuration
device = t.device('cuda' if t.cuda.is_available() else 'cpu')

class _BaseWrapper:
    """
    To do stuff like Forward, Backward
    """
    def __init__(self, model):
        super(_BaseWrapper, self).__init__()
        self.model = model
        self.handlers = []

    def _encode_one_hot(self, class_idx):
        one_hot = t.zeros(self.output.shape).to(device)
        one_hot[0][class_idx] = 1
        one_hot = t.sum(one_hot * self.output)
        return one_hot

    def forward(self, sample: Optional[t.Tensor]):
        self.output = self.model(sample)
        self.prob = F.softmax(self.output, dim=1)
        return self.output, self.prob.sort(dim=1, descending=True)  # ordered results

    def backward(self, class_idx: int):
        """Class-Specific Backpropagation"""
        one_hot = self._encode_one_hot(class_idx)
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

    def generate(self, target_layer):
        raise NotImplementedError
    def remove_hook(self):
        """Remove all the forward/backward hook functions"""
        for handle in self.handlers:
            handle.remove()

class Grad_Cam(_BaseWrapper):
    """
    Compute the GradCAM
    L_c(Saliency Map for Class C) = Sum[ weight_k(kernel) for Class C * A(feature map) of kernel k]
    weight_k(kernel) for Class C = [1/Z] * sum[sum[derivative[ Y(Class Score)] / derivative[ A(feature map) of kernel k]
    weight_k for Class C is certainly the gradient of the score for class C(before the softmax), with respect to
    the feature map of the last convolutional layer. These Gradients flowing back are GAP(global average pooled)
    Returns
    -------
    Final Class discriminative Saliency Map
    """
    def __init__(self, model: t.nn.Module, conv_layer: str = None, relu_used: bool = True, norm_used: bool = True):
        """
        Parameters
        ----------
        model: the trained model
        conv_layer: the (last) convolutional layer for the Feature map and Gradient
        relu_used
        """
        super().__init__(model)
        self.model = model
        self.gradients_conv = None
        self.conv_layer = conv_layer
        self.relu_used = relu_used
        self.norm_used = norm_used
        self.fmap_pool = {}
        self.grad_pool = {}

        def save_fmaps(key):
            def forward_hook(module, input, output):
                self.fmap_pool[key] = output.data
            return forward_hook

        def save_grads(key):
            def backward_hook(module, grad_in, grad_out):
                self.grad_pool[key] = grad_out[0] ## ? grad_in??
            return backward_hook

        for name, module in self.model.named_modules():
            if self.conv_layer is None or name in self.conv_layer:
                self.handlers.append(module.register_forward_hook(save_fmaps(name)))
                self.handlers.append(module.register_backward_hook(save_grads(name)))

    def _find(self, pool, target_layer):
        if target_layer in pool.keys():
            return pool[target_layer]
        else:
            raise ValueError(f"Invalid layer name: {target_layer}")


    def generate(self, target_layer):
        fmaps = self._find(self.fmap_pool, target_layer)        ## Shape =[B, C, Len]
        grads = self._find(self.grad_pool, target_layer)        ## Shape =[B, C, Len] should be same as fmaps

        weights = t.mean(grads, -1).reshape(-1, 1)

        gcam = t.mul(fmaps, weights).sum(dim=1, keepdim=True)
        if self.relu_used:
            gcam = F.relu(gcam)
        if self.norm_used:
            gcam = (gcam - t.min(gcam, dim=-1)[0])/(t.max(gcam, dim=-1)[0] - t.min(gcam, dim=-1)[0])
            gcam *= 100

        return gcam

