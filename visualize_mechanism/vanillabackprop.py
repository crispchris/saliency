"""Source: https://github.com/utkuozbulak/pytorch-cnn-visualizations/blob/master/src/vanilla_backprop.py
    Produces gradients generated with vanilla back propagation
    Notes: Use the (unnormalised) class scores Sc, not returned by the soft-max layer Pc
    Map: Mij = | Wh(i,j) |, where h(i, j) is the index of the element of w
    --> Mij = max_c | Wh(i,j, c)|, c is channels
"""
### -------------------
### --- Third-Party ---
### -------------------
import torch as t
import numpy as np

### -----------
### --- Own ---
### -----------
from utils import load_model

class _GradientBasicWrapper:
    def __init__(self, model, checkpoint: str, use_cuda = True, used_normalized = True):
        """
        Parameters
        ----------
        model: Pytorch model
        checkpoint: path to the checkpoint(ckp)
        use_cuda: use GPU or not
        used_normalized: use min-max normalization
        """
        super(_GradientBasicWrapper, self).__init__()
        self.device = t.device('cuda' if use_cuda else 'cpu')
        self.model = model
        self.gradients = None
        self.used_normalized = used_normalized
        self.model = load_model(self.model, ckp_path=checkpoint, use_cuda=use_cuda)
        ## Get all layers in the model
        self.get_layers()

    def get_layers(self):
        """
        Get layers in the model
        Returns
        -------
            Layers of the model in a List
        """
        self.layers = []
        for module in self.model.modules():
            add = True
            for layer in module.modules():
                if isinstance(layer, t.nn.Sequential):
                    add = False
            if add:
                self.layers.append(module)

    def forward_pass(self, X:t.Tensor):
        """X should be the input sample [B, C, L]
           Returns:
               forward pass of each layer
        """
        ## Propagate the input
        L = len(self.layers)
        ## A_forward[0] is the input
        self.A_forward = [X] + [X] * L  ## Create a List to store the activation produced by each layer
        for num, layer in enumerate(self.layers):
            for l in layer.modules():
                if isinstance(l, t.nn.Linear):  ## For linear layer, the shape should change
                    self.A_forward[num] = self.A_forward[num].transpose(1, 2)
            self.A_forward[num + 1] = layer.forward(self.A_forward[num])
            if isinstance(l, t.nn.Linear):
                self.A_forward[num + 1] = self.A_forward[num + 1].transpose(1, 2)

    def generate_gradients(self, X:t.Tensor, target_class:int = None):
        raise NotImplementedError

class VanillaBackprop(_GradientBasicWrapper):
    """
        Gradient Saliency method for visualization using single back-propagation pass
    """
    def __init__(self, model, checkpoint: str, use_cuda=True, used_normalized=True):
        """
        Parameters
        ----------
        model: Pytorch model
        checkpoint: path to the checkpoint(.ckp)
        use_cuda: use GPU or not
        used_normalized: use min-max normalization
        """
        super().__init__(model=model, checkpoint=checkpoint, use_cuda=use_cuda,
                         used_normalized=used_normalized)

    def generate_gradients(self, X:t.Tensor, target_class:int = None):
        """
        Parameters
        ----------
        X: input sample, Tensor
        target_class: the class to be visualized

        Returns
        -------
        gradients
        target_class: the class to be visualized
        """
        X = X.requires_grad_(True)
        prediction = self.model(X)
        if target_class is None:
            target_class = np.argmax(prediction.cpu().detach().numpy()[0])

        ## compute one hot
        one_hot = t.zeros(prediction.shape).to(self.device)
        one_hot[0][target_class] = 1
        one_hot = one_hot * prediction

        self.model.zero_grad()
        prediction.backward(one_hot, retain_graph=True)

        gradients = t.autograd.grad(t.sum(one_hot), inputs=X)[0].cpu().data.numpy()
        gradients = np.abs(gradients)
        if self.used_normalized:
            print("[Class-Map] --- Min-Max Normalization ---")
            gradients = gradients[0]  ## [C, L]
            gradients = (gradients - np.min(gradients, axis=-1)[0]) / (
                        np.max(gradients, axis=-1)[0] - np.min(gradients, axis=-1)[0])
            gradients *= 100
        return gradients, target_class


