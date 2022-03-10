"""Source: https://github.com/utkuozbulak/pytorch-cnn-visualizations/blob/master/src/integrated_gradients.py
    Produces heatmap generated with Integrated Gradients
"""
### -------------------
### --- Third-Party ---
### -------------------
import torch as t
import numpy as np

### -----------
### --- Own ---
### -----------
from visualize_mechanism.vanillabackprop import _GradientBasicWrapper

class IntegratedGradients(_GradientBasicWrapper):
    """
        Gradient Saliency method for visualization using single back-propagation pass
    """
    def __init__(self, model, checkpoint: str, use_cuda = True, used_normalized = True):
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
    def generate_gradients(self, X:t.Tensor, target_class:int = None, steps:int = 100):
        return self.generate_integrated_gradients(X=X, target_class=target_class, steps=steps)

    def generate_samples_on_linear_path(self, X, steps):
        ## Generate uniform numbers between 0 and steps
        step_list = np.linspace(0, 1, num=steps)
        ## Generate scaled data
        self.sample_list = [X * step for step in step_list]

    def compute_gradients(self, X:t.Tensor, target_class: int = None):
        """
        Parameters
        ----------
        X: input sample, Tensor
        target_class: the class to be visualized

        Returns
        -------
        gradients
        """
        X = X.requires_grad_(True)
        ## Forward pass
        prediction = self.model(X)
        ## compute one hot
        one_hot = t.zeros(prediction.shape).to(self.device)
        one_hot[0][target_class] = 1
        one_hot = one_hot * prediction
        ## model zero grad
        self.model.zero_grad()
        ## Backward pass
        prediction.backward(one_hot, retain_graph=True)
        ## Gradients computation
        gradients = t.autograd.grad(t.sum(one_hot), inputs=X)[0].cpu().data.numpy()
        gradients = np.abs(gradients)
        return gradients

    def generate_integrated_gradients(self, X, target_class: int = None, steps: int = 100):
        ## Forward pass
        prediction = self.model(X)
        if target_class is None:
            target_class = np.argmax(prediction.cpu().detach().numpy()[0])

        # Generate Samples list
        self.generate_samples_on_linear_path(X, steps=steps)
        ## Initialize a integrated gradients composed of zeros
        self.gradients = t.zeros(X.size()).cpu().data.numpy()

        for sample in self.sample_list:
            # Gradients compute for sample
            gradients = self.compute_gradients(sample, target_class=target_class)
            self.gradients += gradients/steps
        heatmap = (X.cpu().data.numpy() * self.gradients) ## [B, C, L]

        if self.used_normalized:
            print("[Class-Map] --- Min-Max Normalization ---")
            heatmap = heatmap[0] ## [C, L]
            heatmap = (heatmap - np.min(heatmap, axis=-1)[0]) / (
                        np.max(heatmap, axis=-1)[0] - np.min(heatmap, axis=-1)[0])
            heatmap *= 100

        return heatmap, target_class