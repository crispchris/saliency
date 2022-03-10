"""Origin Paper: https://arxiv.org/abs/1706.03825
    SmoothGrad: add noise to smooth the noise of gradients
"""
### -------------------
### --- Third-Party ---
### -------------------
import torch as t
import numpy as np

## -----------
## --- Own ---
## -----------
from visualize_mechanism.vanillabackprop import VanillaBackprop, _GradientBasicWrapper

class SmoothGrad(_GradientBasicWrapper):
    """
    Take random samples in a neighborhood of an input x, and average the resulting sensitivty maps
    Parameters: n (the number of samples), Noise: Gaussian noise with standard deviation
    """
    def __init__(self, model, checkpoint: str, noise_sigma: float, num_samples: int = 100, use_cuda = True, used_normalized = True):
        """
        Parameters
        ----------
        model: Pytorch model
        checkpoint: path to the checkpoint(ckp)
        noise_sigma: standard deviation for gaussian normal distribution (Noise)
        num_samples: number of samples for generating noise
        use_cuda: use GPU or not
        used_normalized: use min-max normalization
        """
        super().__init__(model=model, checkpoint=checkpoint, use_cuda=use_cuda, used_normalized=used_normalized)
        self.explainer = VanillaBackprop(model= self.model, checkpoint=checkpoint, use_cuda=use_cuda,
                                         used_normalized=False)
        self.noise_sigma = noise_sigma
        self.num_samples = num_samples

    def generate_gradients(self, X:t.Tensor, target_class: int = None):
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
        noise_level = self.noise_sigma / (t.max(X.data) - t.min(X.data))
        if target_class is None:
            origin_gradients, target_class = self.explainer.generate_gradients(X)

        average_gradients = np.zeros(X.size()) ## [B, C, L]
        for num in range(self.num_samples):
            gaussion_noise = t.randn(X.size()).normal_(mean=0, std= noise_level**2).to(self.device)
            new_input = X + gaussion_noise
            gradients, _ = self.explainer.generate_gradients(new_input, target_class=target_class)

            ## Add the gradients from each sample
            average_gradients += gradients
        average_gradients /= self.num_samples

        if self.used_normalized:
            print("[Class-Map] --- Min-Max Normalization ---")
            average_gradients = average_gradients[0] ## [C, L]
            average_gradients = (average_gradients - np.min(average_gradients, axis=-1)[0])/(
                                np.max(average_gradients, axis=-1)[0] - np.min(average_gradients, axis=-1)[0])
            average_gradients *= 100
        return average_gradients, target_class



