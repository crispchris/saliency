"""LRP, source from https://giorgiomorales.github.io/Layer-wise-Relevance-Propagation-in-Pytorch/
"""
## -------------------
## --- Third-Party ---
## -------------------
import torch as t
import numpy as np
from copy import deepcopy

## -----------
## --- Own ---
## -----------
from models.tcn import Chomp1d

device = t.device('cuda' if t.cuda.is_available() else 'cpu')
class LRP_individual:
    def __init__(self, model,
                 mean: float,
                 std: float,
                 epsilon: float = 0.25,
                 gamma: float = 0.25,
                 rule = 'epsilon',
                 use_cuda = True):
        """
        Layer-Wise individual Computation of LRP method (Gamma and Epsilon)

        Parameters
        ----------
        model: Pytorch model
        mean: mean of the test set
        std: std of the test set
        epsilon (float) : the parameter used for LRP epsilon rule
        gamma (float) : the parameter for LRP Gamma rule
        rule (str) : either 'epsilon' or 'gamma'
        use_cuda: use GPU or not
        Returns
        -------
        LRP heatmap
        """
        self.model = model
        self.mean = mean
        self.std = std
        self.epsilon = epsilon
        self.gamma = gamma
        self.rule = rule
        self.use_cuda = use_cuda


        ## layers
        self.layers = []
        self.visualisation = {}
        # self._get_layers(self.model)
        res = self.get_layers(self.model)
        self.get_sequential(res)

    def forward_pass(self,
                     X,
                     target_class):

        ## Propagate the input
        ## Forward Pass
        L = len(self.layers)
        A_forward = [X] + [X] * L ## Create a List to store the activation produced by each layer
        for num, layer in enumerate(self.layers):
            for l in layer.modules():
                if isinstance(l, t.nn.Linear): ## For linear layer, the shape should change
                    A_forward[num] = A_forward[num][:, :, -1] ## last step model
            A_forward[num + 1] = layer.forward(A_forward[num])
            # if isinstance(l, t.nn.Linear):
            #     A_forward[num + 1] = A_forward[num + 1].transpose(1, 2)

        ## Get the relevance score of the last layer (the highest classification score)
        prediction = A_forward[-1].cpu().detach().numpy()[0]
        predict = A_forward[-1].cpu().detach().numpy()[0]
        if target_class is None:
            target_class = np.argmax(prediction)
        prediction = np.abs(prediction) * 0
        prediction[target_class] = 1
        prediction = t.FloatTensor(prediction)

        # Create the list of relevances with (L + 1) elements and assign the value of the last one
        self.R = [None] * L + [(A_forward[-1].cpu() * prediction).data + 1e-6]
        ## prediction should be as same as model.forward(X)
        return self.R, A_forward ## prediction to check whether forward pass is correct

    def backward_pass(self, A_forward):
        ## Propagation procedure from the top-layer towards the lower layers
        ## Backward Pass to compute the Relevance
        L = len(self.layers)
        for layer in range(1, L)[::-1]:
            if isinstance(self.layers[layer], t.nn.Conv1d) or isinstance(self.layers[layer], t.nn.Linear) \
                    or isinstance(self.layers[layer], t.nn.BatchNorm1d):
                # or isinstance(self.layers[layer], t.nn.AdaptiveMaxPool1d):

                ## Specifies the rho function that will be applied to the weights of the layer
                ## here follow the rule
                ## Last layer (basic rule (LRP-0)
                if layer == L - 1:
                    rho = lambda p: p
                    incr = lambda z: z + 1e-9
                elif self.rule is "epsilon":
                    rho = lambda p: p
                    incr = lambda z: z + 1e-9 + self.epsilon * ((z**2).mean()**.5).data
                elif self.rule is "gamma":
                    rho = lambda p: p + self.gamma * p.clamp(min=0)
                    incr = lambda z: z + 1e-9
                # if 0 < layer < L - 2:
                #     rho = lambda p: p + 0.25 * p.clamp(min=0) ## gamma(0.25)


                A_forward[layer] = A_forward[layer].data.requires_grad_(True)
                # Transform the weights of the layer and executes a forward pass
                # if isinstance(self.layers[layer], t.nn.AdaptiveMaxPool1d):
                #     z = self.epsilon + self.layers[layer].forward(A_forward[layer])
                z = incr(newlayer(self.layers[layer], rho).forward(A_forward[layer]))

                # if isinstance(layers[layer], t.nn.Linear):
                #     z = z.transpose(1, 2)
                # element-wise division between the relevance of the next layer and the denominator
                s = (self.R[layer + 1].to(device) / z).data
                # Calculate the gradients and multiply it by the activation layer
                (z * s).sum().backward()
                c = A_forward[layer].grad
                self.R[layer] = (A_forward[layer] * c).cpu().data
            else:
                self.R[layer] = self.R[layer + 1]

        ## First Layer
        A_forward[0] = A_forward[0].data.requires_grad_(True)
        lb = (A_forward[0].data * 0 + (0 - self.mean) / self.std).requires_grad_(True)
        hb = (A_forward[0].data * 0 + (1 - self.mean) / self.std).requires_grad_(True)
        z = self.layers[0].forward(A_forward[0]) + 1e-9
        z -= newlayer(self.layers[0], lambda p: p.clamp(min=0)).forward(lb)  # step 1 (b)
        z -= newlayer(self.layers[0], lambda p: p.clamp(max=0)).forward(hb)  # step 1 (c)
        s = (self.R[1].to(device) / z).data  # step 2
        (z * s).sum().backward()
        c, cp, cm = A_forward[0].grad, lb.grad, hb.grad  # step 3
        self.R[0] = (A_forward[0] * c + lb * cp + hb * cm).data

        ## just for debugging, to see if the sum of relevances in each layer is the
        print("[Debug] --- Begin ---")
        print("[Debug] see the sum of relevance scores in each layer ")
        for layer in range(0, L):
            print(np.sum(self.R[layer].data.cpu().detach().numpy()))
        print("[Debug] --- Done ---")

        R = self.R[0].data.cpu().detach().numpy()
        ## Return the relevance of the input layer
        return R[0]

    def get_layers(self, model):
        ## Get the list of layers of the network
        # layers = []
        layers = [self.get_layers(module) for module in model.children()]
        res = [model]
        for c in layers:
            res += c
        return res
    def get_sequential(self, res):
        layers = res[0].layers
        for layer in layers:
            for l in layer.modules():
                if isinstance(l, t.nn.Sequential):
                    for le in l.modules():
                        if type(le) in SUPPORTED_LAYERS_LIST:
                            self.layers.append(le)


def newlayer(layer, g):
    """Clone a layer and pass its parameters through the function g"""
    layer = deepcopy(layer)
    layer.weight = t.nn.Parameter(g(layer.weight))
    layer.bias = t.nn.Parameter(g(layer.bias))
    return layer

SUPPORTED_LAYERS_LIST = [t.nn.Conv1d,
                         t.nn.BatchNorm1d,
                         t.nn.AdaptiveMaxPool1d,
                         t.nn.ReLU,
                         t.nn.Dropout,
                         t.nn.Linear,
                         Chomp1d]