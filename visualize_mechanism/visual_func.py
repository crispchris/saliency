"""
Visual Interpretation
Backpropagation based and purturbation based

Captum Library: get the saliency maps
Backpropagation based: Gradients, Integrated Gradients, SmoothGrad, GradCAM, (CAM), LRP,
                    Guided Backprop, Guided GradCAM
Perturbation based: LIME, SHAP
"""

## -------------------
## --- Third-Party ---
## -------------------
import sys
sys.path.append('..')
import math
import numpy as np
import torch as t
import torch.nn as nn

from captum.attr import Saliency, NoiseTunnel, LayerGradCam
from captum.attr import LRP, GuidedBackprop, GuidedGradCam
from captum.attr import LayerAttribution
from captum.attr import IntegratedGradients as IG
from captum.attr import Lime, KernelShap
from captum.attr._utils.lrp_rules import GammaRule, EpsilonRule
from captum._utils.models.linear_model import SkLearnRidge, SkLearnLinearRegression
from captum.attr._core.lime import get_exp_kernel_similarity_function

## -----------
## --- Own ---
## -----------
from visualize_mechanism.lrp_4_lstm import LSTM_bidi, LSTM_unodi
from visualize_mechanism.visual_utils import set_rule, AggSegmentationWrapper
from models.unet import Utime

class SaliencyFunctions:
    def __init__(self,
                 model: nn.Module,
                 tsr_saliency = None,
                 device = None
                 ):
        """
        Parameters
        ----------
        model (nn.Module): the model structure with loaded weights
        device: torch CPU or GPU
        """
        self.model = model
        self.tsr_saliency = tsr_saliency
        self.device = device
        ## for segmentation
        self.has_wrapper_seg = False

        ## visual methods
        ## backprop-Based
        self.gradientSaliency = Saliency(self.model.forward)
        self.ig = IG(self.model.forward)
        self.smoothNoise = NoiseTunnel(Saliency(self.model.forward))
        self.lrp = LRP(self.model)
        self.gbp = GuidedBackprop(self.model)

        ## perturbation-Based
        self.kernelshap = KernelShap(self.model.forward)


    def getGradientSaliency(self, inp, label, absolute: bool):
        """
        Original Paper: https://arxiv.org/pdf/1312.6034.pdf
        Parameters
        ----------
        inp (tensor): the input samples
        label (int or List): the labels of the samples
        absolute: bool, return absolute value after compute the gradients, when True
                (default: True)
        Returns
        -------
        gradients
        """
        gradients = self.gradientSaliency.attribute(inputs=inp,
                                                    target=label,
                                                    abs=absolute)
        if self.tsr_saliency is not None:
            grads = gradients[0] * self.tsr_saliency
            if absolute:
                gradients = t.abs(grads)
            gradients = tuple(gradients)
        
        return gradients

    ## Integrated Gradients
    def getIntegratedGradients(self, inp, label,
                               absolute: bool,
                               ig_steps: int = 60,
                               ):
        """
        Original Paper: https://arxiv.org/abs/1703.01365

        Parameters
        ----------
        inp (tensor): the input samples
        label (int or List): the labels of the samples
        absolute: bool, return absolute value after compute the gradients, when True
                (default: True)
        ig_steps: Steps for Integrated Gradients to addition (integral)

        Returns
        -------
        ig_maps
        """
        ig_maps = self.ig.attribute(inputs=inp,
                                    target=label,
                                    baselines=None,
                                    n_steps=ig_steps,
                                    method="riemann_trapezoid")
        if self.tsr_saliency is not None:
            maps = ig_maps[0] * self.tsr_saliency
            if absolute:
                ig_maps = tuple(t.abs(maps))
        if absolute and self.tsr_saliency is None:
            ig_maps = tuple(
                abs(ig) for ig in ig_maps
            )
            # ig_maps = tuple(ig_maps)
        return ig_maps

    ## Smooth Gradients
    def getSmoothGradients(self,
                           inp, label,
                           absolute: bool,
                           nt_samples: int = 60,
                           stdevs: float = 0.2):
        """

        Parameters
        ----------
        inp (tensor): the input samples
        label (int or List): the labels of the samples
        absolute: bool, return absolute value after compute the gradients, when True
                (default: True)
        nt_samples (int) : Number of random samples
        stdevs (float) : standard deviation of Gaussian Noise with zero mean

        Returns
        -------
        smooth_gradients
        """
        smoothgrad_maps = self.smoothNoise.attribute(inputs=inp,
                                                     target=label,
                                                     nt_type="smoothgrad",
                                                     nt_samples=nt_samples,
                                                     stdevs=stdevs)
        if self.tsr_saliency is not None:
            maps = smoothgrad_maps[0] * self.tsr_saliency
            if absolute:
                smoothgrad_maps = tuple(t.abs(maps))
        if absolute and self.tsr_saliency is None:
            smoothgrad_maps = tuple(
                abs(sm) for sm in smoothgrad_maps
            )
        return smoothgrad_maps

    ## LRP
    def getLRP(self,
               inp, label,
               absolute: bool,
               rule: str = "epsilon",
               ):
        """
        Original Paper: https://doi.org/10.1371/journal.pone.0130140

        Epsilon-Rule for middle-layer
        Gamma-Rule for Lower-layer
        First Input-Layer for Z-Rule

        Parameters
        ----------
        inp (tensor): the input samples
        label (int or List): the labels of the samples
        absolute: bool, return absolute value after compute the gradients, when True
                (default: True)
        rule: use a rule for Propagation. (default: Epsilon)
             it can be "epsilon" or "gamma"

        Returns
        -------
        lrp_map
        """
        ## Assign Rules for LRP layer-wise
        if rule is "epsilon":
            supported_layers_rules = set_rule(EpsilonRule)
        elif rule is "gamma":
            supported_layers_rules = set_rule(GammaRule)
        else:
            raise TypeError("Please select propagation rules inherited from Class"
                            "either epsilon or gamma")

        lrp_maps = self.lrp.attribute(inputs=inp,
                                      target=label,
                                      rule_dict=supported_layers_rules,
                                      return_convergence_delta=True
                                      )[0]
        # lrp_maps = t.tensor(lrp_maps, dtype=t.float64)
        if self.tsr_saliency is not None:
            maps = lrp_maps[0] * self.tsr_saliency
            if absolute:
                lrp_maps = tuple(t.abs(maps))
        if absolute and self.tsr_saliency is None:
            lrp_maps = tuple(
                abs(lrp) for lrp in lrp_maps
            )
        return lrp_maps

    ## LRP for LSTM
    def getLRP4LSTM(self,
                    inp, label,
                    absolute: bool,
                    eps: float = 0.001,
                    bias_factor: float = 0.0
                    ):
        """
        LRP (Layer-Wise relevance propagation) for LSTM

        Parameters
        ----------
        inp (tensor): the input samples
        label (int or List): the labels of the samples
        absolute: bool, return absolute value after compute the gradients, when True
                (default: True)
        eps (float): epsilon propagation, default: 0.001
        bias_factor (float): default: 0.0, for the bias factor in LRP

        Returns
        -------
        Rx
        """
        if type(inp) == tuple:
            inp = inp[0]
        if self.model.num_directions == 2:
            lrp_lstm = LSTM_bidi(model=self.model,
                                 device=self.device)
            Rx, Rx_rev, R_rest = lrp_lstm.lrp(sample=inp,
                                              LRP_class=label,
                                              eps=eps,
                                              bias_factor=bias_factor)
            Rx = Rx + Rx_rev
        elif self.model.num_directions == 1:
            lrp_lstm = LSTM_unodi(model=self.model,
                             device=self.device)
            Rx, R_rest = lrp_lstm.lrp(sample=inp,
                                    LRP_class=label,
                                    eps=eps,
                                    bias_factor=bias_factor)
        Rx = Rx.T
        Rx = t.tensor(Rx).reshape(1, Rx.shape[0], Rx.shape[-1])
        # Rx = Rx.to(self.device)
        if self.tsr_saliency is not None:
            Rx = Rx.to(self.device) * self.tsr_saliency
        if absolute:
            Rx = t.abs(Rx)
        return Rx

    ## Grad-CAM
    def getGradCAM(self,
                   inp,
                   label,
                   absolute: bool,
                   use_relu:bool = False,
                   upsample_to_input:bool = True,
                   layer_to_grad:str = "gap_softmax.conv1",
                   attribute_to_layer_input: bool = True,
                   ):
        """
        Original Paper: https://arxiv.org/pdf/1610.02391.pdf

        Parameters
        ----------
        inp (tensor): the input samples
        label (int or List): the labels of the samples
        absolute: bool, return absolute value after compute the gradients, when True
                (default: True)
        use_relu: bool, Relu after compute the GradCAM when True
                (default: False)
        upsample_to_input: bool, Whether upsample to input (interpolation)
        layer_to_grad: str, the layer to compute the gradients
        attribute_to_layer_input: bool, Indicates whether to compute the attribution
                                with respect to the layer input or output in LayerGradCam.
                                If attribute_to_layer_input is set to True
                                then the attributions will be computed with respect to layer inputs,
                                otherwise it will be computed with respect to layer outputs
        Returns
        -------
        gradcam_map
        """
        ## last conv layer
        if layer_to_grad.split('.')[0] in ["gap_softmax"]:
            if self.has_wrapper_seg:
                self.gradcam_layer = LayerGradCam(self.agg_segmentation_wrapper,
                                                  self.model.gap_softmax.conv1)
            else:
                self.gradcam_layer = LayerGradCam(self.model,
                                                  self.model.gap_softmax.conv1)
        elif layer_to_grad.split('.')[0] in ["tcn_layers"]:
            if self.has_wrapper_seg:
                self.gradcam_layer = LayerGradCam(self.agg_segmentation_wrapper,
                                                  self.model.tcn_layers.network[3].conv2)
            else:
                self.gradcam_layer = LayerGradCam(self.model,
                                                  self.model.tcn_layers.network[3].conv2)
        elif layer_to_grad.split('.')[0] in ["conv_block"]:
            if self.has_wrapper_seg:
                self.gradcam_layer = LayerGradCam(self.agg_segmentation_wrapper,
                                                  self.model.layers[3].conv1)
            else:
                self.gradcam_layer = LayerGradCam(self.model,
                                                  self.model.layers[3].conv1)
        else:
            if self.has_wrapper_seg:
                if type(self.model) != Utime:
                    self.gradcam_layer = LayerGradCam(self.agg_segmentation_wrapper,
                                                      self.model.seg_classifier[0])
                else:
                    self.gradcam_layer = LayerGradCam(self.agg_segmentation_wrapper,
                                                      self.model.segmet_classifier.layers[0])
            else:
                self.gradcam_layer = LayerGradCam(self.model,
                                                  self.model.seg_classifier[0]) ## which is conv1d
        gradcam_maps = self.gradcam_layer.attribute(inputs=inp,
                                                    target=label,
                                                    attribute_to_layer_input=attribute_to_layer_input,
                                                    relu_attributions=use_relu)

        print(f"[Shape] GradCAM Shape before upsample: {gradcam_maps.shape}")
        if upsample_to_input:
            gradcam_maps = gradcam_maps.unsqueeze(dim=0)
            if type(inp) == tuple:
                gradcam_maps = LayerAttribution.interpolate(gradcam_maps,
                                                            interpolate_dims=(1, inp[0].shape[-1]))
            else:
                gradcam_maps = LayerAttribution.interpolate(gradcam_maps,
                                                            interpolate_dims=(1, inp.shape[-1]))
            gradcam_maps = gradcam_maps.squeeze(dim=0)
            print(f"[Shape] GradCAM Shape After upsample: {gradcam_maps.shape}")
        
        if self.tsr_saliency is not None:
            maps = gradcam_maps * self.tsr_saliency
            gradcam_maps = maps
        if absolute:
            gradcam_maps = t.abs(gradcam_maps)
        # gradcam_maps = tuple(gradcam_maps)
        gradcam_maps = tuple(
            abs(gradcam) for gradcam in gradcam_maps
        )
        return gradcam_maps

    ## Guided GradCAM
    def getGuidedGradCAM(self,
                         inp,
                         label,
                         absolute: bool,
                         layer_to_grad:str = "gap_softmax.conv1",
                         attribute_to_layer_input: bool = True
                         ):
        """
         element-wise product of guided backpropagation attributions with
         upsampled (non-negative)[with Relu] GradCAM attributions.

        Original Paper: https://arxiv.org/pdf/1610.02391.pdf
        Parameters
        ----------
        inp (tensor): the input samples
        label (int or List): the labels of the samples
        absolute: bool, return absolute value after compute the gradients, when True
                (default: True)
        layer_to_grad: str, the layer to compute the gradients
        attribute_to_layer_input: bool, Indicates whether to compute the attribution
                                with respect to the layer input or output in LayerGradCam.
                                If attribute_to_layer_input is set to True
                                then the attributions will be computed with respect to layer inputs,
                                otherwise it will be computed with respect to layer outputs
        Returns
        -------
        guided_gradcam_maps
        """
        # ## compute guided gradCAM separately
        # gradcam_maps = self.getGradCAM(inp=inp,
        #                                label=label,
        #                                absolute=False,
        #                                layer_to_grad=layer_to_grad,
        #                                upsample_to_input=True,
        #                                use_relu=True,
        #                                attribute_to_layer_input=attribute_to_layer_input
        #                                )
        # gbp_maps = self.getGuidedBackprop(inp=inp,
        #                                   label=label,
        #                                   absolute=False)
        # g_gradcam_maps = gbp_maps * gradcam_maps

        # Compute guided grad cam
        if layer_to_grad.split('.')[0] in ["gap_softmax"]:
            if self.has_wrapper_seg:
                self.guided_gradCAM = GuidedGradCam(model=self.agg_segmentation_wrapper,
                                                    layer=self.model.gap_softmax.conv1)
            else:
                self.guided_gradCAM = GuidedGradCam(model=self.model,
                                                    layer=self.model.gap_softmax.conv1)
        elif layer_to_grad.split('.')[0] in ["tcn_layers"]:
            if self.has_wrapper_seg:
                self.guided_gradCAM = GuidedGradCam(model=self.agg_segmentation_wrapper,
                                                    layer=self.model.tcn_layers.network[3].conv2)
            else:
                self.guided_gradCAM = GuidedGradCam(model=self.model,
                                                    layer=self.model.tcn_layers.network[3].conv2)
        elif layer_to_grad.split('.')[0] in ["conv_block"]:
            if self.has_wrapper_seg:
                self.guided_gradCAM = GuidedGradCam(model=self.agg_segmentation_wrapper,
                                                    layer=self.model.layers[3].conv1)
            else:
                self.guided_gradCAM = GuidedGradCam(model=self.model,
                                                    layer=self.model.layers[3].conv1)
        else:
            if self.has_wrapper_seg:
                if type(self.model) != Utime:
                    self.guided_gradCAM = GuidedGradCam(model=self.agg_segmentation_wrapper,
                                                        layer=self.model.seg_classifier[0])
                else:
                    self.guided_gradCAM = GuidedGradCam(model=self.agg_segmentation_wrapper,
                                                        layer=self.model.segmet_classifier.layers[0])
            else:
                self.guided_gradCAM = GuidedGradCam(model=self.model,
                                                    layer=self.model.seg_classifier[0]) ## which is conv1d
        g_gradcam_maps = self.guided_gradCAM.attribute(inputs=inp,
                                                       target=label,
                                                       attribute_to_layer_input=attribute_to_layer_input
                                                       )
        
        if self.tsr_saliency is not None:
            maps = g_gradcam_maps[0] * self.tsr_saliency
            if absolute:
                g_gradcam_maps = tuple(t.abs(maps))
        if absolute and self.tsr_saliency is None:
            g_gradcam_maps = tuple(
            abs(g_gradcam) for g_gradcam in g_gradcam_maps
            )
        return g_gradcam_maps

    ## Guided Backprop
    def getGuidedBackprop(self,
                          inp,
                          label,
                          absolute: bool
                          ):
        """
        Original Paper: https://arxiv.org/abs/1412.6806

        Parameters
        ----------
        inp (tensor): the input samples
        label (int or List): the labels of the samples
        absolute: bool, return absolute value after compute the gradients, when True
                (default: True)

        Returns
        -------
        gbp_maps: Heatmap(Saliency Map) from Guided Backprop

        """
        gbp_maps = self.gbp.attribute(inputs=inp,
                                      target=label)
        if self.tsr_saliency is not None:
            maps = gbp_maps[0] * self.tsr_saliency
            if absolute:
                gbp_maps = tuple(t.abs(maps))
        if absolute and self.tsr_saliency is None:
            gbp_maps = tuple(
            abs(gbp) for gbp in gbp_maps
            )
        return gbp_maps

    ## LIME
    def getLIME(self,
                inp,
                label,
                absolute: bool,
                n_sample: int,
                num_features: int,
                baseline: str = None,
                kernel_width: float = 3.0
                ):
        """
        Remark: From Captum, not use the Lasso linear regression classifier
                but Sklearn Linear regression

        Lime is an interpretability method that trains an interpretable surrogate model by
        sampling points around a specified input example and using model evaluations at
        these points to train a simpler interpretable ‘surrogate’ model, such as a linear model.

        An interpretable linear model is trained with input being the binary vectors and
        outputs as the corresponding scores of the image classifier with the appropriate
        super-pixels masked based on the binary vector.
        Coefficients of the trained surrogate linear model convey
        the importance of each super-pixel.

        LimeBase returns a representation of the interpretable model (e.g. coefficients of the linear
        model)

        Parameters
        ----------
        inp (tensor): the input samples
        label (int or List): the labels of the samples
        absolute: bool, return absolute value after compute the gradients, when True
                (default: True)

        n_sample (int) : The number of samples of the original model used to train
                        the surrogate interpretable model
        num_features (int) : The number of interpretable features (used for masking)
            Values across all tensors should be integers in the range 0 to
            num_interp_features - 1,
            and indices corresponding to the same feature should have
            the same value.
            (Note: this should be smaller than the length of sample)

        baseline (str) : Defines how individual slice will be
                        deactivated (can be 'mean', 'total_mean', 'noise', 'None'= 0)

        kernel_width (float) : Kernel Width for exponential kernel with "cosine" distance
                                (Similarity function)
        absolute: bool, use absolute value or not
                (default: True)

        Returns
        -------
        Heatmap(Saliency Map) from LIME: np.array
        """
        # interpretable_model = SkLearnRidge(alpha=1.0)
        interpretable_model = SkLearnLinearRegression()
        kernel_similarity_func = get_exp_kernel_similarity_function(distance_mode="cosine",
                                                                    kernel_width=kernel_width)
        if type(inp) == tuple:
            inp_cp = inp[0]
        else:
            inp_cp = inp
        print('[SHAPE] input shape :', inp_cp.shape)
        feature_maps = t.zeros(inp_cp.shape, dtype=t.long)

        num_per_slice = math.ceil(inp_cp.shape[-1] / num_features)
        for i in range(inp_cp.shape[1]):
            for j in range(num_features):
                start = j * num_per_slice
                end = start + num_per_slice
                end = min(end, inp_cp.shape[-1])
                feature_maps[:, i, start:end] = i * num_features + j

        ## create replace baseline
        ## for perturbation
        tmp_series = t.zeros(inp_cp.shape)
        for i in range(num_features):
            start = i * num_per_slice
            end = start + num_per_slice
            end = min(end, inp_cp.shape[-1])
            if baseline is 'mean':
                # use mean of slice as inactive
                tmp_series[:, :, start:end] = t.mean(
                    inp_cp[:, :, start:end], dim=-1).reshape(inp_cp.shape[0], inp_cp.shape[1], -1)
            elif baseline is 'noise':
                # use random noise as inactive
                tmp_series[:, :, start:end] = t.from_numpy(np.random.uniform(
                    inp_cp.cpu().detach().numpy().min(),
                    inp_cp.cpu().detach().numpy().max(),
                    tmp_series[:, :, start:end].shape))
            elif baseline is 'total_mean':
                # use total series mean as inactive
                tmp_series[:, :, start:end] = t.mean(inp_cp, dim=-1).reshape(inp_cp.shape[0],
                                                                             inp_cp.shape[1],
                                                                             -1)

        if self.has_wrapper_seg:
            self.lime = Lime(self.agg_segmentation_wrapper.forward,
                             interpretable_model=interpretable_model,
                             similarity_func=kernel_similarity_func
                             )
        else:
            self.lime = Lime(self.model.forward,
                             interpretable_model=interpretable_model,
                             similarity_func=kernel_similarity_func)
        if isinstance(inp_cp, t.Tensor):
            device = inp_cp.device
        # baseline_set = t.from_numpy(np.quantile(sample.cpu().detach().numpy(), q=0.3, axis=-1).astype("float32")).to(device)
        lime_maps = []
        for i in range(inp_cp.shape[0]):
            sample = inp_cp[i]
            sample = sample.reshape(1, sample.shape[0], sample.shape[1])
            serie = tmp_series[i].reshape(1, tmp_series.shape[1], tmp_series.shape[-1])
            feature_map = feature_maps[i].reshape(1, feature_maps.shape[1], feature_maps.shape[-1])
            limemap = self.lime.attribute(inputs=sample,
                                            target=label,
                                            baselines=serie.to(device) if baseline is not None else None,
                                            n_samples=n_sample,
                                            feature_mask=feature_map.to(device),
                                            return_input_shape=True)
            if absolute:
                limemap = t.abs(limemap)
            lime_maps.append(limemap)
        return tuple(lime_maps)

    ## SHAP
    def getKernelSHAP(self,
                      inp,
                      label,
                      absolute: bool,
                      n_sample: int,
                      baseline: str = None,
                      num_features: int = 50
                      ):
        """
        Remark: From Captum -> Kernel Function is important here for Kernel Shap to estimate Shapley
                Values

        Kernel SHAP is a method that uses the LIME framework to compute Shapley Values.
        Setting the loss function, weighting kernel and regularization terms appropriately
        in the LIME framework allows theoretically obtaining Shapley Values more efficiently
        than directly computing Shapley Values.

        Parameters
        ----------
        inp (tensor): the input samples
        label (int or List): the labels of the samples
        absolute: bool, return absolute value after compute the gradients, when True
                (default: True)

        n_sample (int) : The number of samples of the original model used to train
                        the surrogate interpretable model
        baseline (str) : Defines how individual slice will be
                        deactivated (can be 'mean', 'total_mean', 'noise', 'None'= 0)
        num_features (int) : The number of interpretable features (used for masking)
                            Values across all tensors should be integers in the range 0 to
                            num_interp_features - 1,
                            and indices corresponding to the same feature should have
                            the same value.
                            (Note: this should be smaller than the length of sample)

        Returns
        -------
        kernelshap_maps: Heatmap(Saliency Map) from Kernel SHAP
        """
        if type(inp) == tuple:
            inp_cp = inp[0]
        else:
            inp_cp = inp
        print('[SHAPE] input shape :', inp_cp.shape)
        ## pass on feature maps
        feature_maps = t.zeros(inp_cp.shape, dtype=t.long)
        num_per_slice = math.ceil(inp_cp.shape[-1] / num_features)
        tmp_series = t.zeros(inp_cp.shape)
        for i in range(inp_cp.shape[1]):
            for j in range(num_features):
                start = j * num_per_slice
                end = start + num_per_slice
                end = min(end, inp_cp.shape[-1])
                feature_maps[:, i, start:end] = i * num_features + j

        ## create replace baseline
        ## for perturbation
        for i in range(num_features):
            start = i * num_per_slice
            end = start + num_per_slice
            end = min(end, inp_cp.shape[-1])
            if baseline is 'mean':
                # use mean of slice as inactive
                tmp_series[:, :, start:end] = t.mean(
                    inp_cp[:, :, start:end], dim=-1).reshape(inp_cp.shape[0], inp_cp.shape[1], -1)
            elif baseline is 'noise':
                # use random noise as inactive
                tmp_series[:, :, start:end] = t.from_numpy(np.random.uniform(
                    inp_cp.cpu().detach().numpy().min(),
                    inp_cp.cpu().detach().numpy().max(),
                    tmp_series[:, :, start:end].shape))
            elif baseline is 'total_mean':
                # use total series mean as inactive
                tmp_series[:, :, start:end] = t.mean(inp_cp, dim=-1).reshape(inp_cp.shape[0], inp_cp.shape[1], -1)

        if isinstance(inp_cp, t.Tensor):
            device = inp_cp.device

        kernelshap_maps = []
        for i in range(inp_cp.shape[0]):
            sample = inp_cp[i]
            sample = sample.reshape(1, sample.shape[0], sample.shape[1])
            serie = tmp_series[i].reshape(1, tmp_series.shape[1], tmp_series.shape[-1])
            feature_map = feature_maps[i].reshape(1, feature_maps.shape[1], feature_maps.shape[-1])
            kernelshap_map = self.kernelshap.attribute(inputs=sample,
                                                       target=label,
                                                       n_samples=n_sample,
                                                       baselines=serie.to(
                                                           device) if baseline is not None else None,
                                                       feature_mask=feature_map.to(device),
                                                       return_input_shape=True)
            if absolute:
                kernelshap_map = t.abs(kernelshap_map)
            kernelshap_maps.append(kernelshap_map)
        return tuple(kernelshap_maps)

    def getRandomSaliency(self,
                          inp,
                          label,
                          absolute: bool):
        """
        Random Saliency with np.random.uniform (uniform distribution)
        it is already normalized to [0, 1] or [-1, 1]

        label does not make any difference here
        Parameters
        ----------
        inp: the input sample
        label: the label of the input sample, however, for random saliency, we do not need it
        absolute: whether take the absolute value or not

        Returns
        -------
        random saliency
        """
        if type(inp[0]) == t.Tensor:
            inp = inp[0].cpu().detach().numpy()
        random_maps = np.random.uniform(low=0.0, high=1.0, size=inp.shape) if absolute else np.random.uniform(
            low=-1.0, high=1.0, size=inp.shape
        )
        if random_maps.shape[1] != 1:
            random_maps = random_maps.reshape(random_maps.shape[0], 1, -1)
        random_maps = t.tensor(random_maps)
        random_maps = tuple(
            abs(random) for random in random_maps
        )
        return random_maps

class SaliencyFunctionsDensely(SaliencyFunctions):
    def __init__(self,
                 model: nn.Module,
                 device=None):
        """
        Parameters
        ----------
        model: nn.Module (the model structure)
        data: The TestData set, to run the Vis.Methods. Data contain the samples and labels
        labels (List): the labels inside the datasets
        use_prediction: bool, use the prediction from the model when True (False: use the labels)
        device: torch CPU or GPU
        """
        super().__init__(model, device)
        self.agg_segmentation_wrapper = AggSegmentationWrapper(model=model)
        ## Vis. Methods
        self.gradientSaliency = Saliency(self.agg_segmentation_wrapper.forward)
        self.ig = IG(self.agg_segmentation_wrapper.forward)
        self.smoothNoise = NoiseTunnel(Saliency(self.agg_segmentation_wrapper.forward))
        self.lrp = LRP(self.agg_segmentation_wrapper)
        self.gbp = GuidedBackprop(self.agg_segmentation_wrapper)

        ## Perturbation Based
        # ## Lime

        ## kernel SHAP is based on LIME
        self.kernelshap = KernelShap(self.agg_segmentation_wrapper.forward)

        self.has_wrapper_seg = True
        # for key in self.__dict__.keys():
        #     if key == 'agg_segmentation_wrapper':
        #         self.has_wrapper_seg = True
