"""
All tools for visualization methods
"""
## -------------------
## --- Third-Party ---
## -------------------
import sys
sys.path.append('../')
import math
from sklearn.preprocessing import minmax_scale, maxabs_scale
from scipy.special import binom
import numpy as np
from typing import Dict, List
import torch as t
import torch.nn as nn

from captum.attr import Saliency, NoiseTunnel, LayerGradCam
from captum.attr import GuidedBackprop, GuidedGradCam
from captum.attr import LayerAttribution
from captum.attr import IntegratedGradients as IG
from captum.attr import Lime, KernelShap
from captum.attr._utils.lrp_rules import GammaRule, EpsilonRule
from captum._utils.models.linear_model import SkLearnRidge, SkLearnLinearRegression
from captum.attr._core.lime import get_exp_kernel_similarity_function

## -----------
## --- Own ---
## -----------
from trainhelper.dataset import Dataset
#from visualize_mechanism.lrp import LRP_individual
from visualize_mechanism.lrp_for_captum import LRP as LRP_individual
from visualize_mechanism.lrp_for_captum import LRP 
# from visualize_mechanism.lime_timeseries import LimeTimeSeriesExplainer
from visualize_mechanism.lrp_4_lstm import LSTM_bidi, LSTM_unodi
from models.tcn import Chomp1d, AdditionModule
from models.unet import Utime
from models.basic_blocks import LastStepModule

"""
Use Captum to get the saliency maps (include Gradients, Integrated Gradients,
    SmoothGrad, GradCAM, (CAM), LRP, LIME, SHAP, Guided Backprop, Guided GradCAM)
"""
class SaliencyConstructor:
    def __init__(self, model: nn.Module,
                 data: Dataset,
                 use_prediction: bool = False, device=None):
        """
        Parameters
        ----------
        model: nn.Module (the model structure)
        data: The TestData set, to run the Vis.Methods. Data contain the samples and labels
        use_prediction: bool, use the prediction from the model when True (False: use the labels)
        device: torch CPU or GPU
        """
        self.model = model
        self.device = device
        self.dataset = data
        self.data = t.tensor(self.dataset.data).float().to(device)  ## Shape = [Batch, Dimension, Length]
        self.labels = t.tensor(self.dataset.labels)
        self.use_prediction = use_prediction
        self.has_wrapper_seg = False


        ## Vis. Methods
        self.gradientSaliency = Saliency(self.model.forward)
        self.ig = IG(self.model.forward)
        self.smoothNoise = NoiseTunnel(Saliency(self.model.forward))
        self.lrp = LRP(self.model)
        self.gbp = GuidedBackprop(self.model)

        ## Perturbation Based
        ## Lime
        # self.lime_timeseries = LimeTimeSeriesExplainer(kernel_width=25,
        #                                                verbose=False,
        #                                                class_names=labels,
        #                                                feature_selection="auto")
        ## kernel SHAP is based on LIME
        self.kernelshap = KernelShap(self.model.forward)

    def get_mean_std(self):
        mean = t.mean(self.data, dim=-1)
        mean = t.mean(mean, dim=0).reshape(1, -1, 1)
        std = t.std(self.data, dim=-1)
        std = t.std(std, dim=0).reshape(1, -1 ,1)
        return mean, std

    def get_sample(self, idx:int):
        sample = self.data[idx, :, :]
        sample = sample.reshape(1, sample.shape[0], sample.shape[1])

        ## Model Prediction (for label or prediction target)
        pred = self.model(sample)
        _, c = t.max(pred, dim=1)
        c = c[0][0]

        label = c if self.use_prediction else self.labels[idx]
        return sample, label

    ## Random Saliency
    def random_saliency(self, idx: int, absolute: bool = True):
        """
        For random baseline, this function is used to created random saliency
        with np.random.uniform (uniform distribution), normalized to [0, 1] or [-1, 1]

        Parameters
        ----------
        idx: index of the dataset(sample), here in order to get the shape of the sample
        absolute: bool, return absolute value when True (default: True)

        Returns
        -------
        random_saliency: np.arrays
        """
        sample, label = self.get_sample(idx=idx)
        self.random_map = np.random.uniform(low=0.0, high=1.0, size=sample.shape) if absolute else np.random.uniform(
            low=-1.0, high=1.0, size=sample.shape
        )
        return self.random_map

    ## Gradients
    def gradient_saliency(self, idx:int, absolute:bool = True):
        """
        Original Paper: https://arxiv.org/pdf/1312.6034.pdf
        Parameters
        ----------
        idx: index of the dataset (sample)
        absolute: bool, return absolute value after compute the gradients, when True
                (default: True)

        Returns
        -------
        gradients: np.arrays
        """
        sample, label = self.get_sample(idx=idx)
        self.gradients = self.gradientSaliency.attribute(inputs=sample,
                                                         target=label,
                                                         abs=absolute)
        return self.gradients.cpu().detach().numpy()

    ## Integrated Gradients
    def integrated_gradients(self, idx:int, ig_steps:int = 60, absolute:bool = True):
        """
        Original Paper: https://arxiv.org/abs/1703.01365
        Parameters
        ----------
        idx: index of the dataset (sample)
        ig_steps: Steps for Integrated Gradients to addition (integral)
        absolute: bool, use absolute value or not
                (default: True)

        Returns
        -------
        ig_map: np.arrays
        """
        sample, label = self.get_sample(idx=idx)
        self.ig_map = self.ig.attribute(inputs=sample,
                                        target=label,
                                        baselines=None,
                                        n_steps=ig_steps,
                                        method="riemann_trapezoid")
        if absolute:
            self.ig_map = t.abs(self.ig_map)
        return self.ig_map.cpu().detach().numpy()

    ## Smooth Gradients
    def smooth_gradients(self, idx:int, nt_samples:int = 60, stdevs:float = 0.2, absolute:bool = True):
        """
        Parameters
        ----------
        idx: index of the dataset (sample)
        nt_samples: number of random samples
        stdevs: standard deviation of Gaussian Noise with zero mean
        absolute: bool, use absolute value or not
                (default: True)

        Returns
        -------
        smoothGradients: np.arrays
        """
        sample, label = self.get_sample(idx=idx)
        self.smoothGradients = self.smoothNoise.attribute(inputs=sample,
                                                          target=label,
                                                          nt_type="smoothgrad",
                                                          nt_samples=nt_samples,
                                                          stdevs=stdevs)
        if absolute:
            self.smoothGradients = t.abs(self.smoothGradients)
        return self.smoothGradients.cpu().detach().numpy()

    ## LRP selbst
    def lrp_selbst(self, idx:int,
                   rule: str = "epsilon",
                   absolute: bool = True):
        """
        Parameters
        ----------
        idx: index of the dataset (sample)
        rule (str) : either 'epsilon' or 'gamma'
        absolute: bool, use absolute value or not
                (default: True)
        Returns
        ----------
        relevance
        """
        sample, label = self.get_sample(idx=idx)
        mean, std = self.get_mean_std()
        lrp_ = LRP_individual(model=self.model,
                              mean=mean,
                              std=std,
                              rule=rule)
        r, A_forward = lrp_.forward_pass(X=sample,
                                         target_class=label)
        relevance = lrp_.backward_pass(A_forward=A_forward)

        if absolute:
            relevance = t.abs(relevance)
        return relevance
    ## LRP
    def lrp_(self, idx:int,
             rule: str = "epsilon",
             absolute:bool = True):
        """
        Original Paper: https://doi.org/10.1371/journal.pone.0130140
        Epsilon-Rule for middle-layer
        Gamma-Rule for Lower-layer
        First Input-Layer for Z-Rule

        Parameters
        ----------
        idx: index of the dataset (sample)
        rule: use uniform rules. (default: Epsilon)
                it can be "epsilon" or "gamma"
        absolute: bool, use absolute value or not
                Default: True

        Returns
        -------
        lrp_map: np.array
        """
        sample, label = self.get_sample(idx=idx)

        ## Assign Rules for LRP layer-wise
        if rule is "epsilon":
            supported_layers_rules = set_rule(EpsilonRule)
        elif rule is "gamma":
            supported_layers_rules = set_rule(GammaRule)
        else:
            raise TypeError("Please select propagation rules inherited from Class"
                            "either epsilon or gamma")

        self.lrp_map = self.lrp.attribute(inputs=sample,
                                          target=label,
                                          rule_dict=supported_layers_rules,
                                          return_convergence_delta=True
                                          )
        lrp_map = t.tensor(self.lrp_map[0], dtype=t.float64)
        # if t.abs(self.lrp_map[1][0, 0]) > 100:
        #     raise ValueError('convergence delta value too big')
        if absolute:
            lrp_map = t.abs(lrp_map)
        return lrp_map.cpu().detach().numpy()

    ## LRP for LSTM
    def lrp4lstm_(self, idx: int,
                  eps: float = 0.001,
                  bias_factor: float = 0.0,
                  absolute: bool = True):
        """
        LRP (Layer-Wise relevance propagation) for LSTM
        Parameters
        ----------
        idx: ndex of the dataset (sample)
        eps (float): epsilon propagation, default: 0.001
        bias_factor (float): default: 0.0, for the bias factor in LRP
        absolute (bool): use absolute value or not
                        (default: True)

        Returns
        -------

        """
        ## segmentation is also same as classification
        # if self.has_wrapper_seg:
        #     if self.
        # else:
        sample, label = self.get_sample(idx=idx)

        if self.model.num_directions == 2:
            lrp_lstm = LSTM_bidi(model=self.model,
                                 device=self.device)
            Rx, Rx_rev, R_rest = lrp_lstm.lrp(sample=sample,
                                              LRP_class=label,
                                              eps=eps,
                                              bias_factor=bias_factor)
            Rx = Rx + Rx_rev
        elif self.model.num_directions == 1:
            lrp_lstm = LSTM_unodi(model=self.model,
                                 device=self.device)

            Rx, R_rest = lrp_lstm.lrp(sample=sample,
                                      LRP_class=label,
                                      eps=eps,
                                      bias_factor=bias_factor)
        Rx = Rx.T
        if absolute:
            Rx = np.abs(Rx)
        # ## for sanity check
        ## forward check
        # scores = lrp_lstm.s.copy()
        # score_torch = self.model.forward(sample)

        # lrp_lstm = LSTM_bidi(model=self.model,
        #                      device=self.device)
        # bias_factor = 1.0
        # Rx_check, R_rest_check = lrp_lstm.lrp(sample=sample,
        #                                       LRP_class=label,
        #                                       eps=eps,
        #                                       bias_factor=bias_factor)
        # R_total = Rx_check.sum() + R_rest_check.sum()
        # print("sanity check passed?:", np.allclose(R_total, lrp_lstm.s[label]))

        return Rx


    ## Grad-CAM
    def grad_cam(self, idx:int, use_relu:bool = False,
                 upsample_to_input:bool = True,
                 layer_to_grad:str = "gap_softmax.conv1",
                 attribute_to_layer_input: bool = True,
                 absolute:bool = True):
        """
        Original Paper: https://arxiv.org/pdf/1610.02391.pdf
        Parameters
        ----------
        idx: index of the dataset (sample)
        use_relu: bool, Relu after compute the GradCAM when True
                (default: False)
        upsample_to_input: bool, Whether upsample to input (interpolation)
        layer_to_grad: str, the layer to compute the gradients
        attribute_to_layer_input: bool, Indicates whether to compute the attribution
                                with respect to the layer input or output in LayerGradCam.
                                If attribute_to_layer_input is set to True
                                then the attributions will be computed with respect to layer inputs,
                                otherwise it will be computed with respect to layer outputs
        absolute: bool, use absolute value or not
                (default: True)

        Returns
        -------
        gradcam_map: np.array
        """
        sample, label = self.get_sample(idx=idx)
        if layer_to_grad.split('.')[0] in ["gap_softmax"]:
            if self.has_wrapper_seg:
                self.gradcam_layer = LayerGradCam(self.agg_segmentation_wrapper,
                                                  self.model.gap_softmax.conv1)
            else:
                self.gradcam_layer = LayerGradCam(self.model,
                                                  self.model.gap_softmax.conv1)
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
        self.gradcam_map = self.gradcam_layer.attribute(inputs=sample,
                                                        target=label,
                                                        attribute_to_layer_input=attribute_to_layer_input,
                                                        relu_attributions=use_relu)

        # print(f"[Shape] GradCAM Shape before upsample: {self.gradcam_map.shape}")
        if upsample_to_input:
            self.gradcam_map = self.gradcam_map.unsqueeze(dim=0)
            self.gradcam_map = LayerAttribution.interpolate(self.gradcam_map,
                                                            interpolate_dims=(1, sample.shape[-1]))
            self.gradcam_map = self.gradcam_map.squeeze(dim=0)

        if absolute:
            self.gradcam_map = t.abs(self.gradcam_map)
        return self.gradcam_map.cpu().detach().numpy()

    ## Guided GradCAM
    def guided_gradCAM_(self, idx:int,
                        use_relu: bool = False,
                        layer_to_grad:str = "gap_softmax.conv1",
                        attribute_to_layer_input: bool = True,
                        absolute:bool = True):
        """
         element-wise product of guided backpropagation attributions with
         upsampled (non-negative)[with Relu] GradCAM attributions.

        Original Paper: https://arxiv.org/pdf/1610.02391.pdf
        Parameters
        ----------
        idx: index of the dataset (sample)
        use_relu: bool, Relu after compute the GradCAM when True
                (default: False)
        layer_to_grad: str, the layer to compute the gradients
        attribute_to_layer_input: bool, Indicates whether to compute the attribution
                                with respect to the layer input or output in LayerGradCam.
                                If attribute_to_layer_input is set to True
                                then the attributions will be computed with respect to layer inputs,
                                otherwise it will be computed with respect to layer outputs
        absolute: bool, use absolute value or not
                (default: True)

        Returns
        -------
        guided_gradcam_map: np.array
        """
        sample, label = self.get_sample(idx=idx)

        # Compute grad cam
        # if layer_to_grad.split('.')[0] in ["gap_softmax"]:
        #     grad_cam = LayerGradCam(self.model, self.model.gap_softmax.conv1)
        # else:
        #     grad_cam = LayerGradCam(self.model, self.model.seg_classifier[0]) ## which is conv1d
        #
        # grad_cam_map = grad_cam.attribute(inputs=sample,
        #                                   target=label,
        #                                   attribute_to_layer_input=attribute_to_layer_input,
        #                                   relu_attributions=use_relu)
        # # upsample to input shape
        # gradcam_map = grad_cam_map.unsqueeze(dim=0)
        # gradcam_map = LayerAttribution.interpolate(gradcam_map,
        #                                            interpolate_dims=(1, sample.shape[-1]))
        # gradcam_map = gradcam_map.squeeze(dim=0)
        #
        # # Compute Guided Backpropagation
        # guided_bp = self.gbp.attribute(inputs=sample,
        #                                target=label)
        #
        # self.g_gradcam_map = guided_bp * gradcam_map

        # Compute guided grad cam
        if layer_to_grad.split('.')[0] in ["gap_softmax"]:
            if self.has_wrapper_seg:
                self.guided_gradCAM = GuidedGradCam(model=self.agg_segmentation_wrapper,
                                                    layer=self.model.gap_softmax.conv1)
            else:
                self.guided_gradCAM = GuidedGradCam(model=self.model,
                                                    layer=self.model.gap_softmax.conv1)
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
        self.g_gradcam_map = self.guided_gradCAM.attribute(inputs=sample,
                                                           target=label,
                                                           attribute_to_layer_input=attribute_to_layer_input
                                                           )
        if absolute:
            self.g_gradcam_map = t.abs(self.g_gradcam_map)
        return self.g_gradcam_map.cpu().detach().numpy()

    ## Guided Backprop
    def guided_backprop(self, idx: int, absolute:bool = True):
        """
        Original Paper: https://arxiv.org/abs/1412.6806
        Parameters
        ----------
        idx: index of the dataset (sample)
        absolute: bool, use absolute value or not
                (default: True)

        Returns
        -------
        Heatmap(Saliency Map) from Guided Backprop: np.array
        """
        sample, label = self.get_sample(idx=idx)
        self.gbp_map = self.gbp.attribute(inputs=sample,
                                          target=label)

        if absolute:
            self.gbp_map = t.abs(self.gbp_map)
        return self.gbp_map.cpu().detach().numpy()

    ## LIME for Time series
    ## not from Captum Library
    def lime_ts(self, idx: int,
                num_slices: int,
                num_features: int,
                n_sample: int,
                model_regressor=None,
                replacement_method='mean',
                absolute: bool = True):
        """
        Generates a LIME explanation for prediction sample
        Not from Captum Library

        Parameters
        ----------
        idx (int):  index of the dataset (sample), which is to be explained
        num_slices (int): the number of slices, the time series will be split up
        num_features (int): maximum number of features present in explanation
        n_sample (int): size of the neighborhood to learn the linear model (which will be created until
                        the size is reached)
        model_regressor: sklearn regressor to use in explanation. Defaults
                to Ridge regression in LimeBase. Must have
                model_regressor.coef_ and 'sample_weight' as a parameter to
                model_regressor.fit()
        replacement_method:  Defines how individual slice will be
                deactivated (can be 'mean', 'total_mean', 'noise')
                'mean': the mean value inside the slice
                'total_mean': the whole mean in the whole time series sequence per sample
                'noise': np random uniform noise will replace the raw signal data in slice
        absolute: bool, use absolute value or not
                (default: True)

        Returns
        -------

        """
        sample, label = self.get_sample(idx=idx)
        if len(label.shape) == 0:
            label = int(label.cpu().detach().numpy())
        else:
            label = label.cpu().detach().numpy()
        self.lime_explainer = self.lime_timeseries.explain_instance(
            timeseries_instance= sample,
            classifier_fn= self.model.forward,
            num_slices= num_slices,
            labels= [label],
            num_features=num_features,
            num_samples=n_sample,
            model_regressor=model_regressor,
            replacement_method=replacement_method
        )
        explainer = self.lime_explainer.as_list(label=label)
        lime_map = np.zeros(sample.shape)
        values_per_slice = math.ceil(sample.shape[-1] / len(explainer))
        for i in range(len(explainer)):
            feature, weight = explainer[i]
            start = feature * values_per_slice
            end = start + values_per_slice
            end = min(end, sample.shape[-1])
            lime_map[0, :, start:end] = abs(weight) if absolute else weight
        return lime_map

    ## KernelShap for Time series
    def kernelShap_ts(self, idx: int,
                    num_slices: int,
                    num_features: int,
                    n_sample: int,
                    model_regressor=None,
                    replacement_method='mean',
                    absolute: bool = True):
        """
        (Remark: the kernel function should not be too small (the weights)
        Generates a SHAP explanation for prediction sample
        Not from Captum Library, use the same LIMEBase with LIME
        From the original Lime Library

        Parameters
        ----------
        idx (int):  index of the dataset (sample), which is to be explained
        num_slices (int): the number of slices, the time series will be split up
        num_features (int): maximum number of features present in explanation
        n_sample (int): size of the neighborhood to learn the linear model (which will be created until
                        the size is reached)
        model_regressor: sklearn regressor to use in explanation. Defaults
                to Ridge regression in LimeBase. Must have
                model_regressor.coef_ and 'sample_weight' as a parameter to
                model_regressor.fit()
        replacement_method:  Defines how individual slice will be
                deactivated (can be 'mean', 'total_mean', 'noise')
                'mean': the mean value inside the slice
                'total_mean': the whole mean in the whole time series sequence per sample
                'noise': np random uniform noise will replace the raw signal data in slice
        absolute: bool, use absolute value or not
                (default: True)

        Returns
        -------

        """
        sample, label = self.get_sample(idx=idx)
        if len(label.shape) == 0:
            label = int(label.cpu().detach().numpy())
        else:
            label = label.cpu().detach().numpy()
        shap_explainer = self.kernel_shap.explain_instance(
            timeseries_instance=sample,
            classifier_fn=self.model.forward,
            num_slices=num_slices,
            labels=[label],
            num_features=num_features,
            num_samples=n_sample,
            model_regressor=model_regressor,
            replacement_method=replacement_method
        )
        explainer = shap_explainer.as_list(label=label)
        shap_map = np.zeros(sample.shape)
        values_per_slice = math.ceil(sample.shape[-1] / len(explainer))
        for i in range(len(explainer)):
            feature, weight = explainer[i]
            start = feature * values_per_slice
            end = start + values_per_slice
            end = min(end, sample.shape[-1])
            shap_map[0, :, start:end] = abs(weight) if absolute else weight
        return shap_map

    ## LIME
    def lime_(self, idx: int,
              n_sample: int,
              num_features: int,
              baseline: str = None,
              kernel_width: float = 3.0,
              absolute: bool = True):
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
        idx: index of the dataset (sample)
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
        sample, label = self.get_sample(idx=idx)
        # interpretable_model = SkLearnRidge(alpha=1.0)
        interpretable_model = SkLearnLinearRegression()
        kernel_similarity_func = get_exp_kernel_similarity_function(distance_mode="cosine",
                                                                    kernel_width=kernel_width)
        feature_maps = t.zeros(sample.shape, dtype=t.long)
        num_per_slice = math.ceil(sample.shape[-1] / num_features)
        for i in range(sample.shape[1]):
            for j in range(num_features):
                start = j * num_per_slice
                end = start + num_per_slice
                end = min(end, sample.shape[-1])
                feature_maps[0, i, start:end] = i * num_features + j

        ## create replace baseline
        ## for perturbation
        tmp_series = t.zeros(sample.shape)
        for i in range(num_features):
            start = i * num_per_slice
            end = start + num_per_slice
            end = min(end, sample.shape[-1])
            if baseline is 'mean':
                # use mean of slice as inactive
                tmp_series[:, :, start:end] = t.mean(
                    sample[:, :, start:end], dim=-1).reshape(sample.shape[0], sample.shape[1], -1)
            elif baseline is 'noise':
                # use random noise as inactive
                tmp_series[:, :, start:end] = t.from_numpy(np.random.uniform(
                    sample.cpu().detach().numpy().min(),
                    sample.cpu().detach().numpy().max(),
                    tmp_series[:, :, start:end].shape))
            elif baseline is 'total_mean':
                # use total series mean as inactive
                tmp_series[:, :, start:end] = t.mean(sample, dim=-1).reshape(sample.shape[0], sample.shape[1], -1)

        ## self.lime
        if self.has_wrapper_seg:
            self.lime = Lime(self.agg_segmentation_wrapper.forward,
                             interpretable_model=interpretable_model,
                             similarity_func=kernel_similarity_func
                             )
        else:
            self.lime = Lime(self.model.forward,
                             interpretable_model=interpretable_model,
                             similarity_func=kernel_similarity_func)
        if isinstance(sample, t.Tensor):
            device = sample.device
        # baseline_set = t.from_numpy(np.quantile(sample.cpu().detach().numpy(), q=0.3, axis=-1).astype("float32")).to(device)
        self.lime_map = self.lime.attribute(inputs=sample,
                                            target=label,
                                            baselines= tmp_series.to(device) if baseline is not None else None,
                                            n_samples=n_sample,
                                            feature_mask=feature_maps.to(device),
                                            return_input_shape=True)
        if absolute:
            self.lime_map = t.abs(self.lime_map)
        return self.lime_map.cpu().detach().numpy()

    ## SHAP
    def kernelshap_(self, idx: int, n_sample: int,
                    baseline: str =None,
                    num_features: int = 50,
                    absolute: bool = True):
        """
        Remark: From Captum -> Kernel Function is important here for Kernel Shap to estimate Shapley
                Values

        Kernel SHAP is a method that uses the LIME framework to compute Shapley Values.
        Setting the loss function, weighting kernel and regularization terms appropriately
        in the LIME framework allows theoretically obtaining Shapley Values more efficiently
        than directly computing Shapley Values.

        Parameters
        ----------
        idx (int) : index of the dataset (sample)

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

        absolute (bool) : use absolute value or not
                        (default: True)

        Returns
        -------
        Heatmap(Saliency Map) from Kernel SHAP: np.array
        """
        sample, label = self.get_sample(idx=idx)
        ## self.shap
        feature_maps = t.zeros(sample.shape, dtype=t.long)
        num_per_slice = math.ceil(sample.shape[-1] / num_features)
        tmp_series = t.zeros(sample.shape)
        for i in range(sample.shape[1]):
            for j in range(num_features):
                start = j * num_per_slice
                end = start + num_per_slice
                end = min(end, sample.shape[-1])
                feature_maps[0, i, start:end] = i * num_features + j

        ## create replace baseline
        ## for perturbation
        for i in range(num_features):
            start = i * num_per_slice
            end = start + num_per_slice
            end = min(end, sample.shape[-1])
            if baseline is 'mean':
                # use mean of slice as inactive
                tmp_series[:, :, start:end] = t.mean(
                    sample[:, :, start:end], dim=-1).reshape(sample.shape[0], sample.shape[1], -1)
            elif baseline is 'noise':
                # use random noise as inactive
                tmp_series[:, :, start:end] = t.from_numpy(np.random.uniform(
                    sample.cpu().detach().numpy().min(),
                    sample.cpu().detach().numpy().max(),
                    tmp_series[:, :, start:end].shape))
            elif baseline is 'total_mean':
                # use total series mean as inactive
                tmp_series[:, :, start:end] = t.mean(sample, dim=-1).reshape(sample.shape[0], sample.shape[1], -1)
        self.kernelshap_map = self.kernelshap.attribute(inputs=sample,
                                                        target=label,
                                                        n_samples=n_sample,
                                                        baselines=tmp_series.to(self.device) if baseline is not None else None,
                                                        feature_mask=feature_maps.to(self.device),
                                                        return_input_shape=True)
        if absolute:
            self.kernelshap_map = t.abs(self.kernelshap_map)
        return self.kernelshap_map.cpu().detach().numpy()

    def get_model_accuracy(self):
        """
        Get the average accruacy of the dataset from the deep learning Model
        Returns
        -------
        accuracy: float
        """
        correct = 0
        predictions = None
        with t.no_grad():
            for i, (data, label) in enumerate(self.dataset, 0): ## data size = [B, C(feature), length]
                data = t.tensor(data).reshape((1, *data.shape))
                xt = data.float().to(self.device)
                label = t.tensor(label).to(self.device).reshape((-1, 1))  ## label should be (len, 1)
                ## Forward pass
                predicted = self.model(xt)
                predicted = t.argmax(predicted, dim=1)
                correct += (predicted == label).sum().item()
                if predictions is None:
                    predictions = predicted
                else:
                    predictions = t.cat((predictions, predicted), dim=0)
        avg_acc = (correct / len(self.dataset.labels)) * 100
        print('[Evaluation Accuracy] On testset accuracy rate: {} % \n'.format(avg_acc))
        return avg_acc

class SaliencyConstructor_densely(SaliencyConstructor):
    def __init__(self, model: nn.Module,
                 data: Dataset,
                 use_prediction: bool = False, device=None):
        """
        Parameters
        ----------
        model: nn.Module (the model structure)
        data: The TestData set, to run the Vis.Methods. Data contain the samples and labels
        use_prediction: bool, use the prediction from the model when True (False: use the labels)
        device: torch CPU or GPU
        """
        super().__init__(model, data, use_prediction, device)
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
        for key in self.__dict__.keys():
            if key == 'agg_segmentation_wrapper':
                self.has_wrapper_seg = True

    def get_sample(self, idx: int):
        sample = self.data[idx, :, :]
        sample = sample.reshape(1, sample.shape[0], sample.shape[1])

        ## Model Prediction (for label or prediction target)
        pred = self.agg_segmentation_wrapper(sample)

        ## we need to compact the densely labels into a single label for interpretation
        if not self.use_prediction:
            labels = self.labels[idx]
            labels_summary, labels_counts = np.unique(labels.cpu().detach().numpy(), return_counts=True)
            ## for tool tracking dataset, we maybe don't want to see the garbage class [4]
            labels_order = np.flip(np.argsort(labels_counts))
            label = labels_summary[labels_order[1]] if labels_summary[labels_order[0]] == 4 else labels_summary[
                    labels_order[0]]
        else:
            ## for tool tracking dataset, we maybe don't want to see the garbage class [4]
            labels_order = np.flip(np.argsort(pred.cpu().detach().numpy()))
            label = labels_order[0][1] if labels_order[0][0] == 4 else labels_order[0][0]
        return sample, t.tensor(label)

class AggSegmentationWrapper(nn.Module):
    """
    A challenge with applying Captum to segmentation models (densely labels) is that we need to attribute with respect
    to a single scalar output, such as a target logit in classification cases. With segmentation models, the model output
    is the size of the input image.
    One way to compute attribution is with respect to a particular pixel output score for a given class.
    This is the right approach if we want to understand the influences for a particular pixel,
    but we often want to understand the prediction of an entire segment, as opposed to attributing each pixel independently.

    This can be done in a few ways, the simplest being summing the output logits of each channel,
    corresponding to a total score for each segmentation class, and attributing with respect to the score for the particular class.
    This approach is simple to implement, but could result in misleading attribution when a pixel is not predicted as a certain class
    but still has a high score for that class.
    Based on this, we sum only the scores corresponding to pixels that are predicted to be a particular class (argmax class)
    and attribute with respect to this sum.

    We define a wrapper function that performs this, and can use this wrapper for attribution instead of the original model.

    This wrapper computes the segmentation model output and sums the pixel scores for
    all pixels predicted as each class, returning a tensor with a single value for
    each class. This makes it easier to attribute with respect to a single output
    scalar, as opposed to an individual pixel output attribution.

    References : https://captum.ai/tutorials/Segmentation_Interpret

    Parameters
    ----------
    inp (Tensor) : The input data to the model

    Returns
    -------
    the prediction of the model, but compact into the classification case (from segmentation)
    """
    def __init__(self, model):
        super(AggSegmentationWrapper, self).__init__()
        self.model = model
    def forward(self, inp):
        model_out = self.model(inp)
        ## find most likely segmentation class for each time points
        out_max = t.argmax(model_out, dim=1, keepdim=True)
        # Creates binary matrix with 1 for original argmax class for each pixel
        # and 0 otherwise.
        selected_inds = t.zeros_like(model_out).scatter_(1, out_max, 1)
        return (model_out * selected_inds).sum(dim=(2))  ## for time series data ## shape [#batch, #feature]
    def model_name(self):
        return "AggSegmentWrapper"

## ---------------
## --- Methods ---
## ---------------

## Normalize Saliency Maps
def abs_normalize(data: np.ndarray, percentile=99):
    """Return absolute value normalized data
        Also based on Min-Max Normalization
        The values are between [0, 1]
        References: https://github.com/PAIR-code/saliency/blob/master/saliency/core/visualization.py
    """
    assert isinstance(data, np.ndarray), "data should be a numpy array"

    shp = data.shape
    if len(shp) != 2:
        raise ValueError("Array should have 2 dims!, consider only one sample")
    data = np.absolute(data)
    vmax = np.percentile(data, percentile)
    vmin = np.min(data)

    return np.clip((data - vmin) / (vmax - vmin + 1e-9), 0, 1)

def diverging_normalize(data: np.ndarray):
    """Return data with positive and negative values
        Also based on Min-Max Normalization
        The values are between [-1, 1]
        References: https://github.com/PAIR-code/saliency/blob/master/saliency/core/visualization.py
    """
    assert isinstance(data, np.ndarray), "data should be a numpy array"
    shp = data.shape
    if len(shp) != 2:
        raise ValueError("Array should have 2 dims!, consider only one sample")
    data_reshape = data.reshape(-1, data.shape[0] * data.shape[1])
    rescaledSaliency = maxabs_scale(data_reshape,
                                    axis=1)
    rescaledSaliency = rescaledSaliency.reshape(data.shape)

    return rescaledSaliency

## Normalize Saliency Maps
def min_max_normalize(data: np.ndarray, feature_range=(0,1)):
    """
    Min-Max Normalization

    Parameters
    ----------
    data: Saliency Maps, which will be normalized # Shape: [Features(C), Length]

    Returns
    -------
    data norm: np.ndarray
    """
    assert isinstance(data, np.ndarray), "data should be a numpy array"

    shp = data.shape
    if len(shp) != 2:
        raise ValueError("Array should have 2 dims!, consider only one sample")
    print("[Normalization] -- Min-Max Normalization -- ")
    data_reshape = data.reshape(-1, data.shape[0] * data.shape[1])
    rescaledSaliency = minmax_scale(data_reshape,
                                    feature_range=feature_range,
                                    axis=1)
    rescaledSaliency = rescaledSaliency.reshape(data.shape)
    # for j in range(data_norm.shape[1]):
    #     min_max_scaler.fit(data[:, j, :])
    #     data_norm[:, j, :] = min_max_scaler.transform(data[:, j, :])
        # min_max = np.max(data[:, j, :], axis=0) - np.min(data[:, j, :], axis=0)
        # data_norm[:, j, :] = (data[:, j, :] - np.min(data[:, j, :], axis=0)) / min_max
    return rescaledSaliency

def set_rule(rule):
    """
    In LRP method, one can choose a rule from Gammarule or Epsilonrule
    Parameters
    ----------
    rule: either Gamma rule or Epsilon rule

    Returns
    -------
    Rules_Dictionary
    """
    supported_layers_with_rules = [
        nn.MaxPool1d,
        nn.MaxPool2d,
        nn.MaxPool3d,
        nn.Conv2d,
        nn.AvgPool2d,
        nn.AdaptiveAvgPool2d,
        nn.Linear,
        nn.BatchNorm2d,
        ## Add by myself
        nn.Conv1d,
        nn.BatchNorm1d,
        nn.AdaptiveMaxPool1d,
        nn.ZeroPad2d,
        Chomp1d,
        AdditionModule,
        nn.Flatten,
        LastStepModule
    ]
    supported_layers_with_rules_dict = {}
    for layer in supported_layers_with_rules:
        supported_layers_with_rules_dict[layer] = rule

    return supported_layers_with_rules_dict

def shap_kernel(z, p):
    """ Kernel SHAP weights for z unmasked features among p
        p should be number of slices (number of features)
    """
    kernel = (p-1)/(binom(p, z)*(z*((p+1) - z)))
    return kernel

