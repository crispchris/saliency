## Reproduce Sanity Check (randomize the model also the data)
"""
refer to: Sanity Checks for Saliency Maps
https://github.com/adebayoj/sanity_checks_saliency/tree/3e24048c570f08ca655fcd332b6128fa069810a0
Model Randomization Test and Data Randomization Test

"""
## ------------------
## --- Third-Party ---
## ------------------
import sys
sys.path.append('../')
sys.path.append('../../')
import numpy as np
import torch as t
import torch.nn as nn
from copy import deepcopy
from scipy.stats.mstats import spearmanr as spr
# from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim
# import matplotlib.pyplot as plt
# from matplotlib import gridspec
# from matplotlib.collections import LineCollection
# import seaborn as sns

### -----------
### --- Own ---
### -----------
from utils import load_model
from utils import get_model_weights, throw_out_wrong_classified
from visualize_mechanism.visual_utils import SaliencyConstructor, min_max_normalize
from visualize_mechanism.visual_utils import abs_normalize, diverging_normalize
from visualize_mechanism.visual_func import SaliencyFunctions
from trainhelper.dataset import Dataset
from visualize_mechanism.tsr import tsr_from_saliencymap

device = t.device('cuda' if t.cuda.is_available() else 'cpu')
class SanityCheck:
    """
    A Class to check the visualize methods with random initialized Models
    "Cascade" or "Independent"
    """
    def __init__(self, model: nn.Module,
                 random_model: nn.Module,
                 dataset: Dataset,
                 mode: str = "cascade"):
        """
        Parameters
        ----------
        model (nn.Module) : Torch Model
        dataset (Dataset) : The dataset to made the analyse
        mode (str) : "Cascade" or "Independent"
        """
        self.model = model
        self.random_model = random_model
        # self.random_model = deepcopy(self.model)
        self.mode = mode
        self.dataset = dataset

    def load_ckp(self, ckp_path: str):
        """
        Parameters
        ----------
        ckp_path (str) : The trained Model(Weights) path
        Returns
        -------
        A Trained (or Random) Model
        """
        self.model = load_model(self.model, ckp_path=ckp_path, randomized=False)
        cleandata, cleanlabels = throw_out_wrong_classified(self.model,
                                                            self.dataset.data,
                                                            self.dataset.labels,
                                                            device)
        testset = Dataset(cleandata, cleanlabels)
        self.dataset = testset
        self.saliency_constructor = SaliencyConstructor(self.model,
                                                        testset,
                                                        use_prediction=False,
                                                        device=self.model.device)
        self.normal_model_acc = self.saliency_constructor.get_model_accuracy()

    def get_random_saliencies(self, ckp_path: str,
                              absolute: bool = False,
                              use_tsr: bool = False,
                              vis_methods: list = ["grads", "integrated_gradients", "smoothgrads",
                                                   "lrp_epsilon", "lrp_gamma",
                                                   "gradCAM", "g_gradCAM", "gbp",
                                                   "lime", "kernelShap", "random"]):
        ## Randomized Model weights
        random_saliency = []  ## Save the saliency from random initiated model {name of methods: saliency}
        random_saliency_abs = []  ## Save the saliency from random initiated model {name of methods: saliency}
        random_names = []  ## Save the name of the random model (layers) # Correspond to saliency

        random_acc_dict = {}  ## Save the acc of the random guassian model (per layer)

        layer_names, model_gaussian_weights = get_model_weights(self.random_model)
        random_acc_dict['Original'] = [self.normal_model_acc]
        num_layers = len(layer_names)

        idx_layer = 0
        while idx_layer < num_layers:
            ## Random Initialization
            if self.mode in ["Cascade"]:
                random_model_, idx_layer = load_model(model=self.random_model,
                                                      ckp_path=ckp_path,
                                                      randomized=True,
                                                      independent=False,
                                                      idx_layer=idx_layer)
            elif self.mode in ["Independent"]:
                random_model_, idx_layer = load_model(model=self.random_model,
                                                      ckp_path=ckp_path,
                                                      randomized=True,
                                                      independent=True,
                                                      idx_layer=idx_layer)

            ## get tsr
            if use_tsr:
                rescaledSaliency = tsr_from_saliencymap(
                    samples=self.dataset.data,
                    labels=self.dataset.labels,
                    dl_model=random_model_,
                    time_groups=5,
                    feature_groups=1,
                    threshold=0.5
                )
                rescaledSaliency = t.tensor(rescaledSaliency).float().to(device)
                rescaledSaliency = rescaledSaliency.unsqueeze(0)
            else:
                rescaledSaliency = None

            saliency_random = SaliencyConstructor(random_model_,
                                                              data=self.dataset,
                                                              use_prediction=False,
                                                              device=device)
            saliency_constructor_random = SaliencyFunctions(model=random_model_,
                                                            tsr_saliency=rescaledSaliency,
                                                            device=device)

            ## Compute Accuracy of the random model
            splitname = layer_names[idx_layer - 2].split('.')[:-1]
            name = ''
            for i, split in enumerate(splitname):
                if i < len(splitname) - 1:
                    name += split + '.'
                else:
                    name += split
            random_acc_dict[name] = [saliency_random.get_model_accuracy()]

            ## Save the results of saliency
            if "grads" in vis_methods:
                grads_random = np.zeros(self.dataset.data.shape)
                grads_random_abs = np.zeros(self.dataset.data.shape)
            if "igs" in vis_methods:
                igs_random = np.zeros(self.dataset.data.shape)
                igs_random_abs = np.zeros(self.dataset.data.shape)
            if "smoothgrads" in vis_methods:
                smoothgrads_random = np.zeros(self.dataset.data.shape)
                smoothgrads_random_abs = np.zeros(self.dataset.data.shape)
            if "lrp_epsilon" in vis_methods:
                lrp_epsilon_random = np.zeros(self.dataset.data.shape)
                lrp_epsilon_random_abs = np.zeros(self.dataset.data.shape)
            if "lrp_gamma" in vis_methods:
                lrp_gamma_random = np.zeros(self.dataset.data.shape)
                lrp_gamma_random_abs = np.zeros(self.dataset.data.shape)
            if "gradCAM" in vis_methods:
                gradCam_maps_random = np.zeros(self.dataset.data.shape)
                gradCam_maps_random_abs = np.zeros(self.dataset.data.shape)
            if "guided_gradcam" in vis_methods:
                g_gradcam_maps_random = np.zeros(self.dataset.data.shape)
                g_gradcam_maps_random_abs = np.zeros(self.dataset.data.shape)
            if "guided_backprop" in vis_methods:
                gbp_maps_random = np.zeros(self.dataset.data.shape)
                gbp_maps_random_abs = np.zeros(self.dataset.data.shape)
            if "lime" in vis_methods:
                lime_maps_random = np.zeros(self.dataset.data.shape)
                lime_maps_random_abs = np.zeros(self.dataset.data.shape)
            if "kernel_shap" in vis_methods:
                shap_maps_random = np.zeros(self.dataset.data.shape)
                shap_maps_random_abs = np.zeros(self.dataset.data.shape)
            if "random" in vis_methods:
                random_maps_random = np.zeros(self.dataset.data.shape)
                random_maps_random_abs = np.zeros(self.dataset.data.shape)

            print("[Saliency Methods] for Randomization Model\n")
            vis_abs_dict = {}
            vis_no_abs_dict = {}
            for idx in range(len(self.dataset.labels)):
                sample = self.dataset.data[idx]
                sample = sample.reshape(1, sample.shape[0], sample.shape[1])
                sample = t.tensor(sample).float().to(device)
                label = t.tensor(self.dataset.labels[i])
                ## Randomization
                if "grads" in vis_methods:
                    grads_random[idx] = saliency_constructor_random.getGradientSaliency(inp=sample,
                                                                                        label=label,
                                                                                        absolute=absolute)[0].cpu().detach().numpy()
                if "igs" in vis_methods:
                    igs_random[idx] = saliency_constructor_random.getIntegratedGradients(inp=sample,
                                                                                         label=label,
                                                                                         ig_steps=60,
                                                                                         absolute=absolute)[0].cpu().detach().numpy()
                if "smoothgrads" in vis_methods:
                    smoothgrads_random[idx] = saliency_constructor_random.getSmoothGradients(inp=sample,
                                                                                           label=label,
                                                                                           nt_samples=60,
                                                                                           stdevs=0.2,
                                                                                           absolute=absolute)[0].cpu().detach().numpy()
                if "lrp_epsilon" in vis_methods:
                    lrp_epsilon_random[idx] = saliency_constructor_random.getLRP(inp=sample,
                                                                                 label=label,
                                                                                 rule='epsilon',
                                                                                 absolute=absolute)[0].cpu().detach().numpy()
                if "lrp_gamma" in vis_methods:
                    lrp_gamma_random[idx] = saliency_constructor_random.getLRP(inp=sample,
                                                                             label=label,
                                                                             rule='gamma',
                                                                             absolute=absolute)[0].cpu().detach().numpy()
                if "gradCAM" in vis_methods:
                    gradCam_maps_random[idx] = saliency_constructor_random.getGradCAM(inp=sample,
                                                                                    label=label,
                                                                                    use_relu=True,
                                                                                    upsample_to_input=True,
                                                                                    attribute_to_layer_input=True,
                                                                                    absolute=absolute)[0].cpu().detach().numpy()
                if "guided_gradcam" in vis_methods:
                    g_gradcam_maps_random[idx] = saliency_constructor_random.getGuidedGradCAM(
                        inp=sample,
                        label=label,
                        attribute_to_layer_input=True,
                        absolute=absolute)[0].cpu().detach().numpy()
                if "guided_backprop" in vis_methods:
                    gbp_maps_random[idx] = saliency_constructor_random.getGuidedBackprop(inp=sample,
                                                                                       label=label,
                                                                                       absolute=absolute)[0].cpu().detach().numpy()
                if "lime" in vis_methods:
                    lime_maps_random[idx] = saliency_constructor_random.getLIME(inp=sample,
                                                                              label=label,
                                                                              n_sample=1200,
                                                                              num_features=125,
                                                                              baseline="total_mean",
                                                                              kernel_width=5.0,
                                                                              absolute=absolute)[0].cpu().detach().numpy()

                if "kernel_shap" in vis_methods:
                    shap_maps_random[idx] = saliency_constructor_random.getKernelSHAP(inp=sample,
                                                                                    label=label,
                                                                                    n_sample=1200,
                                                                                    baseline="total_mean",
                                                                                    num_features=125,
                                                                                    absolute=absolute)[0].cpu().detach().numpy()
                if "random" in vis_methods:
                    ## absolute = False
                    random_maps_random[idx] = saliency_constructor_random.getRandomSaliency(inp=sample,
                                                                                            label=label,
                                                                                            absolute=False)[0]

                ## Min-Max Normalization
                if "grads" in vis_methods:
                    grads_random_abs[idx] = min_max_normalize(grads_random[idx])
                    grads_random[idx] = diverging_normalize(grads_random[idx])
                if "igs" in vis_methods:
                    igs_random_abs[idx] = min_max_normalize(igs_random[idx])
                    igs_random[idx] = diverging_normalize(igs_random[idx])
                if "smoothgrads" in vis_methods:
                    smoothgrads_random_abs[idx] = min_max_normalize(smoothgrads_random[idx])
                    smoothgrads_random[idx] = diverging_normalize(smoothgrads_random[idx])
                if "lrp_epsilon" in vis_methods:
                    lrp_epsilon_random_abs[idx] = min_max_normalize(lrp_epsilon_random[idx])
                    lrp_epsilon_random[idx] = diverging_normalize(lrp_epsilon_random[idx])
                if "lrp_gamma" in vis_methods:
                    lrp_gamma_random_abs[idx] = min_max_normalize(lrp_gamma_random[idx])
                    lrp_gamma_random[idx] = diverging_normalize(lrp_gamma_random[idx])
                if "gradCAM" in vis_methods:
                    gradCam_maps_random_abs[idx] = min_max_normalize(gradCam_maps_random[idx])
                    gradCam_maps_random[idx] = diverging_normalize(gradCam_maps_random[idx])
                if "guided_gradcam" in vis_methods:
                    g_gradcam_maps_random_abs[idx] = min_max_normalize(g_gradcam_maps_random[idx])
                    g_gradcam_maps_random[idx] = diverging_normalize(g_gradcam_maps_random[idx])
                if "guided_backprop" in vis_methods:
                    gbp_maps_random_abs[idx] = min_max_normalize(gbp_maps_random[idx])
                    gbp_maps_random[idx] = diverging_normalize(gbp_maps_random[idx])
                if "lime" in vis_methods:
                    lime_maps_random_abs[idx] = min_max_normalize(lime_maps_random[idx])
                    lime_maps_random[idx] = diverging_normalize(lime_maps_random[idx])
                if "kernel_shap" in vis_methods:
                    shap_maps_random_abs[idx] = min_max_normalize(shap_maps_random[idx])
                    shap_maps_random[idx] = diverging_normalize(shap_maps_random[idx])
                if "random" in vis_methods:
                    random_maps_random_abs[idx] = min_max_normalize(random_maps_random[idx])
                    random_maps_random[idx] = diverging_normalize(random_maps_random[idx])
            if "grads" in vis_methods:
                vis_abs_dict["grads"] = grads_random_abs
                vis_no_abs_dict["grads"] = grads_random
            if "igs" in vis_methods:
                vis_abs_dict["igs"] = igs_random_abs
                vis_no_abs_dict["igs"] = igs_random
            if "smoothgrads" in vis_methods:
                vis_abs_dict["smoothgrads"] = smoothgrads_random_abs
                vis_no_abs_dict["smoothgrads"] = smoothgrads_random
            if "lrp_epsilon" in vis_methods:
                vis_abs_dict["lrp_epsilon"] = lrp_epsilon_random_abs
                vis_no_abs_dict["lrp_epsilon"] = lrp_epsilon_random
            if "lrp_gamma" in vis_methods:
                vis_abs_dict["lrp_gamma"] = lrp_gamma_random_abs
                vis_no_abs_dict["lrp_gamma"] = lrp_gamma_random
            if "gradCAM" in vis_methods:
                vis_abs_dict["gradCAM"] = gradCam_maps_random_abs
                vis_no_abs_dict["gradCAM"] = gradCam_maps_random
            if "guided_gradcam" in vis_methods:
                vis_abs_dict["guided_gradcam"] = g_gradcam_maps_random_abs
                vis_no_abs_dict["guided_gradcam"] = g_gradcam_maps_random
            if "guided_backprop" in vis_methods:
                vis_abs_dict["guided_backprop"] = gbp_maps_random_abs
                vis_no_abs_dict["guided_backprop"] = gbp_maps_random
            if "lime" in vis_methods:
                vis_abs_dict["lime"] = lime_maps_random_abs
                vis_no_abs_dict["lime"] = lime_maps_random
            if "kernel_shap" in vis_methods:
                vis_abs_dict["kernel_shap"] = shap_maps_random_abs
                vis_no_abs_dict["kernel_shap"] = shap_maps_random
            if "random" in vis_methods:
                vis_abs_dict["random"] = random_maps_random_abs
                vis_no_abs_dict["random"] = random_maps_random

            random_saliency_abs.append(vis_abs_dict)
            random_saliency.append(vis_no_abs_dict)
            random_names.append(name)
            if self.mode is "Cascade":
                idx_layer += 1
        print(f"the length of layer names: {len(layer_names)}")
        print(f"the length of random layer names: {len(random_names)}")
        return random_saliency, random_saliency_abs, random_names, random_acc_dict

    def get_spearman_correlation(self, normal_saliency,
#                                  normal_saliency_abs,
                                 random_saliency: list,
#                                  random_saliency_abs: list,
                                 random_names: list
                                 ):
        """
        Parameters
        ----------
        normal_saliency: Normal Saliency without ABS (with original weights)
        normal_saliency_abs: Normal Saliency with ABS (with original weights)
        random_saliency (list): Random Saliency without ABS (for each layer)
        random_saliency_abs (list): Random Saliency with ABS (for each layer)
        random_names (list): the layer names of the models

        Returns
        -------
        corr_value: correlation value for each layer in each methods (without ABS)
        corr_value_std: the standard deviation for the correlation value for each layer in each methods(without ABS)
        corr_value_abs: correlation value for each layer in each methods (with ABS)
        corr_value_std_abs: the standard deviation for the correlation value for each layer in each methods(with ABS)
        """
        ## Dictionary to save all the metrics
        rank_correlation_dict_norm_random = {}
#         rank_correlation_dict_norm_random_abs = {}
        ## initialize the dict appropriately
        for layer in random_names:
            rank_correlation_dict_norm_random[layer] = {}
#             rank_correlation_dict_norm_random_abs[layer] = {}
            for method in random_saliency[0].keys():
                rank_correlation_dict_norm_random[layer][method] = []
#                 rank_correlation_dict_norm_random_abs[layer][method] = []

        ## compare (layer-wise)
        for i, layer in enumerate(random_names):
            for key, method in zip(normal_saliency.keys(), random_saliency[i].keys()):
                for idx_sample in range(normal_saliency[key].shape[0]):  ## all test samples
                    normal_mask = normal_saliency[key][idx_sample]
#                     normal_mask_abs = normal_saliency_abs[key][idx_sample]
                    rand_random_mask = random_saliency[i][method][idx_sample]
#                     rand_random_mask_abs = random_saliency_abs[i][method][idx_sample]
                    ## compute rank correlation
                    rk_random_value, _ = spr(normal_mask,
                                             rand_random_mask,
                                             axis=None,
                                             nan_policy='raise')
#                     rk_random_value_abs, _ = spr(normal_mask_abs,
#                                                  rand_random_mask_abs,
#                                                  axis=None,
#                                                  nan_policy='raise')

                    ## collate all the value into their respective dictionaries
                    rank_correlation_dict_norm_random[layer][method].append(rk_random_value)
#                     rank_correlation_dict_norm_random_abs[layer][method].append(rk_random_value_abs)
        for layer in random_names:
            for method in rank_correlation_dict_norm_random[layer].keys():
#                 if len(rank_correlation_dict_norm_random_abs[layer][method]) != self.dataset.data.shape[0]:
#                     raise ValueError("the Correlation from ABS doesn't have the same length as the Dataset Data")
                if len(rank_correlation_dict_norm_random[layer][method]) != normal_saliency[method].shape[0]:
                    raise ValueError("the Correlation from non ABS doesn't have the same length as the Dataset Data")

        ## Spearman Correlation
        ## Mean and Std of each layer
        rk_mean_dict_random = {}
        rk_std_dict_random = {}
#         rk_mean_dict_random_abs = {}
#         rk_std_dict_random_abs = {}

        for key in rank_correlation_dict_norm_random.keys():  ## layers
            rk_mean_dict_random[key] = {}
            rk_std_dict_random[key] = {}
#             rk_mean_dict_random_abs[key] = {}
#             rk_std_dict_random_abs[key] = {}
            for key2 in rank_correlation_dict_norm_random[key].keys():  ## methods
                rk_mean_dict_random[key][key2] = np.mean(rank_correlation_dict_norm_random[key][key2])
                rk_std_dict_random[key][key2] = np.std(rank_correlation_dict_norm_random[key][key2])
#                 rk_mean_dict_random_abs[key][key2] = np.mean(rank_correlation_dict_norm_random_abs[key][key2])
#                 rk_std_dict_random_abs[key][key2] = np.std(rank_correlation_dict_norm_random_abs[key][key2])

        return rk_mean_dict_random, rk_std_dict_random

    def get_ssim(self,
                 normal_saliency,
#                  normal_saliency_abs,
                 random_saliency: list,
#                  random_saliency_abs: list,
                 random_names: list
                 ):
        """
        Parameters
        ----------
        normal_saliency: Normal Saliency without ABS (with original weights)
        normal_saliency_abs: Normal Saliency with ABS (with original weights)
        random_saliency (list): Random Saliency without ABS (for each layer)
        random_saliency_abs (list): Random Saliency with ABS (for each layer)
        random_names (list): the layer names of the models

        Returns
        -------
        ssim_value: correlation value for each layer in each methods (without ABS)
        ssim_value_std: the standard deviation for the correlation value for each layer in each methods(without ABS)
        ssim_value_abs: correlation value for each layer in each methods (with ABS)
        ssim_value_std_abs: the standard deviation for the correlation value for each layer in each methods(with ABS)
        """
        ## Dictionary to save all the metrics
        ssim_dictionary_norm_random = {}
#         ssim_dictionary_norm_random_abs = {}
        ## initialize the dict appropriately
        for layer in random_names:
            ssim_dictionary_norm_random[layer] = {}
#             ssim_dictionary_norm_random_abs[layer] = {}
            for method in random_saliency[0].keys():
                ssim_dictionary_norm_random[layer][method] = []
#                 ssim_dictionary_norm_random_abs[layer][method] = []

        ## compare (layer-wise)
        for i, layer in enumerate(random_names):
            for key, method in zip(normal_saliency.keys(), random_saliency[i].keys()):
                for idx_sample in range(normal_saliency[key].shape[0]):  ## all test samples
                    normal_mask = normal_saliency[key][idx_sample]
#                     normal_mask_abs = normal_saliency_abs[key][idx_sample]
                    rand_random_mask = random_saliency[i][method][idx_sample]
#                     rand_random_mask_abs = random_saliency_abs[i][method][idx_sample]

                    ## compute structural similarity
                    normal_mask_ = normal_mask.transpose((1, 0))
#                     normal_mask_abs_ = normal_mask_abs.transpose((1, 0))
                    rand_random_mask_ = rand_random_mask.transpose((1, 0))
#                     rand_random_mask_abs_ = rand_random_mask_abs.transpose((1, 0))

                    ss1_random = ssim(normal_mask_, rand_random_mask_,
                                      multichannel=True, gaussian_weights=True)
#                     ss1_random_abs = ssim(normal_mask_abs_, rand_random_mask_abs_,
#                                       multichannel=True, gaussian_weights=True)

                    ## collate all the value into their respective dictionaries
                    ssim_dictionary_norm_random[layer][method].append(ss1_random)
#                     ssim_dictionary_norm_random_abs[layer][method].append(ss1_random_abs)
        for layer in random_names:
            for method in ssim_dictionary_norm_random[layer].keys():
                if len(ssim_dictionary_norm_random[layer][method]) != normal_saliency[method].shape[0]:
                    raise ValueError("the SSIM for non ABS doesn't have the same length as the testset Data")
#                 if len(ssim_dictionary_norm_random_abs[layer][method]) != self.dataset.data.shape[0]:
#                     raise ValueError("the SSIM for ABS doesn't have the same length as the testset Data")

        ## Structural Similarity (SSIM)
        ## Mean and Std of each layer
        ssim_mean_dict_random = {}
        ssim_std_dict_random = {}
#         ssim_mean_dict_random_abs = {}
#         ssim_std_dict_random_abs = {}

        for key in ssim_dictionary_norm_random.keys():  ## layers
            ssim_mean_dict_random[key] = {}
            ssim_std_dict_random[key] = {}
#             ssim_mean_dict_random_abs[key] = {}
#             ssim_std_dict_random_abs[key] = {}
            for key2 in ssim_dictionary_norm_random[key].keys():  ## methods
                ssim_mean_dict_random[key][key2] = np.mean(ssim_dictionary_norm_random[key][key2])
                ssim_std_dict_random[key][key2] = np.std(ssim_dictionary_norm_random[key][key2])
#                 ssim_mean_dict_random_abs[key][key2] = np.mean(ssim_dictionary_norm_random_abs[key][key2])
#                 ssim_std_dict_random_abs[key][key2] = np.std(ssim_dictionary_norm_random_abs[key][key2])

        return ssim_mean_dict_random, ssim_std_dict_random