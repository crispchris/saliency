"""
Explanation Sensitivity Measures
Class Sensitivity
References: https://openaccess.thecvf.com/content_CVPR_2020/papers/Rebuffi_There_and_Back_Again_Revisiting_Backpropagation_Saliency_Methods_CVPR_2020_paper.pdf
also from: https://export.arxiv.org/abs/2012.15616
"""

## -------------------
## --- Third-Party ---
## -------------------
import os
import sys
sys.path.append('../../')
fileDir = os.path.dirname(os.path.abspath(__file__))
parentDir = os.path.dirname(fileDir)
sys.path.append(parentDir)
import argparse
import numpy as np
import torch as t

## -----------
## --- Own ---
## -----------
from visual_interpretability import load_data_and_models
from utils import throw_out_wrong_classified
from visualize_mechanism.visual_func import SaliencyFunctions
from visualize_mechanism.tsr import tsr_from_saliencymap
from trainhelper.dataset import Dataset
from metrics.class_sensitivity import class_sensitivity

def class_sensitivity_evaluation(args,
                                 model,
                                 dataset,
                                 vis_methods: list,
                                 use_tsr: bool = False,
                                 save_scores_path: str = None
                                 ):
    ## setting args
    root_dir = parentDir + '/../'
    # dataset_name = args.Dataset_name
    dataset_name_save = args.Dataset_name_save
    dl_selected_model = args.DLModel
    experiment = args.Experiments
    experiment = experiment[0]

    ## clean dataset
    normal_data, normal_labels = throw_out_wrong_classified(model=model,
                                                            data=dataset.data,
                                                            labels=dataset.labels,
                                                            )
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    normal_data = t.tensor(normal_data).float().to(device)
    normal_labels = t.tensor(normal_labels)
    normal_dataset = Dataset(normal_data, normal_labels)

    ## get tsr
    if use_tsr:
        rescaledSaliency = tsr_from_saliencymap(
            samples=dataset.data,
            labels=dataset.labels,
            dl_model=model,
            time_groups=5,
            feature_groups=1,
            threshold=0.5
        )
        rescaledSaliency = t.tensor(rescaledSaliency).float().to(device)
        rescaledSaliency = rescaledSaliency.unsqueeze(0)
    else:
        rescaledSaliency = None

    ## set saliency functions
    saliency_functions = SaliencyFunctions(model=model,
                                           tsr_saliency=rescaledSaliency,
                                           device=device)

    max_labels = t.zeros((len(normal_dataset.labels)))
    min_labels = t.zeros((len(normal_dataset.labels)))
    for i in range(len(normal_dataset.labels)):
        sample = normal_dataset.data[i]
        sample = sample.reshape(1, *sample.shape)
        prediction = model(sample)
        sort_maxmin = t.argsort(prediction, dim=1, descending=True)
        max_labels[i] = sort_maxmin[0][0]
        min_labels[i] = sort_maxmin[0][-1]

    ## correlation scores calculation
    correlation_scores_dict = {}
    ## max class explanations
    ## min class explanations
    max_explanations = {}
    min_explanations = {}

    if "grads" in vis_methods:
        correlation_scores_dict["grads"], max_explanations["grads"], min_explanations["grads"] = class_sensitivity(
            explanation_function=saliency_functions.getGradientSaliency,
            explanation_name="grads",
            inps=normal_dataset.data,
            max_classes=max_labels,
            min_classes=min_labels,
            absolute=False
        )
    if "igs" in vis_methods:
        correlation_scores_dict["igs"], max_explanations["igs"], min_explanations["igs"] = class_sensitivity(
            explanation_function=saliency_functions.getIntegratedGradients,
            explanation_name="igs",
            inps=normal_dataset.data,
            max_classes=max_labels,
            min_classes=min_labels,
            absolute=False,
            ig_steps=60
        )
    if "smoothgrads" in vis_methods:
        correlation_scores_dict["smoothgrads"], max_explanations["smoothgrads"], min_explanations["smoothgrads"] = class_sensitivity(
            explanation_function=saliency_functions.getSmoothGradients,
            explanation_name="smoothgrads",
            inps=normal_dataset.data,
            max_classes=max_labels,
            min_classes=min_labels,
            absolute=False,
            nt_samples=60,
            stdevs=0.1
        )

    if "lrp_epsilon" in vis_methods and dl_selected_model not in ["LSTM"]:
        correlation_scores_dict["lrp_epsilon"], max_explanations["lrp_epsilon"], min_explanations["lrp_epsilon"] = class_sensitivity(
            explanation_function=saliency_functions.getLRP,
            explanation_name="lrp_epsilon",
            inps=normal_dataset.data,
            max_classes=max_labels,
            min_classes=min_labels,
            absolute=False,
            rule="epsilon"
        )
    elif "lrp_epsilon" in vis_methods and dl_selected_model in ["LSTM"]:
        correlation_scores_dict["lrp_epsilon"], max_explanations["lrp_epsilon"], min_explanations["lrp_epsilon"] = class_sensitivity(
            explanation_function=saliency_functions.getLRP4LSTM,
            explanation_name="lrp_epsilon",
            inps=normal_dataset.data,
            max_classes=max_labels,
            min_classes=min_labels,
            absolute=False
        )
#     if "lrp_gamma" in vis_methods:
#         correlation_scores_dict["lrp_gamma"], max_explanations["lrp_gamma"], min_explanations["lrp_gamma"] = class_sensitivity(
#             explanation_function=saliency_functions.getLRP,
#             inps=normal_dataset.data,
#             max_classes=max_labels,
#             min_classes=min_labels,
#             absolute=False,
#             rule="gamma"
#         )

    if "gradCAM" in vis_methods:
        correlation_scores_dict["gradCAM"], max_explanations["gradCAM"], min_explanations["gradCAM"] = class_sensitivity(
            explanation_function=saliency_functions.getGradCAM,
            explanation_name="gradCAM",
            inps=normal_dataset.data,
            max_classes=max_labels,
            min_classes=min_labels,
            absolute=False,
            use_relu=True,
            upsample_to_input=True,
            layer_to_grad="gap_softmax.conv1",
            attribute_to_layer_input=True
        )
    if "guided_gradcam" in vis_methods:
        correlation_scores_dict["guided_gradcam"], max_explanations["guided_gradcam"], min_explanations["guided_gradcam"] = class_sensitivity(
            explanation_function=saliency_functions.getGuidedGradCAM,
            explanation_name="guided_gradcam",
            inps=normal_dataset.data,
            max_classes=max_labels,
            min_classes=min_labels,
            absolute=False,
            layer_to_grad="gap_softmax.conv1",
            attribute_to_layer_input=True
        )
    if "guided_backprop" in vis_methods:
        correlation_scores_dict["guided_backprop"], max_explanations["guided_backprop"], min_explanations["guided_backprop"] = class_sensitivity(
            explanation_function=saliency_functions.getGuidedBackprop,
            explanation_name="guided_backprop",
            inps=normal_dataset.data,
            max_classes=max_labels,
            min_classes=min_labels,
            absolute=False
        )
    if "lime" in vis_methods:
        correlation_scores_dict["lime"], max_explanations["lime"], min_explanations["lime"] = class_sensitivity(
            explanation_function=saliency_functions.getLIME,
            explanation_name="lime",
            inps=normal_dataset.data,
            max_classes=max_labels,
            min_classes=min_labels,
            absolute=False,
            n_sample=1200,
            num_features=125,
            baseline="total_mean",
            kernel_width=5.0
        )
    if "kernel_shap" in vis_methods:
        correlation_scores_dict["kernel_shap"], max_explanations["kernel_shap"], min_explanations["kernel_shap"] = class_sensitivity(
            explanation_function=saliency_functions.getKernelSHAP,
            explanation_name="kernel_shap",
            inps=normal_dataset.data,
            max_classes=max_labels,
            min_classes=min_labels,
            absolute=False,
            n_sample=1200,
            baseline="total_mean",
            num_features=125
        )
    if "random" in vis_methods:
        ### max and min explanation are just random from uniform distribution
        correlation_scores_dict["random"], max_explanations["random"], min_explanations["random"] = class_sensitivity(
            explanation_function=saliency_functions.getRandomSaliency,
            explanation_name="random",
            inps=normal_dataset.data,
            max_classes=max_labels,
            min_classes=min_labels,
            absolute=True
        )

    if save_scores_path is not None:
        path_2_save = root_dir + "results/" + dataset_name_save + "/" + dl_selected_model + "/" + experiment + "/class_sensitivity/"
        name_scores = path_2_save + dl_selected_model + "_" + dataset_name_save + "_sensitivity_cs_random_" + save_scores_path
        name_max_explain = path_2_save + dl_selected_model + "_" + dataset_name_save + "_sensitivity_cs_max_random_explanation.npy"
        name_min_explain = path_2_save + dl_selected_model + "_" + dataset_name_save + "_sensitivity_cs_min_random_explanation.npy"
        np.save(name_scores, correlation_scores_dict)
        np.save(name_max_explain, max_explanations)
        np.save(name_min_explain, min_explanations)

    return correlation_scores_dict

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    # parser.add_argument("--Root_Dir", type=str, default='../')
    parser.add_argument("--Dataset_name", type=str, default='GunPointAgeSpan')
    parser.add_argument("--Dataset_name_save", type=str, default='GunPointAgeSpan_Cluster')
    parser.add_argument("--Experiments", nargs='+', default='experiment_0')
    parser.add_argument("--DLModel", type=str, default='TCN_laststep')
    parser.add_argument("--Use_tsr", action="store_true", default=False)
    parser.add_argument("--Save_scores_path", type=str, default='correlation.npy')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])
    print("Load Data and Model")
    testsets, models, model_softmaxs, saliency_constructor_gcs, saliency_constructors = load_data_and_models(
        args=args
    )
    
    root_dir = parentDir + '/../'
    dataset_name = args.Dataset_name
    dataset_name_save = args.Dataset_name_save
    dl_selected_model = args.DLModel
    
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    if dl_selected_model not in ['LSTM', 'MLP']:
        #methods = ["grads",
        #   "smoothgrads",
        #   "igs",
        #   "lrp_epsilon",
        #   "gradCAM",
        #   "guided_gradcam",
        #   "guided_backprop",
        #   "lime",
        #   "kernel_shap"]
        methods = ["random"]
    else:
        #methods = ["grads",
        #           "smoothgrads",
        #           "igs",
        #           "lrp_epsilon",
        #           "lime",
        #           "kernel_shap"]
        methods = ["random"]
        # methods = ["grads",
        #            "smoothgrads",
        #            "igs",
        #            "lrp_epsilon"]

    save_scores_path = args.Save_scores_path
    use_tsr = args.Use_tsr
    correlation = class_sensitivity_evaluation(args=args,
                                               model=models[0],
                                               dataset=testsets[0],
                                               vis_methods=methods,
                                               use_tsr=use_tsr,
                                               save_scores_path=save_scores_path)
