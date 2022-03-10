"""
Explanation Sensitivity Measures
Sensitivity-Max
References: https://arxiv.org/pdf/1901.09392.pdf
also from Captum: https://captum.ai/api/metrics.html

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
from metrics.sensitivity_max import sensitivity_test
from trainhelper.dataset import Dataset


def sensitivity_test_eval(args,
                          model,
                          dataset,
                          vis_methods: list,
                          use_tsr: bool = False,
                          save_scores_path: str = None
                          ):
    ## setting args
    perturb_radius = args.Perturb_radius
    n_perturb_samples = args.N_perturb_samples

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

    ## compute sensitivity scores
    sensitivity_scores = {}
    for radius in perturb_radius:
        scores = {}
        for method in vis_methods:
            scores[method] = t.zeros(len(normal_labels))
        for i in range(len(normal_labels)):
            sample = normal_dataset.data[i]
            sample = sample.reshape(1, sample.shape[0], sample.shape[1])

            if "grads" in vis_methods:
                scores["grads"][i] = sensitivity_test(
                    explanation_function=saliency_functions.getGradientSaliency,
                    inputs=sample,
                    perturb_radius=radius,
                    n_perturb_samples=n_perturb_samples,
                    max_examples_per_batch=n_perturb_samples,
                    label=normal_dataset.labels[i],
                    absolute=True
                )
            if "igs" in vis_methods:
                scores["igs"][i] = sensitivity_test(
                    explanation_function=saliency_functions.getIntegratedGradients,
                    inputs=sample,
                    perturb_radius=radius,
                    n_perturb_samples=n_perturb_samples,
                    max_examples_per_batch=n_perturb_samples,
                    label=normal_dataset.labels[i],
                    absolute=True,
                    ig_steps=60
                )
            if "smoothgrads" in vis_methods:
                scores["smoothgrads"][i] = sensitivity_test(
                    explanation_function=saliency_functions.getSmoothGradients,
                    inputs=sample,
                    perturb_radius=radius,
                    n_perturb_samples=n_perturb_samples,
                    max_examples_per_batch=n_perturb_samples,
                    label=normal_dataset.labels[i],
                    absolute=True,
                    nt_samples=60,
                    stdevs=0.1
                )
            if "lrp_epsilon" in vis_methods and dl_selected_model not in ["LSTM"]:
                scores["lrp_epsilon"][i] = sensitivity_test(
                    explanation_function=saliency_functions.getLRP,
                    inputs=sample,
                    perturb_radius=radius,
                    n_perturb_samples=n_perturb_samples,
                    max_examples_per_batch=n_perturb_samples,
                    label=normal_dataset.labels[i],
                    absolute=False,
                    rule="epsilon"
                )
            elif "lrp_epsilon" in vis_methods and dl_selected_model in ["LSTM"]:
                scores["lrp_epsilon"][i] = sensitivity_test(
                    explanation_function=saliency_functions.getLRP4LSTM,
                    inputs=sample,
                    perturb_radius=radius,
                    n_perturb_samples=n_perturb_samples,
                    max_examples_per_batch=n_perturb_samples,
                    label=normal_dataset.labels[i],
                    absolute=False
                )
    #         if "lrp_gamma" in vis_methods:
    #             sensitivity_scores["lrp_gamma"][i] = sensitivity_test(
    #                 explanation_function=saliency_functions.getLRP,
    #                 inputs=sample,
    #                 perturb_radius=perturb_radius,
    #                 n_perturb_samples=n_perturb_samples,
    #                 max_examples_per_batch=n_perturb_samples,
    #                 label=normal_dataset.labels[i],
    #                 absolute=False,
    #                 rule="gamma"
    #             )
            if "gradCAM" in vis_methods:
                scores["gradCAM"][i] = sensitivity_test(
                    explanation_function=saliency_functions.getGradCAM,
                    inputs=sample,
                    perturb_radius=radius,
                    n_perturb_samples=n_perturb_samples,
                    max_examples_per_batch=n_perturb_samples,
                    label=normal_dataset.labels[i],
                    absolute=True,
                    use_relu=True,
                    upsample_to_input=True,
                    layer_to_grad="gap_softmax.conv1",
                    attribute_to_layer_input=True
                )
            if "guided_gradcam" in vis_methods:
                scores["guided_gradcam"][i] = sensitivity_test(
                    explanation_function=saliency_functions.getGuidedGradCAM,
                    inputs=sample,
                    perturb_radius=radius,
                    n_perturb_samples=n_perturb_samples,
                    max_examples_per_batch = n_perturb_samples,
                    label=normal_dataset.labels[i],
                    absolute=False,
                    layer_to_grad="gap_softmax.conv1",
                    attribute_to_layer_input=True
                )
            if "guided_backprop" in vis_methods:
                scores["guided_backprop"][i] = sensitivity_test(
                    explanation_function=saliency_functions.getGuidedBackprop,
                    inputs=sample,
                    perturb_radius=radius,
                    n_perturb_samples=n_perturb_samples,
                    max_examples_per_batch=n_perturb_samples,
                    label=normal_dataset.labels[i],
                    absolute=False
                )
            if "lime" in vis_methods:
                scores["lime"][i] = sensitivity_test(
                    explanation_function=saliency_functions.getLIME,
                    inputs=sample,
                    perturb_radius=radius,
                    n_perturb_samples=n_perturb_samples,
                    max_examples_per_batch=n_perturb_samples,
                    label=normal_dataset.labels[i],
                    absolute=False,
                    n_sample=1200,
                    num_features=125,
                    baseline="total_mean",
                    kernel_width=5.0
                )
            if "kernel_shap" in vis_methods:
                scores["kernel_shap"][i] = sensitivity_test(
                    explanation_function=saliency_functions.getKernelSHAP,
                    inputs=sample,
                    perturb_radius=radius,
                    n_perturb_samples=n_perturb_samples,
                    max_examples_per_batch=n_perturb_samples,
                    label=normal_dataset.labels[i],
                    absolute=False,
                    n_sample=1200,
                    baseline="total_mean",
                    num_features=125
                )
            if "random" in vis_methods:
                scores["random"][i] = sensitivity_test(
                    explanation_function=saliency_functions.getRandomSaliency,
                    inputs=sample,
                    perturb_radius=radius,
                    n_perturb_samples=n_perturb_samples,
                    max_examples_per_batch=n_perturb_samples,
                    label=normal_dataset.labels[i],
                    absolute=True
                )
        sensitivity_scores[str(radius)] = scores
        
    if save_scores_path is not None:
        path_2_save = root_dir + "results/" + dataset_name_save + "/" + dl_selected_model + "/" + experiment + "/sensitivity_max/"
        name_scores = path_2_save + dl_selected_model + "_" + dataset_name_save + "_stability_sm_random_" + save_scores_path
        np.save(name_scores, sensitivity_scores)
    mean_sensitivity_score = {}
    std_sensitivity_score = {}
    
    for key in sensitivity_scores.keys():
        mean_sensitivity_score[key] = {}
        std_sensitivity_score[key] = {}
        for method in sensitivity_scores[key].keys():
            mean_sensitivity_score[key][method] = np.mean(sensitivity_scores[key][method].cpu().detach().numpy())
            std_sensitivity_score[key][method] = np.std(sensitivity_scores[key][method].cpu().detach().numpy())

    return mean_sensitivity_score, std_sensitivity_score

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    # parser.add_argument("--Root_Dir", type=str, default='../')
    parser.add_argument("--Dataset_name", type=str, default='GunPointAgeSpan')
    parser.add_argument("--Dataset_name_save", type=str, default='GunPointAgeSpan_Cluster')
    parser.add_argument("--Experiments", nargs='+', default='experiment_0')
    parser.add_argument("--DLModel", type=str, default='LSTM')

    parser.add_argument("--Perturb_radius", nargs='+', type=float, default=0.05)
    parser.add_argument("--N_perturb_samples", type=int, default=10)
    parser.add_argument("--Use_tsr", action="store_true", default=False)
    parser.add_argument("--Save_scores_path", type=str, default='scores.npy')

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
    mean_scores, std_scores = sensitivity_test_eval(args=args,
                                                    model=models[0],
                                                    dataset=testsets[0],
                                                    vis_methods=methods,
                                                    use_tsr=use_tsr,
                                                    save_scores_path=save_scores_path)