"""
Evaluation:
    Sanity Check:
    https://github.com/adebayoj/sanity_checks_saliency/tree/3e24048c570f08ca655fcd332b6128fa069810a0
    Model Randomization Test
"""
## ------------------
## --- Third-Party ---
## ------------------
import os
import sys
sys.path.append('../../')
fileDir = os.path.dirname(os.path.abspath(__file__))
parentDir = os.path.dirname(fileDir)
sys.path.append(parentDir)
import argparse
import numpy as np
import pandas as pd
import torch as t
import matplotlib.pyplot as plt
import seaborn as sns


### -----------
### --- Own ---
### -----------
from utils import load_saliencies
from metrics.sanity_check import SanityCheck
from visual_interpretability import load_data_and_models

def clean_normal_saliency(model,
                          dataset,
                          normal_saliency,
#                           normal_saliency_abs
                         ):
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    normal_saliency_clean = {}
#     normal_saliency_abs_clean = {}
    ## clean the normal saliency for the correct classification
    data = t.tensor(dataset.data).float().to(device)
    labels = t.tensor(dataset.labels)
    print(f"[INFO] [before throwing wrong classified] The number of data :{len(labels)}")
    predicted = t.zeros(labels.shape)
    with t.no_grad():
        i = 0
        for d, l in zip(data, labels):
            d = d.reshape((1, *d.shape))
            l = l.reshape((-1, 1))
            ## Forward pass
            prediction = model(d)
            predicted[i] = t.argmax(prediction, dim=1)
            i += 1
    mask = labels == predicted
    for method in normal_saliency.keys():
        normal_saliency_clean[method] = normal_saliency[method][mask]
#         normal_saliency_abs_clean[method] = normal_saliency_abs[method][mask]
    print(f"[INFO] [after throwing wrong classified] The number of data :{len(normal_saliency_clean[method])}")
#     return normal_saliency_clean, normal_saliency_abs_clean
    return normal_saliency_clean

def saliency_sanitycheck(args,
                         models,
                         datasets,
                         normal_saliency,
#                          normal_saliency_abs,
                         methods: list,
                         save_randsaliency: bool = False):
    root_dir = parentDir + '/../'
    dataset_name = args.Dataset_name
    dataset_name_save = args.Dataset_name_save
    dl_selected_model = args.DLModel
    experiment = args.Experiments
    experiment = experiment[0]
    mode = args.Evaluation_mode
    use_randommaps = args.use_randommaps
    model = models[0]
    random_model = models[0]
    dataset = datasets[0]
    
    ## clean out the saliency with wrong classified
    normal_saliency = clean_normal_saliency(model=model,
                                            dataset=dataset,
                                            normal_saliency=normal_saliency)

    sanitycheck_object = SanityCheck(model=model,
                                     random_model=random_model,
                                     dataset=dataset,
                                     mode=mode)

    path_2_parameters = root_dir + "results/" + dataset_name_save + "/" + dl_selected_model + "/" + experiment + "/"
    report = pd.read_csv(path_2_parameters + "reports.csv")
    ## model setting and loading from checkpoint
    if int(report["best_epoch"][0]) >= 100:
        ckp_path = path_2_parameters + "checkpoints/checkpoint_{}.ckp".format(report["best_epoch"][0])
    else:
        ckp_path = path_2_parameters + "checkpoints/checkpoint_0{}.ckp".format(report["best_epoch"][0])

    sanitycheck_object.load_ckp(ckp_path=ckp_path)
    rand_sali, rand_sali_abs, rand_names, rand_acc_dict = sanitycheck_object.get_random_saliencies(
        absolute=False,
        use_tsr=args.Use_tsr,
        ckp_path=ckp_path,
        vis_methods=methods)
    if save_randsaliency:
        path_2_save = root_dir + "results/" + dataset_name_save + "/" + dl_selected_model + "/" + experiment + "/sanity_check/"
        name_acc = path_2_save + dl_selected_model + "_" + dataset_name_save + "_sanity_" + mode + "_random_accuracy_layers.npy"
        np.save(name_acc, rand_acc_dict)
        for i, rand_name in enumerate(rand_names):
            name = path_2_save + dl_selected_model + "_" + dataset_name_save + "_sanity_" + mode + "_random_saliencymaps_randombaseline" + rand_name + "_non_abs.npy"
            name_abs = path_2_save + dl_selected_model + "_" + dataset_name_save + "_sanity_" + mode + "_random_saliencymaps_randombaseline" + rand_name + "_abs.npy"
            np.save(name, rand_sali[i])
            np.save(name_abs, rand_sali_abs[i])

    rand_sali_list = []

    for i, rand_name in enumerate(rand_names):
        rand_sali_dict = {}
        for method in methods:
            if use_randommaps:
                rand_sali_dict[method] = rand_sali_abs[i][method]
            else:
                if method in absolute_methods:
                    rand_sali_dict[method] = rand_sali_abs[i][method]
                else:
                    rand_sali_dict[method] = rand_sali[i][method]
        rand_sali_list.append(rand_sali_dict)

    random_correlation_stat = {}
    rand_corr, rand_corr_std = sanitycheck_object.get_spearman_correlation(
        normal_saliency= normal_saliency,
#         normal_saliency_abs=normal_saliency_abs,
        random_saliency=rand_sali_list,
#         random_saliency_abs=rand_sali_abs,
        random_names=rand_names
    )
    random_ssim_stat = {}
    rand_ssim, rand_ssim_std = sanitycheck_object.get_ssim(
        normal_saliency= normal_saliency,
#         normal_saliency_abs=normal_saliency_abs,
        random_saliency=rand_sali_list,
#         random_saliency_abs=rand_sali_abs,
        random_names=rand_names
    )
    random_correlation_stat['corr'] = rand_corr
    random_correlation_stat['corr_std'] = rand_corr_std
    # random_correlation_stat['corr_abs'] = rand_corr_abs
    # random_correlation_stat['corr_std_abs'] = rand_corr_std_abs

    random_ssim_stat['ssim'] = rand_ssim
    random_ssim_stat['ssim_std'] = rand_ssim_std
    # random_ssim_stat['ssim_abs'] = rand_ssim_abs
    # random_ssim_stat['ssim_std_abs'] = rand_ssim_std_abs

    path_2_save = root_dir + "results/" + dataset_name_save + "/" + dl_selected_model + "/" + experiment + "/sanity_check/"
    name_corr = path_2_save + dl_selected_model + "_" + dataset_name_save + "_sanity_" + mode + "_correlation_random.npy"
    name_ssim = path_2_save + dl_selected_model + "_" + dataset_name_save + "_sanity_" + mode + "_ssim_random.npy"
    np.save(name_corr, random_correlation_stat)
    np.save(name_ssim, random_ssim_stat)

    return random_correlation_stat, random_ssim_stat, rand_names, rand_acc_dict

def plot_statistic(args,
                   rand_correlation_stat,
                   rand_ssim_stat,
                   rand_names,
                   rand_acc_dict,
                   vis_methods: list,
                   title: str):
    corr_df = pd.DataFrame(rand_correlation_stat['corr'])
    corr_df_std = pd.DataFrame(rand_correlation_stat['corr_std'])
    corr_df_abs = pd.DataFrame(rand_correlation_stat['corr_abs'])
    corr_df_std_abs = pd.DataFrame(rand_correlation_stat['corr_std_abs'])
    acc_df_random = pd.DataFrame(rand_acc_dict)

    corr_df["Original"] = [1.0] * corr_df.shape[0]
    corr_df_std["Original"] = [0.0] * corr_df_std.shape[0]
    corr_df_abs["Original"] = [1.0] * corr_df_abs.shape[0]
    corr_df_std_abs["Original"] = [0.0] * corr_df_std_abs.shape[0]

    layer_order = ["Original"]
    for val in rand_names:
        layer_order.append(val)
    corr_df = corr_df.reindex(columns=layer_order)
    corr_df_std = corr_df_std.reindex(columns=layer_order)
    corr_df_abs = corr_df_abs.reindex(columns=layer_order)
    corr_df_std_abs = corr_df_std_abs.reindex(columns=layer_order)

    acc_df_random = acc_df_random.reindex(columns=layer_order)

    ## correlation
    ## no abs
    no_abs_title = title + '_no_abs_correlation'
    show_stats(args=args,
               rk_df=corr_df,
               rk_df_std=corr_df_std,
               acc_df=acc_df_random,
               layer_order=layer_order,
               vis_methods=vis_methods,
               title=no_abs_title,
               ytitle='Correlation',
               save_plot=True)
    ## abs
    abs_title = title + '_abs_correlation'
    show_stats(args=args,
               rk_df=corr_df_abs,
               rk_df_std=corr_df_std_abs,
               acc_df=acc_df_random,
               layer_order=layer_order,
               vis_methods=vis_methods,
               title=abs_title,
               ytitle='Correlation',
               save_plot=True)

    ssim_df = pd.DataFrame(rand_ssim_stat['ssim'])
    ssim_df_std = pd.DataFrame(rand_ssim_stat['ssim_std'])
    ssim_df_abs = pd.DataFrame(rand_ssim_stat['ssim_abs'])
    ssim_df_std_abs = pd.DataFrame(rand_ssim_stat['ssim_std_abs'])

    ssim_df["Original"] = [1.0] * ssim_df.shape[0]
    ssim_df_std["Original"] = [0.0] * ssim_df_std.shape[0]
    ssim_df_abs["Original"] = [1.0] * ssim_df_abs.shape[0]
    ssim_df_std_abs["Original"] = [0.0] * ssim_df_std_abs.shape[0]

    layer_order = ["Original"]
    for val in rand_names:
        layer_order.append(val)
    ssim_df = ssim_df.reindex(columns=layer_order)
    ssim_df_std = ssim_df_std.reindex(columns=layer_order)
    ssim_df_abs = ssim_df_abs.reindex(columns=layer_order)
    ssim_df_std_abs = ssim_df_std_abs.reindex(columns=layer_order)
    ## ssim
    ## no abs
    no_abs_title = title + '_no_abs_ssim'
    show_stats(args=args,
               rk_df=ssim_df,
               rk_df_std=ssim_df_std,
               acc_df=acc_df_random,
               layer_order=layer_order,
               vis_methods=vis_methods,
               title=no_abs_title,
               ytitle='SSIM',
               save_plot=True)
    ## abs
    abs_title = title + '_abs_ssim'
    show_stats(args=args,
               rk_df=ssim_df_abs,
               rk_df_std=ssim_df_std_abs,
               acc_df=acc_df_random,
               layer_order=layer_order,
               vis_methods=vis_methods,
               title=abs_title,
               ytitle='SSIM',
               save_plot=True)



def show_stats(args,
               rk_df,
               rk_df_std,
               acc_df,
               layer_order,
               vis_methods: list,
               title,
               ytitle = "Correlation",
               save_plot: bool = True):
    sns.set(style="ticks")
    fig, ax = plt.subplots(figsize=(64, 32), dpi=200)
    x = [i + 1 for i in range(len(layer_order))]

    if "grads" in vis_methods:
        ax.plot(x, rk_df.loc['grads', :].values,
                'ro-', lw=3, linestyle='--', label='Gradient')
        ax.fill_between(x, rk_df.loc['grads', :].values - rk_df_std.loc['grads', :].values,
                        rk_df.loc['grads', :].values + rk_df_std.loc['grads', :].values,
                        facecolor='r', alpha=0.05)
    if "smoothgrads" in vis_methods:
        ax.plot(x, rk_df.loc['smoothgrads', :].values,
                'go-', lw=3, linestyle='--', label='Smooth Gradients')
        ax.fill_between(x, rk_df.loc['smoothgrads', :].values - rk_df_std.loc['smoothgrads', :].values,
                        rk_df.loc['smoothgrads', :].values + rk_df_std.loc['smoothgrads', :].values,
                        facecolor='g', alpha=0.05)
    if "igs" in vis_methods:
        ax.plot(x, rk_df.loc['igs', :].values,
                'co-', lw=3, linestyle='--', label='Integrated Gradients')
        ax.fill_between(x, rk_df.loc['igs', :].values - rk_df_std.loc['igs', :].values,
                        rk_df.loc['igs', :].values + rk_df_std.loc['igs', :].values,
                        facecolor='c', alpha=0.05)
    if "lrp_epsilon" in vis_methods:
        ax.plot(x, rk_df.loc['lrp_epsilon', :].values,
                'mo-', lw=3, linestyle='--', label='LRP epsilon')
        ax.fill_between(x, rk_df.loc['lrp_epsilon', :].values - rk_df_std.loc['lrp_epsilon', :].values,
                        rk_df.loc['lrp_epsilon', :].values + rk_df_std.loc['lrp_epsilon', :].values,
                        facecolor='m', alpha=0.05)

    if "lrp_gamma" in vis_methods:
        ax.plot(x, rk_df.loc['lrp_gamma', :].values,
                color='orange', marker='o', lw=3, linestyle='--', label='LRP gamma')
        ax.fill_between(x, rk_df.loc['lrp_gamma', :].values - rk_df_std.loc['lrp_gamma', :].values,
                        rk_df.loc['lrp_gamma', :].values + rk_df_std.loc['lrp_gamma', :].values,
                        facecolor='orange', alpha=0.05)
    if "gradCAM" in vis_methods:
        ax.plot(x, rk_df.loc['gradCAM', :].values,
                'bo-', lw=3, linestyle='--', label='GradCAM')
        ax.fill_between(x, rk_df.loc['gradCAM', :].values - rk_df_std.loc['gradCAM', :].values,
                        rk_df.loc['gradCAM', :].values + rk_df_std.loc['gradCAM', :].values,
                        facecolor='b', alpha=0.05)
    if "guided_gradcam" in vis_methods:
        ax.plot(x, rk_df.loc['guided_gradcam', :].values,
                'yo-', lw=3, linestyle='--', label='Guided GradCAM')
        ax.fill_between(x, rk_df.loc['guided_gradcam', :].values - rk_df_std.loc['guided_gradcam', :].values,
                        rk_df.loc['guided_gradcam', :].values + rk_df_std.loc['guided_gradcam', :].values,
                        facecolor='y', alpha=0.05)
    if "guided_backprop" in vis_methods:
        ax.plot(x, rk_df.loc['guided_backprop', :].values,
                'ko-', lw=3, linestyle='--', label='Guided Backprop')
        ax.fill_between(x,
                        rk_df.loc['guided_backprop', :].values - rk_df.loc['guided_backprop', :].values,
                        rk_df.loc['guided_backprop', :].values + rk_df.loc['guided_backprop', :].values,
                        facecolor='k', alpha=0.05)
    if "lime" in vis_methods:
        ax.plot(x, rk_df.loc['lime', :].values,
                color='sienna', marker='x', lw=3, linestyle='--', label='Lime')
        ax.fill_between(x, rk_df.loc['lime', :].values - rk_df_std.loc['lime', :].values,
                        rk_df.loc['lime', :].values + rk_df_std.loc['lime', :].values,
                        facecolor='sienna', alpha=0.05)
    if "kernel_shap" in vis_methods:
        ax.plot(x, rk_df.loc['kernel_shap', :].values,
                color='lawngreen', marker='x', lw=3, linestyle='--', label='KernelShap')
        ax.fill_between(x, rk_df.loc['kernel_shap', :].values - rk_df_std.loc['kernel_shap', :].values,
                        rk_df.loc['kernel_shap', :].values + rk_df_std.loc['kernel_shap', :].values,
                        facecolor='lawngreen', alpha=0.05)


    ## in the same plot add accuracy (second axis)
    ax2 = ax.twinx()
    ax2.plot(x, acc_df.iloc[0].values, marker='x')
    ax2.set_ylabel("Accuracy")

    ax.set_xticks(x)
    ax.set_xticklabels(layer_order)
    plt.setp(ax.get_xticklabels(), rotation=90,
             ha="right",
             rotation_mode="anchor")
    ax.axhline(y=0.0, color='r', linestyle='--')
    ax.axvline(x=2.0, ymin=0.0, ymax=1.0, color='k', linestyle='--')
    ax.set_ylim([-0.5, 1.2])
    # plt.xlim([0.5, 4.5])
    ax.set_ylabel(ytitle)
    ax.set_title(title)
    # plt.legend(loc=8, ncol=2, fontsize=7, frameon=False)
    ax.tick_params(axis='x', which='both', top='off')

    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -1.0), ncol=3)

    if save_plot:
        root_dir = parentDir + '/../'
        dataset_name = args.Dataset_name
        dataset_name_save = args.Dataset_name_save
        dl_selected_model = args.DLModel
        mode = args.Evaluation_mode
        experiment = args.Experiments
        experiment = experiment[0]
        path_2_save = root_dir + "results/" + dataset_name_save + "/" + dl_selected_model + "/" + experiment + "/sanity_check"
        name = path_2_save + dl_selected_model + "_" + dataset_name_save + "_sanity_" + mode + "_" + title + ".png"
        fig.tight_layout()
        plt.savefig(name)
        plt.close()
    else:
        fig.tight_layout()
        plt.show()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    # parser.add_argument("--Root_Dir", type=str, default='../')
    parser.add_argument("--Dataset_name", type=str, default='GunPointAgeSpan')
    parser.add_argument("--Dataset_name_save", type=str, default='GunPointAgeSpan_Cluster')
    parser.add_argument("--Experiments", nargs='+', default='experiment_0')
    parser.add_argument("--DLModel", type=str, default='MLP')
    parser.add_argument("--Evaluation_mode", type=str, default='Cascade')
    parser.add_argument("--Title", type=str, default='Cascade_correlation')
    parser.add_argument("--Use_tsr", action="store_true", default=False)
    parser.add_argument("--use_randommaps", action="store_true", default=True)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])
    print("Load Data and Model")
    cleantestsets, models, model_softmaxs, saliency_constructor_gcs, saliency_constructors = load_data_and_models(
        args=args
    )

    root_dir = parentDir + '/../'
    dataset_name = args.Dataset_name
    dataset_name_save = args.Dataset_name_save
    dl_selected_model = args.DLModel
    path_2_saliency = root_dir + "results/" + dataset_name_save + "/" + dl_selected_model + "/"
    experiments = args.Experiments
    use_randommaps = args.use_randommaps
    saliency_lists = load_saliencies(path_2_saliency, experiments, randombaseline=use_randommaps)

    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    if use_randommaps:
        methods = ["random_abs",
                  "random_noabs"]
    else:
        if dl_selected_model not in ['LSTM', 'MLP']:

            methods = ["grads",
                       "smoothgrads",
                       "igs",
                       "lrp_epsilon",
                       "gradCAM",
                       "guided_gradcam",
                       "guided_backprop",
                       "lime",
                       "kernel_shap"]
            absolute_methods = ["grads", "smoothgrads", "igs", "gradCAM"]
        else:
            methods = ["grads",
                       "smoothgrads",
                       "igs",
                       "lrp_epsilon",
                       "lime",
                       "kernel_shap"]
            absolute_methods = ["grads", "smoothgrads", "igs"]
        # methods = ["grads",
        #            "smoothgrads",
        #            "igs",
        #            "lrp_epsilon"]
    
    saliency_dict = {}
    for method in methods:
        if use_randommaps and method == "random_abs":
            saliency_dict["random"] = saliency_lists[2][0][method]
        elif not use_randommaps:
            if method in absolute_methods:
                saliency_dict[method] = saliency_lists[0][0][method]
            else:
                saliency_dict[method] = saliency_lists[1][0][method]
                
    if use_randommaps:
        methods = ["random"]
    saliency_list = [saliency_dict]

    random_correlation_stat, random_ssim_stat, rand_names, rand_acc_dict = saliency_sanitycheck(
        args=args,
        models=models,
        datasets=cleantestsets,
        normal_saliency=saliency_list[0],
#         normal_saliency_abs=saliency_abs_list[0],
        methods=methods,
        save_randsaliency=True
    )
    # plot_statistic(args=args,
    #                rand_correlation_stat = random_correlation_stat,
    #                rand_ssim_stat = random_ssim_stat,
    #                rand_names = rand_names,
    #                rand_acc_dict = rand_acc_dict,
    #                vis_methods = methods,
    #                title = args.Title
    # )