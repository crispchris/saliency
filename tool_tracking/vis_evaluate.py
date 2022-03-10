"""
Visualization
"""


## ------------------
## --- Third-Party ---
## ------------------
import os
import matplotlib.pyplot as plt
## -----------
## --- own ---
## -----------
from tool_tracking.dataknowing.loadData import load_data, pipe_segment, filteroutlabels
from tool_tracking.dataknowing.loadData import summarize_labels, balance_classes, split_data
from models.models import TCN_4base, FCN_4baseline, TCN_3base, FCN_3baseline
from models.tcn_layer import TCN_layer
from tool_tracking.tool_utils import Traindata
from visualize_mechanism.cam import interpolate_smooth, grad_cam
from visualize_mechanism.plot_vis_plt import plot_vis_plt

source = "../data/tool-tracking/tool-tracking-data"
tool = "electric_screwdriver"
sensors = ['acc', 'gyr', 'mag'] # 'mic']
garbage_labels = [-1, 6, 7, 8, 14]
summary_labels = {0:[2], 1:[3], 2:[4], 3:[5]}
## set window length and overlap
window_length = 0.2 # unit in s
overlap = 0.5 # unit in percent
## train/test size
test_size = 0.25

## whole dataset from tool in data dict separately
data_dict = load_data(source, tool=tool) ## each measurement is a DataBunch

dataset = pipe_segment(data_dict, window_length=window_length, overlap=overlap, enforce_size=True,
                       sensors=sensors)
## --- Preprocess ---
## filter out garbage labels
dataset_f = filteroutlabels(labels=garbage_labels, data=dataset)
## summarize labels according to the summary labels dictionary
dataset_summarized = summarize_labels(dataset_f, summary_labels, window_length)
# apply class balancing (every labels to have the same size)
data_balanced = balance_classes(dataset_summarized, sensors)
# split dataset into train and test set
traindata, testdata = split_data(data_balanced, test_size=test_size, sensors=sensors)

## --- Train ---
## transfer train and test set into Torch dataset
trainset = Traindata(traindata)
testset = Traindata(testdata)

selected_model = "TCN_CAM_nosigmoid" ## ["BaselineFCN", "TCN"]
batch_size = 32
dropout = 0.2
num_classes = 4

# model setting
if selected_model is "TCN_CAM_nosigmoid":
    tcn_mag = TCN_layer(ch_in=3, dilation=[1, 2, 4])
    tcn_gyr = TCN_layer(ch_in=3, dilation=[1, 2, 4])
    tcn_acc = TCN_layer(ch_in=3, dilation=[1, 2, 4])
    if trainset.has_audio():
        tcn_aud = TCN_layer(ch_in=1, dilation=[1, 2, 4])
        # concat inputs
        model = TCN_4base(tcn_acc, tcn_gyr, tcn_mag, tcn_aud, dropout_rate=dropout,
                        num_classes=num_classes)
    else:
        model = TCN_3base(tcn_acc, tcn_gyr, tcn_mag, dropout_rate=dropout,
                        num_classes=num_classes)

if selected_model is "BaselineFCN":
    # Fully Convolutional inputs
    if trainset.has_audio():
        model = FCN_4baseline(in_mag=3, in_acc=3, in_gyr=3, in_aud=1, num_classes=num_classes)
    else:
        model = FCN_3baseline(in_mag=3, in_acc=3, in_gyr=3, num_classes=num_classes)

model_ckp = os.getcwd() + "/checkpoints/" + selected_model + "/checkpoint_216.ckp"

## visualize Methods
map_method = "Grad-Cam"
# class_map, prediction, target = apply_torch_cam(testset, model=model, checkpoint=model_ckp)
class_map, prediction, target = grad_cam(testset, model=model, checkpoint=model_ckp, target_layer="gap_softmax.conv1")
# class_map = cam(testset, model=model, checkpoint=model_ckp)
rand_num = 20
win_l, inter_CAM, acc, gyr, mag = interpolate_smooth(data=testset, cam_out=class_map,
                                                     num_idx=rand_num)

# plot_vis_plt(inter_CAM, win_l, acc.transpose(0, 1), target, f'acc_target_{target}')
# plot_vis_plt(inter_CAM, win_l, gyr.transpose(0, 1), target, f'gyr_target_{target}')
# plot_vis_plt(inter_CAM, win_l, mag.transpose(0, 1), target, f'mag_target_{target}')
fig_path = os.getcwd() + "/figures"
if not os.path.isdir(fig_path + "/visualize"):
    os.mkdir(fig_path + "/visualize")
if not os.path.isdir(fig_path + "/visualize/" + map_method):
    os.mkdir(fig_path + "/visualize/" + map_method)
fig_path = fig_path + "/visualize/" + map_method
plot_vis_plt(inter_CAM, win_l, acc.transpose(0, 1), prediction, f'acc_predict_{prediction}',
             save_path=fig_path, method_name=map_method)
plot_vis_plt(inter_CAM, win_l, gyr.transpose(0, 1), prediction, f'gyr_predict_{prediction}',
             save_path=fig_path, method_name=map_method)
plot_vis_plt(inter_CAM, win_l, mag.transpose(0, 1), prediction, f'mag_predict_{prediction}',
             save_path=fig_path, method_name=map_method)
plt.show()