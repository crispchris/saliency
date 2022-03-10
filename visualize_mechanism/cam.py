"""
CAM (Class Activation Mapping) based Mechanism
TODO: General the Methods, first use for tool-tracking
origin paper: https://openaccess.thecvf.com/content_cvpr_2016/papers/Zhou_Learning_Deep_Features_CVPR_2016_paper.pdf
also refer to: Torchcam https://frgfm.github.io/torch-cam/installing.html
"""

## -------------------
## --- Third-party ---
## -------------------
import numpy as np
import torch as t
import torch.nn.functional as F
# import cv2
from typing import Optional, List
from scipy.interpolate import interp1d

# from torchcam.cams import CAM, GradCAM
# from torchray.attribution.grad_cam import grad_cam
# from torchray.benchmark import plot_example, get_example_data

## -----------
## --- Own ---
## -----------
# from visualize_mechanism.utils import Grad_extractor
from visualize_mechanism.grad_cam import Grad_Cam as GRADCAM

## Device Configuration
device = t.device('cuda' if t.cuda.is_available() else 'cpu')
class _CAM:
    """
    Class Activation Map for Time Series Data

    Args:
        model: torch input model
        conv_layer: name of the last convolutional layer
    """

    def __init__(self, model: t.nn.Module, conv_layer: str) -> None:
        self.model = model
        self.sub_module_dict = dict(model.named_modules())

        if conv_layer not in self.sub_module_dict.keys():
            raise ValueError(f"Unable to fine submodule {conv_layer} in the model")

        self.relu_used = False
        self.score_used = False
        self.scores = None
        ## Hook for feature map
        self.hook_use = True
        self.feature_map = None

        self.sub_module_dict[conv_layer].register_forward_hook(hook=self._hook)

    def _hook(self, module: t.nn.Module, input: t.Tensor, output: t.Tensor) -> None:
        print("feature map shape: ", output.shape)
        self.feature_map = output.data

    def _get_weights(self, class_idx: int, input: t.Tensor = None, scores: Optional[t.Tensor] = None):
        return NotImplementedError
    def __call__(self, class_idx: int, input: t.Tensor = None, scores: Optional[t.Tensor] = None, use_normalized: bool = True):
        """input: use for compute the forwardprop of the convolutional network
            which later to use for the gradient computation
        """
        ## check if only a 1-sized batch:
        if self.feature_map.shape[0] != 1:
            raise ValueError(f"expected a 1-sized batch data, but received: {self.feature_map.shape[0]}")

        if class_idx < 0:
            raise ValueError("Incorrect `class_idx` argument value")
        return self.compute_cams(class_idx, input, scores, use_normalized)

    def compute_cams(self, class_idx: int, input: t.Tensor = None, scores: Optional[t.Tensor] = None, used_normalized: bool = True):
        """
        Compute the CAM for a specific output of class
        Parameters
        ----------
        class_idx: output class index of the target class who CAM will be computed
        scores: forward output scores of the hooked model
        used_normalized: CAM should be normalized or not

        Returns
        -------
        t.Tensor: class activation map for time series data
        """
        weights = self._get_weights(class_idx, input, scores) ##weights = [Dim]
        feature_map = self.feature_map[0]       ## feature_map = [Dim, Len]
        weights_ = weights.reshape(-1, 1)
        ### CAM only one class
        cam_ = None
        cam_mul = t.mul(feature_map, weights_).sum(dim=0, keepdim=True)
        # CAM = weights.view(*weights.shape, 1, 1) * feature_map
        # for i in range(CAM.shape[0]):
        #     if cam_ is None:
        #         cam_ = CAM[i, i, :]
        #     else:
        #         cam_ += CAM[i, i, :]
        cam_ = cam_mul
        cam_ = cam_.reshape(1, -1).cpu().detach().numpy()
        # CAM = np.dot(get_last_conv.transpose(0, 1).cpu().detach().numpy(), softmax_weight_data.cpu().detach().numpy())
        if self.relu_used:
            cam_ = F.relu(t.tensor(cam_), inplace=True)
            cam_ = cam_.detach().numpy()
        if used_normalized:
            cam_ -= np.min(cam_, axis=1)
            cam_ /= np.max(cam_, axis=1)
            cam_ *= 100
        return cam_

class CAM(_CAM):
    """
        L_c(Saliency Map for Class C) = Sum[ weight_k(kernel) for Class C * A(feature map) of kernel k]

        Class Activation Mapping

        Parameters
        ----------
        model: the trained model
        conv_layer: name of the last convolutional layer
        fc_layer: name of the fully connected layer

    """
    def __init__(self, model: t.nn.Module, conv_layer: str, fc_layer: str,
                 relu_used: bool = False):
        super().__init__(model, conv_layer)
        self.fc_weights = self.sub_module_dict[fc_layer].weight.data
        self.relu_used = relu_used

    def _get_weights(self, class_idx: int, input: t.Tensor = None, scores: Optional[t.Tensor] = None):
        return self.fc_weights[class_idx, :]

# class Grad_Cam(_CAM):
#     """
#         Grad-Cam method
#         L_c(Saliency Map for Class C) = Sum[ weight_k(kernel) for Class C * A(feature map) of kernel k]
#         weight_k(kernel) for Class C = [1/Z] * sum[sum[derivative[ Y(Class Score)] / derivative[ A(feature map) of kernel k]
#         weight_k for Class C is certainly the gradient of the score for class C(before the softmax), with respect to
#         the feature map of the last convolutional layer. These Gradients flowing back are GAP(global average pooled)
#         Returns
#         -------
#         Final Class discriminative Saliency Map
#         """
#     def __init__(self, model: t.nn.Module, conv_layer: str, relu_used: bool = True):
#         super().__init__(model, conv_layer)
#         self.model = model
#         self.gradients_conv = None
#         self.conv_layer = conv_layer
#         self.relu_used = relu_used
#
#         ## gradient of conv
#     #     self.sub_module_dict[conv_layer].register_backward_hook(self._get_gradient)
#     # def _get_gradient(self, module: t.nn.Module, grad_input: t.Tensor, grad_output: t.Tensor):
#     #     print("Grad output shape :", grad_output.shape)
#     #     self.gradients_conv = grad_output.data
#
#     def _compute_gradient(self, score: Optional[t.Tensor], class_idx: int): ## size of score should be like [N, C, L] = [N, C, 1]
#         score = score.squeeze(0)
#         one_hot = np.zeros((1, score.shape[0]), dtype=np.float32)
#         one_hot[0][class_idx] = 1
#         one_hot = t.from_numpy(one_hot).requires_grad_(True).to(device)
#         one_hot = t.sum(one_hot * score.reshape(1, -1))
#         print("one hot shape: ", one_hot.shape)
#         self.model.zero_grad()
#         self.one_hot = one_hot
#
#
#     def _get_weights(self, class_idx: int, input: t.Tensor = None, scores: Optional[t.Tensor] = None):
#         """
#         Parameters
#         ----------
#         input: use to compute the forward output of convolutional layer
#         scores: Y_c
#
#         Returns
#         -------
#         weights for Grad-CAM
#         """
#         self._compute_gradient(score=scores, class_idx=class_idx)
#         grad_extractor = Grad_extractor(model=self.model, output_score=self.one_hot, target_layer=self.conv_layer)
#         self.gradients_conv = grad_extractor(input=input)
#         weights = self.gradients_conv.mean(dim=1)
#         return weights

# def apply_torch_cam(testdata, model=None, checkpoint=None, class_idx: int=0, use_cuda: bool = True):
#     """
#     From Torchcam
#     Parameters
#     ----------
#     testdata
#     model
#     checkpoint
#     class_idx
#     use_cuda
#
#     Returns
#     -------
#     CAM: [C, L] --> Classes, Length of feature map
#     """
#     rand_num = 20
#     ## data preparation
#     xt_acc = t.tensor(testdata.Xt_acc[rand_num]).float().to(device)  ## data shape = [B, Features, window size]
#     xt_gyr = t.tensor(testdata.Xt_gyr[rand_num]).float().to(device)
#     xt_mag = t.tensor(testdata.Xt_mag[rand_num]).float().to(device)
#     xt_acc = t.reshape(xt_acc, (1, xt_acc.shape[0], xt_acc.shape[1])).transpose(1, 2)
#     xt_gyr = t.reshape(xt_gyr, (1, xt_gyr.shape[0], xt_gyr.shape[1])).transpose(1, 2)
#     xt_mag = t.reshape(xt_mag, (1, xt_mag.shape[0], xt_mag.shape[1])).transpose(1, 2)
#     if testdata.has_audio():
#         xt_aud = t.tensor(testdata.Xt_aud[rand_num]).float().to(device)
#         xt_aud = t.reshape(xt_aud, (1, xt_aud.shape[0], xt_aud.shape[1])).transpose(1, 2)
#     target = int(testdata.y[rand_num])
#
#     model_ckp = t.load(checkpoint, 'cuda' if use_cuda else None)
#     model.load_state_dict(model_ckp['state_dict'])
#     model.to(device)
#     model.eval()
#
#     def fc_hook(module: t.nn.Module, input: t.Tensor, output: t.Tensor):
#         global fc_output
#         fc_output = output
#     def conv_hook(module: t.nn.Module, input: t.Tensor, output: t.Tensor):
#         global conv_output
#         conv_output = output
#     cam_torch = Grad_Cam(model=model, conv_layer="gap_softmax.conv1")
#     model.gap_softmax.fc.register_forward_hook(fc_hook)
#     model.convblock2.relu.register_forward_hook(conv_hook)
#
#     with t.no_grad():
#         # prediction
#         if testdata.has_audio():
#             prediction = model(xt_acc, xt_gyr, xt_mag, xt_aud)
#         else:
#             prediction = model(xt_acc, xt_gyr, xt_mag)
#     prediction = t.argmax(prediction, dim=1).cpu()[0][0]
#     convout = conv_output
#     cam_ = cam_torch(class_idx=prediction, input=conv_output, scores=fc_output.transpose(1, 2))
#     ## have done the normalization min-max between [0, 100]
#     return cam_, prediction, target

def grad_cam(testdata, model=None, checkpoint=None,  class_idx: int=0, target_layer: str = "gap_softmax.conv1", use_cuda: bool = True):
    """for tool tracking (separated input)"""
    rand_num = 20
    ## data preparation
    xt_acc = t.tensor(testdata.Xt_acc[rand_num]).float().to(device)  ## data shape = [B, Features, window size]
    xt_gyr = t.tensor(testdata.Xt_gyr[rand_num]).float().to(device)
    xt_mag = t.tensor(testdata.Xt_mag[rand_num]).float().to(device)
    xt_acc = t.reshape(xt_acc, (1, xt_acc.shape[0], xt_acc.shape[1])).transpose(1, 2)
    xt_gyr = t.reshape(xt_gyr, (1, xt_gyr.shape[0], xt_gyr.shape[1])).transpose(1, 2)
    xt_mag = t.reshape(xt_mag, (1, xt_mag.shape[0], xt_mag.shape[1])).transpose(1, 2)
    if testdata.has_audio():
        xt_aud = t.tensor(testdata.Xt_aud[rand_num]).float().to(device)
        xt_aud = t.reshape(xt_aud, (1, xt_aud.shape[0], xt_aud.shape[1])).transpose(1, 2)
    target = int(testdata.y[rand_num])

    model_ckp = t.load(checkpoint, 'cuda' if use_cuda else None)
    model.load_state_dict(model_ckp['state_dict'])
    model.to(device)
    model.eval()
    sub_module_dict = dict(model.named_modules())
    fmaps = {}
    def conv_hook(module: t.nn.Module, input: t.Tensor, output: t.Tensor):
        fmaps["input"] = output.data
    sub_module_dict["convblock2.relu"].register_forward_hook(conv_hook)  ## as input data to GradCAM

    with t.no_grad():
        # prediction
        if testdata.has_audio():
            prediction = model(xt_acc, xt_gyr, xt_mag, xt_aud)
        else:
            prediction = model(xt_acc, xt_gyr, xt_mag)
    prediction = t.argmax(prediction, dim=1).cpu()[0][0]


    list_target_layers = target_layer.split('.')
    len_target_layers = len(list_target_layers)
    for name, module in model.named_modules():
        if name in list_target_layers:
            target_module = module
            target_layer = list_target_layers[1]
    gradcam = GRADCAM(model=target_module)
    output, ids = gradcam.forward(fmaps["input"])
    gradcam.backward(class_idx=ids[1][0][0])

    print(f"Generating Grad-CAM for {target_layer}")
    gcam = gradcam.generate(target_layer)[0]

    return gcam, prediction, target

class CAM_UCR:
    def __init__(self, testdata, model=None, checkpoint=None, samples_idx: int = None, class_idx: bool = True,
                 target_layer: str = "gap_softmax.conv1",
                 fc_layer: str = "gap_softmax.fc",
                 use_cuda: bool = True,
                 used_relu: bool = False,
                 model_loaded: bool = True):
        self.testdata = testdata
        self.model = model
        self.checkpoint = checkpoint
        self.samples_idx = samples_idx
        self.class_idx = class_idx
        self.target_layer = target_layer
        self.fc_layer = fc_layer
        self.use_cuda = use_cuda
        self.used_relu = used_relu
        self.model_loaded = model_loaded

    def generate(self, idx):
        return cam_ucr(self.testdata, self.model, self.checkpoint, samples_idx=idx, class_idx=self.class_idx,
                       target_layer=self.target_layer, fc_layer=self.fc_layer, model_loaded=self.model_loaded)

def cam_ucr(testdata, model=None, checkpoint=None, samples_idx: int = None, class_idx: bool = None,
            target_layer: str = "gap_softmax.conv1",
            fc_layer: str = "gap_softmax.fc",
            use_cuda: bool = True,
            used_relu: bool = False,
            model_loaded: bool = True):
    """
    CAM Visualization Methods for UCR Datasets

    Parameters
    ----------
    testdata: testset from dataset
    model: the model to view for visualization on data
    checkpoint: the stored checkpoint on dataset
    samples_idx: the index of sample (should be the same length as class_idx
    class_idx: the index of class (target label)
    target_layer: the layer, where the gradient should be computed, compared to the output
    fc_layer: the last layer before output(sigmoid) usually fully connected layer
    use_cuda: GPU or not
    used_relu: Boolean (for GradCam result relu or not)
    model_loaded: Boolean (for model already be loaded)

    Returns
    -------
    Grad-CAM attention map
    """

    if checkpoint is not None and model_loaded is False:
        ## Model Load and Setting
        model_ckp = t.load(checkpoint, 'cuda' if use_cuda else None)
        model.load_state_dict(model_ckp['state_dict'])
        model.to(device)
        model.eval()
        model_loaded = True
    elif checkpoint is None and model_loaded is False:
        print("Use untrained Model")
        model.to(device)
        model.eval()

    ## Data preparation
    sample = t.tensor(testdata.data[samples_idx, :, :]).float().to(device) ## data_shape = [B, Dim, Length_onesample]
    sample = t.reshape(sample, (1, sample.shape[0], sample.shape[1]))
    target = testdata.labels[samples_idx]

    ## Use Class CAM
    cam_view = CAM(model=model, conv_layer=target_layer, fc_layer=fc_layer, relu_used=used_relu)

    with t.no_grad():
        # prediction
        prediction = model(sample)
    predicted_label = t.argmax(prediction, dim=1).cpu().reshape(-1, 1)
    predicted_label = predicted_label[0][0]

    if class_idx:
        cam_ = cam_view.compute_cams(class_idx=target)
    else:
        cam_ = cam_view.compute_cams(class_idx=predicted_label)
    return cam_, predicted_label, target

class Grad_Cam_UCR:
    def __init__(self, testdata, model=None, checkpoint=None, samples_idx: int = None, class_idx: bool = False,
                 target_layer: str = "gap_softmax.conv1",
                 before_block: str = "convblock4.relu",
                 use_cuda: bool = True,
                 used_relu: bool = False,
                 model_loaded: bool = True):
        self.testdata = testdata
        self.model = model
        self.checkpoint = checkpoint
        self.samples_idx = samples_idx
        self.class_idx = class_idx
        self.target_layer = target_layer
        self.befor_block = before_block
        self.use_cuda = use_cuda
        self.used_relu = used_relu
        self.model_loaded = model_loaded

    def generate(self, idx):
        return grad_cam_ucr(self.testdata, self.model, self.checkpoint, idx, self.class_idx,
                            self.target_layer,
                            self.befor_block,
                            self.use_cuda,
                            self.used_relu,
                            self.model_loaded)

def grad_cam_ucr(testdata, model=None, checkpoint=None,  samples_idx: int = None, class_idx: bool = None,
                 target_layer: str = "gap_softmax.conv1",
                 before_block: str = "convblock4.relu",
                 use_cuda: bool = True,
                 used_relu: bool = False,
                 model_loaded: bool = True):
    """
    Grad-CAM Visualization Methods for UCR Datasets

    Parameters
    ----------
    testdata: testset from dataset
    model: the model to view for visualization on data
    checkpoint: the stored checkpoint on dataset
    samples_idx: the index of sample
    class_idx:  Use the (target label) as reference or not (default : True)
    target_layer: the layer, where the gradient should be computed, compared to the output
    before_block: the last layer before the last block, target_layer should be in Last block
    use_cuda: GPU or not
    used_relu: Boolean (for GradCam result relu or not)
    model_loaded: Boolean (for model already be loaded)

    Returns
    -------
    Grad-CAM attention map
    """
    if checkpoint is not None and model_loaded is False:
        ## Model Load and Setting
        model_ckp = t.load(checkpoint, map_location= t.device('cuda') if use_cuda else t.device('cpu'))
        model.load_state_dict(model_ckp['state_dict'])
        model.to(device)
        model.eval()
        model_loaded = True
    elif checkpoint is None and model_loaded is False:
        print("Use untrained Model")
        model.to(device)
        model.eval()

    ## Data preparation
    sample = t.tensor(testdata.data[samples_idx, :, :]).float().to(device) ## data_shape = [B, Dim, Length_onesample]
    sample = t.reshape(sample, (1, sample.shape[0], sample.shape[1]))
    target = testdata.labels[samples_idx]

    ## Get Name of Layers in Model
    sub_module_dict = dict(model.named_modules())
    fmaps = {}

    def conv_hook(module: t.nn.Module, input: t.Tensor, output: t.Tensor):
        fmaps["input"] = output.data
    ## as input data to GradCAM ## So the Layer should be the one before the LAST BLOCK (GAP_SOFTMAX_block)
    sub_module_dict[before_block].register_forward_hook(conv_hook)

    with t.no_grad():
        # prediction
        prediction = model(sample)
    predicted_label = t.argmax(prediction, dim=1).cpu().reshape(-1, 1)
    predicted_label = predicted_label[0][0]

    list_target_layers = target_layer.split('.')
    len_target_layers = len(list_target_layers)
    target_module = None
    for name, module in model.named_modules():
        if name in list_target_layers:
            target_module = module
            target_layer = list_target_layers[1]
    if target_module is None:
        raise ValueError("target Module not Found")
    gradcam = GRADCAM(model=target_module, relu_used=used_relu)      ## The module maybe not be the original model
    output, ids = gradcam.forward(fmaps["input"])
    if class_idx:
        gradcam.backward(class_idx=target)
    else: ## use prediction
        gradcam.backward(class_idx=ids.indices[0][0][0])

    print(f"Generating Grad-CAM for {target_layer}")
    gcam = gradcam.generate(target_layer)[0]

    return gcam, predicted_label, target, target_module, fmaps["input"] ## target_module, fmaps are temporally solution


def grad_cam_pp():
    pass

def interpolate_smooth_ucr(dataset, cam_out, num_idx, num_sampling: int = 10):
    """
        Smoothing the Graph and CAM_OUT
        cam_out: [C, L]
        num_sampling: the data sample point * num_sampling = max_length
        For plotting
    """
    ## Data preparation
    sample = t.tensor(dataset.data[num_idx, :, :]).float().to(device)  ## data_shape = [Dim, Length_onesample]
    target = int(dataset.labels[num_idx])

    ## window length
    # max_length = sample.shape[1]
    max_length = sample.shape[1] * num_sampling
    win_len = np.linspace(0, sample.shape[1] - 1, num=max_length, endpoint=True)

    # linear interpolation to smooth
    sample_out = None
    for i in range(sample.shape[0]):
        inter_f = interp1d(range(sample.shape[1]), sample[i, :].cpu() if t.is_tensor(sample) else sample[i, :])
        inter_sample = inter_f(win_len).reshape(1, -1)
        if sample_out is None:
            sample_out = inter_sample
        else:
            sample_out = np.concatenate((sample_out, inter_sample), axis=0)

    ## linear interpolation to smooth
    inter_CAM = None
    for i in range(cam_out.shape[0]):
        inter_f = interp1d(range(cam_out.shape[1]), cam_out[i, :].cpu() if t.is_tensor(cam_out) else cam_out[i, :],
                           fill_value="extrapolate")
        heatmap = inter_f(win_len).astype(int).reshape(1, -1)
        if inter_CAM is None:
            inter_CAM = heatmap
        else:
            inter_CAM = np.concatenate((inter_CAM, heatmap), axis=0)


    return win_len, inter_CAM, sample_out


def interpolate_smooth(data, cam_out, num_idx):
    """cam_out: [C, L]
        For plotting
    """
    ## data preparation
    xt_acc = t.tensor(data.Xt_acc[num_idx]).float().to(device)  ## data shape = [B, Features, window size]
    xt_gyr = t.tensor(data.Xt_gyr[num_idx]).float().to(device)
    xt_mag = t.tensor(data.Xt_mag[num_idx]).float().to(device)
    xt_acc = t.reshape(xt_acc, (1, xt_acc.shape[0], xt_acc.shape[1])).transpose(1, 2)
    xt_gyr = t.reshape(xt_gyr, (1, xt_gyr.shape[0], xt_gyr.shape[1])).transpose(1, 2)
    xt_mag = t.reshape(xt_mag, (1, xt_mag.shape[0], xt_mag.shape[1])).transpose(1, 2)
    if data.has_audio():
        xt_aud = t.tensor(data.Xt_aud[num_idx]).float().to(device)
        xt_aud = t.reshape(xt_aud, (1, xt_aud.shape[0], xt_aud.shape[1])).transpose(1, 2)
    target = int(data.y[num_idx])
    ## window length
    win_len = np.linspace(0, cam_out.shape[1] - 1, num=200, endpoint=True)
    ## linear interpolation to smooth
    inter_CAM = None
    for i in range(cam_out.shape[0]):
        inter_f = interp1d(range(cam_out.shape[1]), cam_out[i, :].cpu() if t.is_tensor(cam_out) else cam_out[i, :])
        heatmap = inter_f(win_len).astype(int).reshape(1, -1)
        if inter_CAM is None:
            inter_CAM = heatmap
        else:
            inter_CAM = np.concatenate((inter_CAM, heatmap), axis=0)
    ## padding
    xt_acc_pad = t.zeros((xt_acc.shape[0], xt_acc.shape[1], cam_out.shape[1]))
    xt_acc_pad[:, :, :xt_acc.shape[2]] = xt_acc
    xt_gyr_pad = t.zeros((xt_gyr.shape[0], xt_gyr.shape[1], cam_out.shape[1]))
    xt_gyr_pad[:, :, :xt_gyr.shape[2]] = xt_gyr

    # linear interpolation to smooth
    acc = None
    gyr = None
    mag = None
    for i in range(xt_acc.shape[1]):
        inter_f = interp1d(range(cam_out.shape[1]), xt_acc_pad[0, i, :].cpu())
        inter_acc = inter_f(win_len).reshape(1, -1)
        inter_f = interp1d(range(cam_out.shape[1]), xt_gyr_pad[0, i, :].cpu())
        inter_gyr = inter_f(win_len).reshape(1, -1)
        inter_f = interp1d(range(cam_out.shape[1]), xt_mag[0, i, :].cpu())
        inter_mag = inter_f(win_len).reshape(1, -1)
        if acc is None:
            acc = inter_acc
        else:
            acc = np.concatenate((acc, inter_acc), axis=0)
        if gyr is None:
            gyr = inter_gyr
        else:
            gyr = np.concatenate((gyr, inter_gyr), axis=0)
        if mag is None:
            mag = inter_mag
        else:
            mag = np.concatenate((mag, inter_mag), axis=0)
    return win_len, inter_CAM, acc, gyr, mag





