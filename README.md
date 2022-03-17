# Don't get me wrong
## How to apply Deep Visual Interpretations to Time Series
You can find the preprint here: https://arxiv.org/abs/2203.07861
```
@misc{loeffler2022dont,
title={Don't Get Me Wrong: How to apply Deep Visual Interpretations to Time Series},
author={Christoffer Loeffler and Wei-Cheng Lai and Bjoern Eskofier and Dario Zanca and Lukas Schmidt and Christopher Mutschler},
year={2022},
eprint={2203.07861},
archivePrefix={arXiv},
primaryClass={cs.CV}
}
```
## Train Model
1. get training data: [UCR](https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/), [NATOPS](http://groups.csail.mit.edu/mug/natops/) 
```
# unpack UCR to ./data/Univariate_ts
# unpack NATOPS to ./data/Multivariate_ts/NATOPS
```
2. train a model (exemplary for FordA and l2 regularization)
```
# go to run_evaluation/conf to see different available models
python ./run_evaluation/train.py models=fcn_l2_0.03 data=forda
```
## Generate Saliency
Our visual interpretation methods are based on [Captum](https://captum.ai/).
1. compute saliency )
```
python ./run_evaluation/evaluation/visual_interpretability.py --Dataset_name FordA --Dataset_name_save FordA --Experiments experiment_0 --DLModel FCN_withoutFC --Regularization_norm l2_regularization --Regularization_parameter 0.03
```
2. normalize saliency
```
python ./run_evaluation/evaluation/visual_interpret_normalization.py --Dataset_name FordA --Experiments experiment_0 --DLModel FCN_withoutFC --Regularization_norm l2_regularization --Regularization_parameter 0.03
```
## Evaluate saliency
1. Run Faithfulness (Temporal Importance)
```
python ./run_evaluation/evaluation/temporal_sequence_importance.py --Dataset_name FordA --Dataset_name_save FordA --Experiments experiment_0 --DLModel FCN_withoutFC --Evaluation_mode "mean" --Evaluation_length 0.2 --Save_to ~/results/FordA/FCN_withoutFC/l2_regularization/loss_0.03/experiment_0/temporal_importance --Regularization_norm l2_regularization --Regularization_parameter 0.03
```
2. Run Faithfulness (Temporal Sequence) and any other metric analogously.
## Plot Results
1. Set paths correctly and plot results
```
# set paths in ./plotting/results_reader.py correctly
python ./plotting/plot.py
```
2. More plotting in separate python files available:
```
# see plot_regularization_trends_summary.py for details
```
# Tool Tracking
1. Follow instructions for data (loader) ([Tool Tracking](https://github.com/mutschcr/tool-tracking))
```
unpack tool tracking data to ./data/tool-tracking/tool-tracking-data
```
3. Run separate training script
```
python ./tool_tracking/train.py
```
4. Evaluate
```
python ./tool_tracking/evaluation/iosr_precision_recall_ts_eval.py --Dataset_name tool_tracking --Dataset_name_save tool_tracking --Experiments experiment_0 --DLModel TCN_dense --Save_scores_path pointing_game_scores.npy
```

### other 

License: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License

The code is largely adapted from external sources. Then the external licences apply
* [Tool Tracking](https://github.com/mutschcr/tool-tracking)
* [Captum](https://captum.ai/)
* and others (see each source file for specifics)
