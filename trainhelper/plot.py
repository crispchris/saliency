"""
use to plot the training result and evaluation
"""
## ----------------
## --- build-in ---
## ----------------
import sys
sys.path.append('..')
from typing import Dict
## -------------------
## --- Third-party ---
## -------------------
import matplotlib.pyplot as plt
import numpy as np

## -----------
## --- Own ---
## -----------
from utils import create_directory

def plot_model_result(criterions: Dict, plot_path: str = None) -> object:
    if "train_loss" in criterions.keys():
        train_losses = criterions["train_loss"]
        test_losses = criterions["val_loss"]
        plot_loss = True
    else:
        plot_loss = False
    if "train_accuracy" in criterions.keys():
        train_acc = criterions["train_accuracy"]
        test_acc = criterions["val_accuracy"]
        plot_accuracy = True
    else:
        plot_accuracy = False

    if plot_path is None:
        plot_path = "../results/" + criterions["Dataset"] + "/" + criterions["Classifier"] + "/result_plots/"
    path_done = create_directory(plot_path)

    ## Plot the results
    if plot_loss:
        plt.figure(dpi=265)
        plt.plot(np.arange(len(train_losses)), train_losses, label='Train loss')
        plt.plot(np.arange(len(test_losses)), test_losses, label='Val loss')
        # plt.yscale('log')
        # plt.ylim(-10, 60)
        plt.legend()
        plt.title(f"[{criterions['Dataset']}] Cross Entropy Loss for {criterions['Classifier']}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.savefig(plot_path + 'lr_{}_loss.png'.format(criterions["learning_rate"]))
        print(f"[INFO] Loss Plot is saved under {plot_path}")
    if plot_accuracy:
        plt.figure(dpi=265)
        plt.plot(np.arange(len(train_acc)), train_acc, label='Train accuracy')
        plt.plot(np.arange(len(test_acc)), test_acc, label='Val accuracy')
        plt.title(f"[{criterions['Dataset']}] Accuracy for {criterions['Classifier']}")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        plt.savefig(plot_path + 'lr_{}_accuracy.png'.format(criterions["learning_rate"]))
        print(f"[INFO] Accuracy Plot is saved under {plot_path}")

    plt.clf()
    plt.imshow(criterions["confusion_matrix_testset"], interpolation='nearest', cmap=plt.cm.Blues)
    classNames = criterions["label_summary"]

    plt.title(f'Confusion Matrix of {criterions["Dataset"]} on Testset with {criterions["Classifier"]}')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames, rotation=45)
    plt.yticks(tick_marks, classNames)

    # thresh = criterions["confusion_matrix"].max() / 2.
    for i in range(len(classNames)):
        for j in range(len(classNames)):
            plt.text(j, i, str(criterions["confusion_matrix_testset"][i][j]),
                     horizontalalignment="center")
                     # color="white" if criterions["confusion_matrix"][i][j] > thresh else "black")
    plt.colorbar()

    plt.savefig(plot_path + '{}_lr_{}_confusion_matrix.png'.format(criterions["Classifier"],
                                                                   criterions["learning_rate"]))
    # plt.show()

