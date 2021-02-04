# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 00:51:55 2020

@author: Jason
"""


# Model metric plotter

# Plots graphs of the training process using a keras
# model. On graph is the chosen metric and the other 
# is the loss. Or plots the confusion matrix of the
# results of evaluating the trained model

import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
import pandas as pd


def plot_metrics(history, save_path, metrics):
    grid_size = len(metrics)/2
    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        plt.subplot(2, grid_size, n + 1)
        plt.plot(history.epoch, history.history[metric], color="k", label="Train")
        plt.plot(history.epoch, history.history["val_" + metric],
                 color="k", linestyle="--", label="Val")
        plt.xlabel("Epoch")
        plt.ylabel(name)
        if metric == "loss":
            plt.ylim([0, plt.ylim()[1]])
        elif metric == "auc":
            plt.ylim([0.8,1])
        else:
            plt.ylim([0,1])

        plt.legend()
        plt.grid()
    
    plt.savefig(save_path)
    plt.show()
    
    
    
def confusion_matrix_plot(y_true, y_pred, target_names, title="Confusion matrix",
                          cmap=None, normalize=True, save=False,
                          save_path=None):
    """
    from: https://www.kaggle.com/grfiv4/plot-a-confusion-matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy
    
    if cmap is None:
        cmap = plt.get_cmap("Blues")

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label\naccuracy={:0.4f}; misclass={:0.4f}".format(accuracy, misclass))
    
    if save:
        plt.savefig(save_path)
    
    plt.show()
    
    
    
def save_history_to_csv(history, save_path, metrics):
    training_metric_log = {}
    
    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        training_metric_log["Training " + name] = history.history[metric]
        training_metric_log["Validation " + name] = history.history["val_" + metric]
    
    df = pd.DataFrame(training_metric_log,
                      index=history.epoch)
    # Display first 5 rows
    df.head()
    
    df.to_csv(save_path)