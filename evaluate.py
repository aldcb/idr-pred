#!/usr/bin/env python3
# Copyright (c) 2024 aldcb - GPLv3 (http://gnu.org/licenses/gpl.html)

import argparse
import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import torch
from torch.utils.data import DataLoader, random_split

from configs.config import config
from models.dataset import ProteinDataset, custom_collate, score_to_classes, subdivide_by_length
from models.model import ProteinRNN

def get_confusion_matrix(model, dataloader, thresholds):
    """Calculates the classification error matrix of a regression model given a thresholds list."""
    model.eval()
    device = next(model.parameters()).device
    y_true = torch.empty(0, device=device)
    y_pred = torch.empty(0, device=device)
    with torch.no_grad():
        for sequences, targets, masks in dataloader:
            predictions = model(sequences)
            y_true = torch.cat((y_true, targets.masked_select(masks)))
            y_pred = torch.cat((y_pred, predictions.masked_select(masks)))
    cm = confusion_matrix(score_to_classes(y_true, thresholds),
                          score_to_classes(y_pred, thresholds))
    return cm

def plot_confusion_matrices(cm_list, class_thresholds, subset_thresholds = None):
    """Plots the list of confusion matrices for the sequence length subsets, and a cumulative matrix """
    n_subsets = len(cm_list)
    gs = gridspec.GridSpec(2, n_subsets+2, height_ratios=[2.5, 1], hspace=0.55)
    fig = plt.figure()
    cumulative_cm = np.sum(cm_list, axis=0)
    ax_cumulative = fig.add_subplot(gs[0, :])
    ax_cumulative.imshow(cumulative_cm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0)
    ticklabels = _thresholds_to_labels(class_thresholds)
    ax_cumulative.set(xticks=range(cumulative_cm.shape[1]), yticks=range(cumulative_cm.shape[0]),
                      xticklabels=ticklabels, yticklabels=ticklabels,
                      title='Confusion Matrix', ylabel='Ground truth', xlabel='Predicted')
    total = sum(sum(row) for row in cumulative_cm)
    for i in range(cumulative_cm.shape[0]):
        for j in range(cumulative_cm.shape[1]):
            percentage = (cumulative_cm[i, j] / total) * 100
            ax_cumulative.text(j, i, format(percentage, '.2f') + '%',
                    ha="center", va="center",
                    color="white" if percentage > 50 else "black")
    if n_subsets > 1:
        ax_subset = [fig.add_subplot(gs[1, n+1]) for n in range(n_subsets)]
        titles = _thresholds_to_labels(subset_thresholds)
        for n in range(n_subsets):
            ax_subset[n].imshow(cm_list[n], interpolation='nearest', cmap=plt.cm.Reds, vmin=0)
            ax_subset[n].set_title(f"{titles[n]}", fontsize=9)
            ax_subset[n].set_xticks([])
            ax_subset[n].set_yticks([])
            ax_subset[n].set_xticklabels([])
            ax_subset[n].set_yticklabels([])
    plt.savefig(f"models/{config.model_name}/conf_matrix.jpg", bbox_inches='tight')
    plt.show()

def _thresholds_to_labels(thresholds):
    if isinstance(thresholds[0], int):
        labels = [f"≤{thresholds[0]} aa"]
        for i in range(len(thresholds)-1):
                labels.append(f"{thresholds[i]+1}-{thresholds[i+1]} aa")
        labels.append(f">{thresholds[-1]} aa")
        return labels
    else:
        labels = [f"<{thresholds[0]}"]
        for i in range(len(thresholds)-1):
                labels.append(f"{thresholds[i]}-{thresholds[i+1]}")
        labels.append(f">{thresholds[-1]}")
        return labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
    args = parser.parse_args()
    config.update_from_yaml(file_path = args.config)
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ProteinDataset(config.data_file, device)
    _, _, test_dataset = random_split(dataset, [config.train_ratio, config.val_ratio, config.test_ratio])
    subset_list = subdivide_by_length(test_dataset, config.subset_thresholds)
    dataloader_list = [DataLoader(subset, collate_fn=custom_collate, batch_size=1, shuffle=False) for subset in subset_list]
    model = ProteinRNN(**config.model).to(device)
    checkpoint = f"{os.path.dirname(__file__)}/models/{config.model_name}/model.pth"
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"The trained model {config.model_name} was not found.")
    model.load_state_dict(torch.load(checkpoint, map_location = device))
    cm_list = [get_confusion_matrix(model, dataloader, config.class_thresholds) for dataloader in dataloader_list]
    plot_confusion_matrices(cm_list, config.class_thresholds, config.subset_thresholds)

if __name__ == "__main__":
    main()
