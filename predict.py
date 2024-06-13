#!/usr/bin/env python3
# Copyright (c) 2024 aldcb - GPLv3 (http://gnu.org/licenses/gpl.html)

import argparse
import os

import matplotlib.pyplot as plt
import torch

from configs.config import config
from models.dataset import protein_to_indexes
from models.model import ProteinRNN

def predict(protein_seq, model):
    """Predicted score list from a protein sequence string."""
    model.eval()
    device = next(model.parameters()).device
    sequence = protein_to_indexes(protein_seq, device)
    with torch.no_grad():
        prediction = model(sequence.unsqueeze(0)).squeeze()
    return prediction.tolist()

def plot_prediction(y_pred, y_true = None):
    """Plots prediction, optionally with ground truth."""
    x = range(1, len(y_pred)+1)
    plt.plot(x, y_pred, color="red", label="Predicted")
    if y_true is not None:
        plt.plot(x, y_true, color="blue", label="Ground truth")
        plt.legend()
    plt.xlabel("Sequence position")
    plt.ylabel("Structure score")
    plt.xlim(1, len(y_pred))
    plt.ylim(-0.1, 1.1)
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file')
    parser.add_argument('--seq', type=str, required=True, help='Protein sequence')
    args = parser.parse_args()
    protein_seq = args.seq.upper()
    if set(protein_seq) > set("ACDEFGHIKLMNPQRSTVWY"):
        raise ValueError(f"The string is not a protein sequence")
    config.update_from_yaml(file_path = args.config)
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ProteinRNN(**config.model).to(device)
    checkpoint = f"{os.path.dirname(__file__)}/models/{config.model_name}/model.pth"
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"The trained model {config.model_name} was not found.")
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    y_pred = predict(protein_seq, model)
    plot_prediction(y_pred)

if __name__ == "__main__":
    main()
