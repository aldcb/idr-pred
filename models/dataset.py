# Copyright (c) 2024 aldcb - GPLv3 (http://gnu.org/licenses/gpl.html)

import os

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, Subset

class ProteinDataset(Dataset):
    """
    Dataset for protein sequences in NFQ (Not FastQ) files using different @/+ headers and two score bytes per residue.
    Arguments:
        file_name (str): name of NFQ file in the data directory.
        device (torch.device): device where tensors will be stored.
    """
    def __init__(self, file_name, device):
        self.device = device
        file_path = self.get_nfq_path(file_name)
        self.x, self.y, self.metadata = self.load_nfq(file_path)
    
    def get_nfq_path(self, file_name):
        """Gets the absolute path to a NFQ file."""
        data_dir = "data"
        file_path = os.path.dirname(__file__) + f"/../{data_dir}/" + file_name
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file {file_path} doesn't exist")
        if not file_path.lower().endswith((".nfq", ".nfastq")):
            raise FileNotFoundError(f"Data file {file_path} has unexpected extension")
        return file_path
    
    def load_nfq(self, file_path):
        """Loads and processes data from the NFQ file."""
        x, y, metadata = [], [], []
        with open(file_path, "r") as file:
            lines = file.readlines()
            for i in range(0, len(lines), 4):
                x.append(protein_to_indexes(lines[i+1].strip(), self.device))
                scores = [((ord(chr1)-27)/1e2 + (ord(chr2)-27)/1e4)  # normalized
                          for chr1, chr2 in zip(lines[i+3][::2], lines[i+3][1::2])]
                y.append(torch.tensor(scores, dtype=torch.float16).to(self.device))
                metadata.append({
                    "name": lines[i].strip("@\n"),
                    "id": lines[i+2].strip("+\n"),
                    "sequence": lines[i+1].strip()
                })
        return x, y, metadata
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, i):
        """Returns the index-encoded sequence and target score tensors."""
        return self.x[i], self.y[i]

def protein_to_indexes(sequence, device, amino_acids = "ACDEFGHIKLMNPQRSTVWY"):
    """Converts a protein sequence string to a tensor of amino acid indexes."""
    mapping = {aa: i for i, aa in enumerate(amino_acids)}
    indexes = torch.tensor([mapping[aa] for aa in sequence.upper()], dtype=torch.int32).to(device)
    return indexes

def score_to_classes(score, thresholds):
    """Converts a score tensor to a class array based on a list of thresholds."""
    thresholds.sort()
    classes = torch.zeros(len(score), dtype=torch.int32, device=score.device)
    for i, threshold in enumerate(thresholds):
        classes[score > threshold] = i + 1
    return classes.cpu().numpy()

def subdivide_by_length(dataset, thresholds = None):
    """Splits a dataset into a list of subsets based on sequence length using the provided thresholds."""
    if thresholds == None:
        return [dataset]
    thresholds.sort()
    n_buckets = len(thresholds) + 1
    indices = [[] for _ in range(n_buckets)]
    for i, (sequence, _) in enumerate(dataset):
        label_length = len(sequence)
        for j, threshold in enumerate(thresholds):
            if label_length <= threshold:
                indices[j].append(i)
                break
        else:
            indices[-1].append(i)
    return [Subset(dataset, i) for i in indices]

def custom_collate(batch):
    """Collates a batch of variable-length sequences with padding and masking."""
    sequences, targets = zip(*batch)
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=0)
    masks = (padded_targets != 0).type(torch.bool)
    return padded_sequences, padded_targets, masks
