# Copyright (c) 2024 aldcb - GPLv3 (http://gnu.org/licenses/gpl.html)

import torch.nn as nn

class ProteinRNN(nn.Module):
    """
    Model for protein sequence per-residue regression using a LSTM-based bidirectional RNN.
    Arguments:
        embedding_dim (int): dimension of the amino acid embedding vector.
        hidden_size (int): number of cells per layer.
        num_layers (int): number of layers.
        dropout (float): dropout probability at the last layer.
    """
    def __init__(self, embedding_dim, hidden_size, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(20, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True, bidirectional=True)
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
        self.dropout = nn.Dropout(p=dropout)
        self.predictor = nn.Linear(2*hidden_size, 1)

    def forward(self, sequences):
        embedded = self.embedding(sequences)
        output, _ = self.lstm(embedded)
        output = self.dropout(output)
        output = self.predictor(output)
        return output.squeeze(-1)
