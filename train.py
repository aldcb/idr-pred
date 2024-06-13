#!/usr/bin/env python3
# Copyright (c) 2024 aldcb - GPLv3 (http://gnu.org/licenses/gpl.html)

import argparse
import os

import torch
from torch.nn.utils import clip_grad_value_
from torch.utils.data import DataLoader, random_split

from configs.config import config
from models.dataset import ProteinDataset, custom_collate
from models.model import ProteinRNN

def train_loop(model, dataloader, optimizer, criterion):
    model.train()
    losses = []
    for sequences, targets, masks in dataloader:
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs[masks], targets[masks].float())
        losses.append(loss.item())
        loss.backward()
        clip_grad_value_(model.parameters(), clip_value=1.0)
        optimizer.step()
    avg_loss = sum(losses) / len(losses)
    return avg_loss

def eval_loop(model, dataloader, criterion):
    model.eval()
    losses = []
    with torch.no_grad():
        for sequences, targets, masks in dataloader:
            outputs = model(sequences)
            loss = criterion(outputs[masks], targets[masks])
            losses.append(loss.item())
    avg_loss = sum(losses) / len(losses)
    return avg_loss

def inspect_prediction(model, dataloader, index):  # for monitoring/debugging (not used)
    """Returns prediction and ground truth for a single accession."""
    model.eval()
    with torch.no_grad():
        for i, (sequences, targets, masks) in enumerate(dataloader):
            if i == int(index/dataloader.batch_size):
                predictions = model(sequences)
                j = index % dataloader.batch_size
                length = sum(masks[j].tolist())
                y_real = targets[j].tolist()[:length]
                y_pred = predictions[j].tolist()[:length]
                return y_pred, y_real

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
    args = parser.parse_args()
    config.update_from_yaml(file_path = args.config)
    torch.manual_seed(config.seed)
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ProteinRNN(**config.model).to(device)
    dataset = ProteinDataset(config.data_file, device)
    train_dataset, val_dataset, _ = random_split(dataset, [config.train_ratio, config.val_ratio, config.test_ratio], generator=generator)
    train_dataloader = DataLoader(train_dataset, collate_fn=custom_collate, **config.dataloader, shuffle=True)
    val_dataloader = DataLoader(val_dataset, collate_fn=custom_collate, **config.dataloader, shuffle=False)
    criterion = getattr(torch.nn, config.criterion_name)()
    optimizer = getattr(torch.optim, config.optimizer_name)(model.parameters(), **config.optimizer)
    checkpoint_path = f"{os.path.dirname(__file__)}/models/{config.model_name}"
    if os.path.exists(checkpoint_path):
        raise FileExistsError("Trained model with same name already exists")
    os.mkdir(checkpoint_path)
    config_path = f"{os.path.dirname(__file__)}/configs/{config.model_name}.yml"
    if not os.path.exists(config_path):
        with open(args.config, "r") as config_file, open(config_path, "w") as copy_file:
            for line in config_file:
                copy_file.write(line)
    best_val_loss = float("inf")
    patience_count = 0
    for epoch in range(1, config.epochs + 1):
        print(f"Epoch {epoch}")
        train_loss = train_loop(model, train_dataloader, optimizer, criterion)
        print(f"Average Training Loss: {train_loss:.4f}")
        val_loss = eval_loop(model, val_dataloader, criterion)
        print(f"Average Validation Loss: {val_loss:.4f}")
        with open(f"{checkpoint_path}/losses.csv", mode='a') as csv_file:
            csv_file.write(f"{epoch},{train_loss},{val_loss}\n")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{checkpoint_path}/model.pth")
            torch.save(optimizer.state_dict(), f"{checkpoint_path}/optimizer.pth")
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= config.patience:
                print("Early stopping")
                break

if __name__ == "__main__":
    main()
