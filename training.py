import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer


def train_epoch(model: nn.Module, dataloader: DataLoader, loss: _Loss, optimizer: Optimizer, device: torch.device):
    loss_sum = 0
    batches = 0

    model.train()

    for image_batch in dataloader:
        image_batch = image_batch.to(device)
    
        out, _ = model(image_batch)
        loss = loss(out, image_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
        batches += 1

        print(f"Loss: {loss.item()}")

    return loss_sum / batches


def validate(model: nn.Module, dataloader: DataLoader, loss: _Loss, optimizer: Optimizer, device: torch.device):
    loss_sum = 0
    batches = 0

    model.eval()

    with torch.no_grad():
        for image_batch in dataloader:
            image_batch = image_batch.to(device)

            out, _ = model(image_batch)
            loss = loss(out, image_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            batches += 1
    
    return loss_sum / batches