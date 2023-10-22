import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer


def train_epoch(model: nn.Module, dataloader: DataLoader, loss: _Loss, optimizer: Optimizer):
    loss_sum = 0
    batches = 0

    for image_batch in dataloader:
    
        out, latent = model(image_batch)
        loss = loss(out, image_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
        batches += 1

        print(f"Loss: {loss.item()}")

    return loss_sum / batches


def validate(model: nn.Module, dataloader: DataLoader, loss: _Loss, optimizer: Optimizer):
    loss_sum = 0
    batches = 0

    with torch.no_grad():
        for image_batch in dataloader:
            out, latent = model(image_batch)
            loss = loss(out, image_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            batches += 1

            print(f"Loss: {loss.item()}")
    
    return loss_sum / batches