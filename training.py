import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer


def train_epoch(model: nn.Module, dataloader: DataLoader, loss_fn: _Loss, optimizer: Optimizer, device: torch.device):
    loss_sum = 0
    batches = 0

    model.train()

    for image_batch, transformed_image_batch in dataloader:
        image_batch = image_batch.to(device)
        transformed_image_batch = transformed_image_batch.to(device)

        # Use augmented images as input to the model
        out, _ = model(transformed_image_batch)
        # but original images as targets
        loss = loss_fn(out, image_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
        batches += 1

        print(f"Loss: {loss.item()}")

    return loss_sum / batches


def validate(model: nn.Module, dataloader: DataLoader, loss_fn: _Loss, device: torch.device):
    loss_sum = 0
    batches = 0

    model.eval()

    with torch.no_grad():
        for image_batch, transformed_image_batch in dataloader:
            image_batch = image_batch.to(device)
            transformed_image_batch = transformed_image_batch.to(device)

            # Use augmented images as input to the model
            out, _ = model(transformed_image_batch)
            # but original images as targets
            loss = loss_fn(out, image_batch)

            loss_sum += loss.item()
            batches += 1
    
    return loss_sum / batches