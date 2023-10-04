import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import os

from typing import List, Callable, Any


class RatDataset(Dataset):
    def __init__(self, file_path: str, transform: Callable = None):
        self.file_path = file_path
        self.transform = transform

        self.data = []

        for image_files in os.listdir(file_path):
            image = Image.open(os.path.join(file_path, image_files)).convert('RGB')
            self.data.append(image)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Any:
        image = self.data[idx]

        if self.transform:
            image = self.transform(image)

        return image