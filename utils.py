import torch
import torchvision.transforms as T

def unnormalize(tensor, mean, std):
    mean = torch.Tensor(mean)
    std = torch.Tensor(std)
    
    return T.functional.normalize(tensor, -mean/std, 1/std)

class Unnormalize(callable):
    def __init__(self, mean, std):
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)

    def __call__(self, tensor):
        return unnormalize(tensor, self.mean, self.std)