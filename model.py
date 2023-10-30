import torch
import torch.nn as nn

from typing import List, Callable, Any


class ConvBlock(nn.Module):
    def __init__(self,
                in_channels: int,
                out_channels: int,
                kernel_size: int,
                stride: int = 1,
                padding: int = 0,
            ) -> None:
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)

        return x


class Encoder(nn.Module):
    def __init__(self,
                channels: List[int],
                kernel_sizes: List[int],
                strides: List[int],
                paddings: List[int],
                fc_hidden_dims: List[int],
                output_dim: int
                ) -> None:
        super(Encoder, self).__init__()

        self.conv_blocks = nn.ModuleList()
        self.fc_layers = nn.ModuleList()

        for i in range(len(kernel_sizes) - 1):
            conv_block = ConvBlock(
                channels[i],
                channels[i+1],
                kernel_sizes[i],
                strides[i],
                paddings[i]
            )
            self.conv_blocks.append(conv_block)
        
        for i in range(len(fc_hidden_dims) - 1):
            fc_layer = nn.Linear(fc_hidden_dims[i], fc_hidden_dims[i+1])
            self.fc_layers.append(fc_layer)

        self.fc_layers.append(nn.Linear(fc_hidden_dims[-1], output_dim))
    
    def forward(self, x):
        for conv_block in self.conv_blocks:
            x = conv_block(x)
            x = nn.functional.max_pool2d(x, 2)
        
        x = torch.flatten(x, start_dim=1)

        for fc_layer in self.fc_layers:
            x = fc_layer(x)
            x = nn.functional.relu(x)
            x = nn.functional.dropout(x, p=0.2)

        return x


class Decoder(nn.Module):
    def __init__(self, fc_layers: List[int], output_dim: List[int] = [3, 224, 224]):
        super(Decoder, self).__init__()

        if len(output_dim) != 3:
            raise ValueError("output_dim must be a list of length 3")

        self.output_dim = output_dim

        self.fc_layers = nn.ModuleList()
        for i in range(len(fc_layers) - 1):
            fc_layer = nn.Linear(fc_layers[i], fc_layers[i+1])
            self.fc_layers.append(fc_layer)

    def forward(self, x):
        for fc_layer in self.fc_layers:
            x = fc_layer(x)
            x = nn.functional.relu(x)
            x = nn.functional.dropout(x, p=0.2)
        
        x = x.view(-1, self.output_dim[0], self.output_dim[1], self.output_dim[2])

        return x


class ResnetEncoder(nn.Module):
    def __init__(self, output_dim: int):
        super(ResnetEncoder, self).__init__()
        self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        self.resnet.fc = nn.Linear(512, output_dim)
    
    def forward(self, x):
        x = self.resnet(x)

        return x


class AutoEncoder(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder):
        super(AutoEncoder, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, x):
        latent = self.encoder(x)
        out = self.decoder(latent)

        return out, latent
    
decoder = Decoder([128, 256])