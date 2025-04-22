#!/usr/bin/env python3

import torch
import numpy as np
import os

# neural network models for collision prediction
# these implement the paper's architecture but simplified

class standaloneencoder(torch.nn.Module):
    """resnet8-based encoder for depth images"""
    def __init__(self, size_latent=128, dropout_rate=0.0):
        super().__init__()
        self.nb_chan = 1  # depth images
        self.size_latent = size_latent
        self.dropout_rate = dropout_rate

        # this architecture closely follows the paper's resnet8
        self.layers = torch.nn.ModuleDict({
            'input': torch.nn.Sequential(
                torch.nn.Conv2d(self.nb_chan, 32, kernel_size=(5,5), stride=2, padding=(1,3)),
                torch.nn.MaxPool2d(kernel_size=(3,3), stride=2),
            ),
            'res_layer_1': torch.nn.Conv2d(32, 32, kernel_size=(1,1), stride=2, padding=0),
            'res_block_1': torch.nn.Sequential(
                torch.nn.BatchNorm2d(32),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.dropout_rate),
                torch.nn.Conv2d(32, 32, kernel_size=(3,3), stride=2, padding=1),
                torch.nn.BatchNorm2d(32),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.dropout_rate),
                torch.nn.Conv2d(32, 32, kernel_size=(3,3), stride=1, padding=1),
            ),
            'res_layer_2': torch.nn.Conv2d(32, 64, kernel_size=(1,1), stride=2, padding=0),
            'res_block_2': torch.nn.Sequential(
                torch.nn.BatchNorm2d(32),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.dropout_rate),
                torch.nn.Conv2d(32, 64, kernel_size=(3,3), stride=2, padding=1),
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.dropout_rate),
                torch.nn.Conv2d(64, 64, kernel_size=(3,3), stride=1, padding=1),
            ),
            'res_layer_3': torch.nn.Conv2d(64, 128, kernel_size=(1,1), stride=2, padding=0),
            'res_block_3': torch.nn.Sequential(
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.dropout_rate),
                torch.nn.Conv2d(64, 128, kernel_size=(3,3), stride=2, padding=1),
                torch.nn.BatchNorm2d(128),
                torch.nn.Conv2d(128, 128, kernel_size=(3,3), stride=1, padding=1),
            ),
            'output': torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Dropout(self.dropout_rate),
                torch.nn.AvgPool2d(kernel_size=(3,3), stride=2),
                torch.nn.Flatten(),
            ),
            'mean': torch.nn.Linear(3584, size_latent),
            'logvar': torch.nn.Linear(3584, size_latent),
        })

    def forward(self, input):
        # forward pass through the network
        x = self.layers['input'](input)
        res = self.layers['res_layer_1'](x)
        x = self.layers['res_block_1'](x)
        x = x + res  # residual connection
        res = self.layers['res_layer_2'](x)
        x = self.layers['res_block_2'](x)
        x = x + res  # residual connection
        res = self.layers['res_layer_3'](x)
        x = self.layers['res_block_3'](x)
        x = x + res  # residual connection
        x = self.layers['output'](x)
        mean = self.layers['mean'](x)
        logvar = self.layers['logvar'](x)
        return mean, logvar

    def encode(self, input):
        # just get the mean (used during inference)
        mean, _ = self.forward(input)
        return mean


class standalonelinear(torch.nn.Module):
    """mlp-based collision predictor"""
    def __init__(self, nb_states=3, size_latent=128):
        super().__init__()
        # smaller network than paper but same concept
        # we keep it simple for inference speed
        self.layers = torch.nn.ModuleDict({
            'state_in': torch.nn.Sequential(
                torch.nn.Linear(nb_states, 16),
                torch.nn.Tanh(),
            ),
            'main': torch.nn.Sequential(
                torch.nn.Linear(16+size_latent, 64),
                torch.nn.Tanh(),
                torch.nn.Linear(64, 64),
                torch.nn.Tanh(),
                torch.nn.Linear(64, 32),
                torch.nn.Tanh(),
            ),
            'colpred': torch.nn.Sequential(
                torch.nn.Linear(32, 1),
            ),
        })

    def forward(self, state, latent):
        # forward pass of collision prediction
        x = self.layers['state_in'](state)
        x = torch.cat((x, latent), 1)
        x = self.layers['main'](x)
        x = self.layers['colpred'](x)
        # sigmoid activation to output probability of collision
        x = torch.nn.functional.sigmoid(x)
        return x


def load_neural_models(model_path, size_latent=128, device='cuda'):
    """load both neural networks from checkpoint file"""
    # create models
    encoder = standaloneencoder(size_latent).to(device)
    linear = standalonelinear(3, size_latent).to(device)
    
    print(f"loading model from: {model_path}")
    
    try:
        # load weights - might need to fix the paths here
        checkpoint = torch.load(model_path, map_location=device)
        
        # split into encoder and linear parts
        encoder_dict, linear_dict = {}, {}
        
        for key, value in checkpoint.items():
            if key.startswith('encoder.'):
                encoder_dict[key.replace('encoder.', '')] = value
            elif key.startswith('linear.'):
                linear_dict[key.replace('linear.', '')] = value
            elif any(key.startswith(x) for x in ['input', 'res_', 'output', 'mean', 'logvar']):
                encoder_dict[key] = value
            elif any(key.startswith(x) for x in ['state_in', 'main', 'colpred']):
                linear_dict[key] = value
        
        # load weights
        encoder.load_state_dict(encoder_dict, strict=False)
        linear.load_state_dict(linear_dict, strict=False)
        
        # set to evaluation mode
        encoder.eval()
        linear.eval()
        
        return encoder, linear
    except Exception as e:
        print(f"error loading models: {e}")
        return None, None
