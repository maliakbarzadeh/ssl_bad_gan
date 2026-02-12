import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np

class Discriminative(nn.Module):
    """Discriminator for 2D two circles dataset
    
    Outputs K+1 classes where:
    - Classes 0 to K-1: Real data classes (inner=0, outer=1)
    - Class K: Generated/Fake data (fake=2)
    """
    def __init__(self, config):
        super(Discriminative, self).__init__()
        
        self.noise_size = config.noise_size
        self.num_label = config.num_label  # K real classes
        self.num_classes = getattr(config, 'num_classes', config.num_label + 1)  # K+1 classes
        
        # Simple MLP for 2D input
        self.core_net = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        
        self.out_net = nn.Linear(128, self.num_classes)  # K+1 outputs
    
    def forward(self, X, feat=False):
        features = self.core_net(X)
        if feat:
            return features
        else:
            return self.out_net(features)


class Generator(nn.Module):
    """Generator for 2D two circles dataset"""
    def __init__(self, noise_size=20, output_size=2):
        super(Generator, self).__init__()
        
        self.noise_size = noise_size
        self.output_size = output_size
        
        self.core_net = nn.Sequential(
            nn.Linear(noise_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, output_size),
            nn.Tanh()  # Output in range [-1, 1]
        )
    
    def forward(self, noise):
        return self.core_net(noise)


class Encoder(nn.Module):
    """Encoder for 2D two circles dataset (maps back to noise)"""
    def __init__(self, input_size=2, noise_size=20, output_params=False):
        super(Encoder, self).__init__()
        
        self.noise_size = noise_size
        self.output_params = output_params
        
        self.core_net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        
        if output_params:
            # Output mean and log_sigma for variational inference
            self.output_layer = nn.Linear(128, noise_size * 2)
        else:
            self.output_layer = nn.Linear(128, noise_size)
    
    def forward(self, x):
        features = self.core_net(x)
        output = self.output_layer(features)
        
        if self.output_params:
            mu, log_sigma = torch.chunk(output, 2, dim=1)
            return mu, log_sigma
        else:
            return output
