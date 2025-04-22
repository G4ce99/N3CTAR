import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import pandas as pd
import os
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_seed(n_channels, env_dim):
    seed = torch.zeros((n_channels, env_dim, env_dim, env_dim), dtype=torch.float32)
    seed[3:, env_dim//2, env_dim//2, env_dim//2] = 1.
    return seed

class NCA(nn.Module):
    def __init__(self, n_channels, env_dim, learn_seed=True, seed_std=0.01, update_prob=0.5, alive_thres=0.1, overgrowth_to_undergrowth_penalty=1.0):
        super().__init__()

        self.update_prob = update_prob
        self.alive_thres = alive_thres
        self.overgrowth_to_undergrowth_penalty = overgrowth_to_undergrowth_penalty

        self.seed = generate_seed(n_channels, env_dim).to(device)
        if learn_seed:
            self.seed[4:, env_dim//2, env_dim//2, env_dim//2] = torch.randn(n_channels-4) * seed_std
            self.seed = nn.Parameter(self.seed)

        # Perception layer
        self.perceive = nn.Conv3d(n_channels, 3*n_channels, kernel_size=3, padding=1)
        self.norm1 = nn.LayerNorm((3 * n_channels, env_dim, env_dim, env_dim))
        
        # Processing layers with residual connections
        self.process1 = nn.Conv3d(3*n_channels, 2*n_channels, kernel_size=1)
        self.process2 = nn.Conv3d(2*n_channels, n_channels, kernel_size=1)
    
    def forward(self, x, use_soft_living_mask=False):
        update_mask = (torch.rand(x[:, :1].shape, dtype=torch.float32).to(device) <= self.update_prob).float()
        
        y = self.perceive(x)
        y = self.norm1(y)
        y = F.relu(y)
        
        z = self.process1(y)
        z = F.relu(z)
        dx = self.process2(z)
        
        x = x + dx*update_mask
        
        if self.training:
            alpha = 0.8 
            soft_mask = F.sigmoid((F.max_pool3d(x[:, 3:4], kernel_size=3, stride=1, padding=1) - self.alive_thres) * 50)
            hard_mask = (F.max_pool3d(x[:, 3:4], kernel_size=3, stride=1, padding=1) > self.alive_thres).float()
            living_mask = alpha * soft_mask + (1-alpha) * hard_mask
        else:
            living_mask = (F.max_pool3d(x[:, 3:4], kernel_size=3, stride=1, padding=1) > self.alive_thres).float()
            
        return x * living_mask
    
    def get_loss(self, x, target_vox):
        target = torch.tile(target_vox, (x.shape[0], 1, 1, 1, 1))
        x_rgba = x[:, :4]

        x_living_mask = (x_rgba[:, 3:4] > self.alive_thres).float()
        target_living_mask = (target[:, 3:4] > self.alive_thres).float()

        undergrowth_loss = F.mse_loss(x_rgba * target_living_mask, target * target_living_mask) 
        overgrowth_loss = self.overgrowth_to_undergrowth_penalty * F.mse_loss(x_rgba * (1.-target_living_mask) * x_living_mask, torch.zeros(target.shape).to(device))
        loss = undergrowth_loss + overgrowth_loss
        return loss, undergrowth_loss, overgrowth_loss
