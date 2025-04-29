import numpy as np
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm
import os

import modal

from config import *
from model import NCA

#################
# setup modal   #
#################
# NOTE: go into settings and setup token if you haven't already
# app = modal.App("n3ctar-training")

#################
# prepare data  #
#################

vox_data = np.load(img_data_path)
coords = vox_data[:, :3]
min_coords = coords.min(axis=0)
max_coords = coords.max(axis=0)

# Normalize to [0, 1], then scale to [0, environment_size-1]
scaled_coords = (coords - min_coords) / (max_coords - min_coords + 1e-8)
voxel_coords = (scaled_coords * (environment_size - 1)).astype(int)  # shape: (N, 3)

rgba_vox = np.zeros((environment_size, environment_size, environment_size, 4), dtype=np.float32)

for i in range(len(voxel_coords)):
  x, y, z = voxel_coords[i]
  r, g, b = vox_data[i, 3:] / 255.0
  rgba_vox[x, y, z] = [r, g, b, 1.0] # 1.0 is for alpha

rgba_voxels = torch.tensor(rgba_vox, dtype=torch.float32).permute(3, 0, 1, 2).to(device)
print(f"rgba_voxels shape: {rgba_voxels.shape}, {rgba_voxels.dtype}") # (4, 32, 32, 32) 

#################
#   init model  #
#################
model = NCA(input_channels, 
            environment_size, 
            learn_seed=learn_seed, 
            update_prob=update_prob, 
            alive_thres=alive_thres, 
            overgrowth_to_undergrowth_penalty=over_to_under_penalty)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

def init_weights(m):
  if isinstance(m, nn.Conv3d):
    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    m.weight.data *= 0.1
    if m.bias is not None:
      nn.init.constant_(m.bias, 0)

model.apply(init_weights)


##############
# init wandb # - move this to config later
##############

if wandb_log:
  project_name = "n3ctar"
  run_name = model_name
  wandb_run = wandb.init(project=project_name, name=run_name)

# @app.function(gpu="A100-40GB", region="us-east")
def train():
  losses = []
  empty_cache_n_iter = 10 # 25

  for i in tqdm(range(epochs)):
    x = model.seed.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
    curriculum = np.random.randint(0, min(len(curriculum_eval_updates), 1 + (i//epoch_per_curriculum)))
    min_eval_iter, max_eval_iter = curriculum_eval_updates[curriculum]
    n_eval_iter = np.random.randint(min_eval_iter, max_eval_iter)
    if n_eval_iter > 0:
      model.eval()
      with torch.no_grad():
          for _ in range(n_eval_iter):
              x = model(x, i<0.05*epochs)

    model.train()
    n_iter = np.random.randint(min_iter, max_iter)
    for _ in range(n_iter):
      x = model(x, i<0.05*epochs)
    loss, underloss, overloss = model.get_loss(x, rgba_voxels)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # Tighter gradient clipping
    optimizer.step()
    losses.append(loss.item())
    if wandb_log:
      metrics = {
          "Total Loss": loss.item(),
          "Undergrowth Loss": underloss.item(),
          "Overgrowth Loss": overloss.item()
      }
      wandb_run.log(metrics, step=i)

    if  i % empty_cache_n_iter == empty_cache_n_iter-1:
      if not wandb_log:
          print(f"Epoch: {i}, Loss: {loss.item()}, Undergrowth Loss: {underloss.item()}, Overgrowth Loss: {overloss.item()}")
      torch.cuda.empty_cache()

  if wandb_log:
    wandb_run.finish()

  os.makedirs("./ckpts", exist_ok=True)
  torch.save(model.state_dict(), f"./ckpts/{model_name}.pth")

train()