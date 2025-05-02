import numpy as np
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm
import os

from torch.profiler import profile, record_function, ProfilerActivity
import modal

from config import *
from utils import *
from model import NCA


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
  project_name = "n3ctar_experiments"
  run_name = model_name
  wandb_run = wandb.init(project=project_name, name=run_name)

# @app.function(gpu="A100-40GB", region="us-east")
def train():
  losses = []
  empty_cache_n_iter = 10 # 25
  epoch_ckpt_freq = 1000

  with check_profile(
      enable_profiling,
      activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
      profile_memory=True,
      record_shapes=True,
      with_stack=True,
  ) as prof:
    for i in tqdm(range(epochs)):
      # stop profiling after 3 epochs
      if enable_profiling and i>= 3:
        break

      model.temperature = 5 + 45 * min(i/1000, 1.0)
      seed = model.get_seed()
      x = seed.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1).to(device)
      living_mask = (x[:,3:4] > model.alive_thres).float()
      
      n_eval_iter = eval_update_samples[i]
      
      if n_eval_iter > 0:
          model.eval()
          with torch.no_grad():
              for _ in range(n_eval_iter):
                  x, living_mask = model(x, living_mask, i<0.05*epochs)

              if np.random.rand() < damage_prob * min(1.0, (i%epochs_per_curric)/(0.2*epochs_per_curric)):
                  for __ in range(np.random.randint(1, 2)):
                      damage_x = np.random.randint(0, 27)
                      damage_y = np.random.randint(0, 27)
                      damage_z = np.random.randint(0, 27)
                      x[:, :, damage_x:damage_x+5, damage_y:damage_y+5, damage_z:damage_z+5] = 0
                      living_mask[:, :, damage_x:damage_x+5, damage_y:damage_y+5, damage_z:damage_z+5] = 0

      model.train()
      n_iter = np.random.randint(min_iter, max_iter)
      prev_x=None
      for _ in range(n_iter):
          prev_x=x
          x, living_mask = model(x, living_mask, i<0.05*epochs)
      loss, underloss, overloss, stabilityloss = model.get_loss(x, rgba_voxels, prev_x, stability_factor=10*i/epochs)
      optimizer.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # Tighter gradient clipping
      optimizer.step()
      losses.append(loss.item())
      if wandb_log:
          metrics = {
              "Total Loss": loss.item(),
              "Undergrowth Loss": underloss.item(),
              "Overgrowth Loss": overloss.item(),
              "Stability Loss": stabilityloss.item()
          }
          wandb_run.log(metrics, step=i)

      if (i+1) % epoch_ckpt_freq == 0:
          os.makedirs("./ckpts", exist_ok=True)
          torch.save(model.state_dict(), os.path.join(f"./ckpts/{model_name}_{i+1}.pth"))

      if  (i+1) % empty_cache_n_iter == 0:
          if not wandb_log:
              print(f"Epoch: {i}, Loss: {loss.item()}, Undergrowth Loss: {underloss.item()}, Overgrowth Loss: {overloss.item()}")
          torch.cuda.empty_cache()

  if enable_profiling:
    print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
    print()
    print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))
    # prof.export_chrome_trace(f"trace_rank{rank}.json")

  if wandb_log:
    wandb_run.finish()

  # os.makedirs("./ckpts", exist_ok=True)
  # torch.save(model.state_dict(), f"./ckpts/{model_name}.pth")

train()