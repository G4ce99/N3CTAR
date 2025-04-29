import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

import os
from pathlib import Path

# data settings
img_data_path = Path().resolve().parent / "data" / "npy_voxels" / "mario_vox.npy"
environment_size = 32
alive_thres = 0.1

## model settings
model_name = "mario_filled_curriculum_epochs_1000_env64"
input_channels = 16
learn_seed = True
seed_std = 0.05
update_prob = 0.9 # 0.75
over_to_under_penalty = 1

# # training hyperparams
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 8
epochs = 1000
min_iter, max_iter = 48, 64 # 96, 128
curriculum_eval_updates = [(0, 1), (32, 65)]
epoch_per_curriculum = epochs // len(curriculum_eval_updates)

learning_rate = 2e-4
weight_decay = 0

# misc settings
wandb_log = True

# DDP settings
n_gpus = 2

def setup(rank, world_size):
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '12355'

  dist.init_process_group("nccl", rank=rank, world_size=world_size)
  torch.cuda.set_device(rank)

def cleanup():
  dist.destroy_process_group()