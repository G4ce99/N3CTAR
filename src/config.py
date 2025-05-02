import os
import torch
from pathlib import Path
import numpy as np

# data settings
img_data_path = Path().resolve().parent / "data" / "npy_voxels" / "mario_vox.npy"
environment_size = 32
alive_thres = 0.1

## model settings
model_name = "mario_testing_env32"

batch_size = 8
input_channels = 16
learn_seed = True
seed_std = 0.05
update_prob = 0.9 # 0.75
over_to_under_penalty = 10
damage_prob = 0.4

# seeting up curriculums
epochs_per_curric = 500
eval_update_samples = [0] * epochs_per_curric
interval_start, interval_end = 1, 64
interval_step = 64
j = epochs_per_curric
while interval_end <= 1024:
    sample = np.random.rand()
    if sample < 1/6:
        eval_update_samples.append(0)
    elif sample < 1/2:
        eval_update_samples.append(np.random.randint(0, interval_start))
    else:
        eval_update_samples.append(np.random.randint(interval_start, interval_end))
    
    if (j+1) % epochs_per_curric == 0:
        interval_start, interval_end = interval_end, interval_end + interval_step
    j += 1
epochs = len(eval_update_samples)

# # training hyperparams
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


min_iter, max_iter = 16, 58
learning_rate = 2e-3
weight_decay = 0

# misc settings
wandb_log = True
enable_profiling = False

# DDP settings
# n_gpus = 2