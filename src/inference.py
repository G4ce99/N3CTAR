import os
import matplotlib
import matplotlib.pyplot as plt
from moviepy import ImageSequenceClip
from tqdm import tqdm

from config import *
from utils import *
from model import NCA

def voxel_plot(color_voxels, fig, ax, save_fig=False, img_dir=None, iter=None):
    # os.makedirs(img_dir, exist_ok=True)
    if save_fig:
        os.makedirs(os.path.join(img_dir, model_name), exist_ok=True)

    ax.cla()
    rgba_vox = color_voxels.permute(1, 2, 3, 0).detach().cpu().numpy()
    ax.voxels(rgba_vox[..., 3] > alive_thres, facecolors=rgba_vox[..., :3])
    ax.set_xlim(0, environment_size)
    ax.set_ylim(0, environment_size)
    ax.set_zlim(0, environment_size)
    if save_fig:
        plt.savefig(os.path.join(img_dir, model_name, f"voxel_{iter}.png"))
    else:
        plt.show()

fig = plt.figure(figsize=(32, 32))
ax = fig.add_subplot(projection='3d')

model = NCA(input_channels, 
            environment_size, 
            learn_seed=learn_seed, 
            update_prob=update_prob, 
            alive_thres=alive_thres, 
            overgrowth_to_undergrowth_penalty=over_to_under_penalty)

model.load_state_dict(torch.load(f"ckpts/{model_name}_1000.pth"))
model.to(device)

matplotlib.use('Agg')

model.eval()
num_imgs = 200
# img_dir = "/home/henry/N3CTAR/data/test/imgs"
# video_dir = "/home/henry/N3CTAR/data/test"

img_dir = "./data/test/imgs"
video_dir = "./data/test"

with torch.no_grad():
    x = model.seed.unsqueeze(0)
    voxel_plot(torch.clamp(x[0, :4], min=0., max=1.), fig, ax, save_fig=True, img_dir=img_dir, iter=0)
    for i in tqdm(range(num_imgs)):
        x = model(x)
        voxel_plot(torch.clamp(x[0, :4], min=0., max=1.), fig, ax, save_fig=True, img_dir=img_dir, iter=i+1)


file_list = [os.path.join(img_dir, model_name, f"voxel_{i}.png") for i in range(num_imgs+1)]

clip = ImageSequenceClip(file_list, fps=12)
clip.write_videofile(os.path.join(video_dir, f"{model_name}.mp4"))