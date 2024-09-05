from pandas.plotting import scatter_matrix
from tqdm import tqdm
import torch
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import numpy as np
from models.VAE_landscape import VAE_SR_landscape
from train import train_model
from analysis import run_full_analysis
from utils import evaluate_latent_batches
import image_utils as i_utils

import shutil, os

os.system('clear')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('\n\nDevice:', device, '\n\n')
vid_p = {
    'clouds': '/home/roman/Desktop/ML/Datasets/videos/clouds_4_resize.mp4',
}

if os.path.isdir('session'):
  shutil.rmtree('session')

os.mkdir('session')
os.mkdir('session/images')
os.mkdir('session/videos')

to_horizontal = False

ds, dl = i_utils.load_video_ds(
    video_path = vid_p['clouds'],
    id_from = 0,
    id_to=-1,
    # zoom_pixels=50,
    max_size=64*2, # dont change these values
    ratio=4, # dont change these values,
    to_horizontal=to_horizontal
)
X_gt = ds.X_gt
y_gt = ds.y_gt
latent_dim = 10
n_epochs = 3

example_index = -49
test_image = X_gt[example_index].to(device)
test_target = y_gt[example_index].to(device)


print("Saving data sample from: X_gt, y_gt")
i_utils.render_image(test_image.cpu(), 'X_gt', to_horizontal=to_horizontal)
i_utils.render_image(test_target.cpu(), 'y_gt', to_horizontal=to_horizontal)

vae_sr = VAE_SR_landscape(latent_dim).to(device)
vae_sr.load_state_dict(torch.load("model.pth"))
losses, frames = train_model(
  dl=dl,
  model=vae_sr,
  n_epochs=n_epochs,
  device=device,
  test_image=test_image,
  lr=1e-4
)

print("Saving weights")
torch.save(vae_sr.state_dict(), 'model.pth')
i_utils.plot_losses(losses)
latents_np = evaluate_latent_batches(vae_sr, X_gt, batches=16)


to_vid = np.array(frames)
while to_vid.shape[0] > 500:
  to_vid = to_vid[::5]

print("Plotting loss")
losses = losses.reshape(n_epochs, -1).mean(axis=1)
i_utils.plot_interpolation(vae_sr, latents_np)

run_full_analysis(latents_np)
i_utils.create_video('learning', to_vid, to_horizontal=to_horizontal)



i_utils.sample_images(
    n=2,
    net=vae_sr,
    images=X_gt,
    labels=y_gt,
    title='samples',
    to_horizontal=to_horizontal
)


s = X_gt.shape[0]
# s = 100
sampled_indices = np.random.choice(s, 20, replace=False)
sampled_indices = np.sort(sampled_indices)
_imgs = X_gt[-s:][sampled_indices]
i_utils.interpolate_images(vae_sr, _imgs, steps=5, to_horizontal=to_horizontal, sharpen=True)

index = torch.randint(0, X_gt.size(0), (1,)).item()
print('image index', index)

i_utils.random_walk_image(
    img=X_gt[index],
    net=vae_sr,
    angle=25,
    steps=200,
    change_prob=.95,
    to_horizontal=to_horizontal
)
