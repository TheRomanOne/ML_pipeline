from pandas.plotting import scatter_matrix
from tqdm import tqdm
import torch
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import numpy as np
from models.VAE import VAE_SR
from train import train_model
from analysis import run_full_analysis
from utils import evaluate_latent_batches
from session_utils import start_session, load_weights, parse_args
import image_utils as i_utils
import os

os.system('clear')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('\n\nDevice:', device, '\n\n')



if __name__ == '__main__':
  config = parse_args(
    default='config/default.yaml'
  ).config

  session = start_session(config)
  scene_type = session['type']
  scene_name = session['name']
  session_path = session['path']

  to_horizontal = False
  ds, dl = i_utils.load_image_ds(
      video_path = session['video_path'],
      max_size=64*2, # dont change these values
      ratio=4, # dont change these values,
      to_horizontal=to_horizontal
  )

  X_gt = ds.X_gt
  y_gt = ds.y_gt
  
  params = session['params']
  n_epochs = params['n_epochs']
  model_type = params['model_type']

  example_index = 50
  test_image = X_gt[example_index].to(device)
  test_target = y_gt[example_index].to(device)


  print("Saving data sample from: X_gt, y_gt")
  i_utils.render_image(test_image.cpu(), f'{session_path}/images', 'X_gt', to_horizontal=to_horizontal)
  i_utils.render_image(test_target.cpu(), f'{session_path}/images', 'y_gt', to_horizontal=to_horizontal)

  if model_type == 'VAE_SR':
    latent_dim = params['latent_dim']
    vae_sr = VAE_SR(latent_dim, in_shape=X_gt.shape).to(device)
    
    if params['load_weights']:
      load_weights(vae_sr, session['weights_path'])
      
    else:
      print('Creating a new model')

  else:
    print(f"Model {model_type} is not supported yet")
    exit()

  losses, frames = train_model(
    dl=dl,
    model=vae_sr,
    n_epochs=n_epochs,
    device=device,
    test_image=test_image,
    lr=1e-3
  )

  print("Saving weights")
  torch.save(vae_sr.state_dict(), session['weights_path'])
  i_utils.plot_losses(losses, save_path=f'{session_path}/images')


  latents_np = evaluate_latent_batches(vae_sr, X_gt, batches=16)


  print("Plotting loss")
  losses = losses.reshape(n_epochs, -1).mean(axis=1)
  i_utils.plot_interpolation(vae_sr, latents_np, f'{session_path}/videos')

  run_full_analysis(latents_np, save_path=f'{session_path}/images')

  i_utils.create_video('learning', frames, save_path=f'{session_path}/videos', to_horizontal=to_horizontal, limit_frames=500)

  i_utils.sample_images(
      n=2,
      net=vae_sr,
      images=X_gt,
      labels=y_gt,
      title='samples',
      to_horizontal=to_horizontal,
      save_path=f'{session_path}/images'
  )


  s = X_gt.shape[0]
  sampled_indices = np.random.choice(s, 20, replace=False)
  sampled_indices = np.sort(sampled_indices)
  _imgs = X_gt[-s:][sampled_indices]
  i_utils.interpolate_images(vae_sr, _imgs, steps=5, save_path=f'{session_path}/videos', to_horizontal=to_horizontal, sharpen=True)

  index = torch.randint(0, X_gt.size(0), (1,)).item()
  print('image index:', index)

  i_utils.random_walk_image(
      img=X_gt[index],
      net=vae_sr,
      angle=25,
      steps=200,
      change_prob=.95,
      to_horizontal=to_horizontal,
      save_path=f'{session_path}/videos'
  )
