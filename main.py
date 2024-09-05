import os
import torch
from train import train_model
from analysis import run_full_analysis, eval_and_interp
from utils.utils import evaluate_latent_batches
from utils.session_utils import start_session, load_model_from_params, parse_args
import utils.image_utils as i_utils
import warnings

warnings.filterwarnings("ignore")

os.system('clear')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('\n\nDevice:', device, '\n\n')



if __name__ == '__main__':
  config = parse_args(
    default='config/default.yaml'
  ).config

  
  # ------------------------ Init session and DB -----------------------

  session = start_session(config)
  session_name = session['session_name']
  session_path = session['path']
  
  dataset = session['dataset']
  scene_type = dataset['type']
  
  resize_ratio = dataset['resize_ratio']
  max_size = dataset['max_size']
  to_horizontal = dataset['is_horizontal']

  params = session['nn_params']
  n_epochs = params['n_epochs']
  
  # Create Dataset
  if scene_type == 'videos':
    ds, dataloader = i_utils.load_video_ds(
        video_path = session['dataset_path'],
        id_from = 0,
        id_to=-1,
        max_size=max_size,
        ratio=resize_ratio,
        to_horizontal=to_horizontal
    )
  elif scene_type == 'images':
    ds, dataloader = i_utils.load_image_ds(
        video_path = session['dataset_path'],
        max_size=max_size,
        ratio=resize_ratio,
        to_horizontal=to_horizontal
    )

  X_gt = ds.X_gt
  y_gt = ds.y_gt
  example_index = 50
  test_image = X_gt[example_index].to(device)
  test_target = y_gt[example_index].to(device)


  print("Saving data sample from: X_gt, y_gt")
  i_utils.render_image(test_image.cpu(), f'{session_path}/images', 'X_gt', to_horizontal=to_horizontal)
  i_utils.render_image(test_target.cpu(), f'{session_path}/images', 'y_gt', to_horizontal=to_horizontal)






  # ---------------------- Create and train model  ---------------------
  session.update({'input_shape': X_gt.shape[2:]})
  vae_sr = load_model_from_params(session)
  vae_sr.to(device)

  losses, frames = train_model(
    dataloader=dataloader,
    vae=vae_sr,
    test_image=test_image,
    params=params
  )
  i_utils.plot_losses(losses, n_epochs, save_path=f'{session_path}/images')

  print("Saving weights")
  torch.save(vae_sr.state_dict(), session['weights_path'])

  print('Creating learning video')
  i_utils.create_video('learning', frames, save_path=f'{session_path}/videos', to_horizontal=to_horizontal, limit_frames=500)






  # --------------------------- Run analysis --------------------------- 

  if len(session['analysis']) == 0:
    print('No analysis was specified')
    exit()

  if 'evaluate_and_interpolate':
    eval_and_interp(vae_sr, X_gt, y_gt, to_horizontal, session_path)
  
  if 'full_latent_analysis' in session['analysis']:
    latents_np = evaluate_latent_batches(vae_sr, X_gt, batches=16)
    i_utils.plot_interpolation(vae_sr, latents_np, f'{session_path}/images')
    run_full_analysis(latents_np, save_path=f'{session_path}/images')


