import torch
from train.train import train_model
from utils.analysis import run_full_analysis, eval_and_interp, sample_plot_image
from utils.utils import evaluate_latent_batches, print_model_params
from utils.session_utils import load_model_from_params
import utils.vision_utils as i_utils
from datasets.Loader import load_dataset
import global_settings as gs


device = gs.device


def process_vision_session(session):
    
    session_path = session['path']
    training_config = session['training']
    dataset_config = session['dataset']
    nn_config = session['nn']



# ________________________________ Prepare dataset ________________________________ 

    dataset, dataloader = load_dataset(session)
    X_gt = dataset.X_gt
    y_gt = dataset.y_gt
    nn_config['params'].update({'input_shape': X_gt.shape[2:]})

    example_index = -49
    test_image = X_gt[example_index].to(device)
    test_target = y_gt[example_index].to(device)
    to_horizontal = dataset_config['is_horizontal']

    print("Saving data sample from: X_gt, y_gt")
    i_utils.render_image(test_image.cpu(), f'{session_path}/images', 'X_gt', to_horizontal=to_horizontal)
    i_utils.render_image(test_target.cpu(), f'{session_path}/images', 'y_gt', to_horizontal=to_horizontal)


    

# ________________________________ Initialize model ________________________________ 
 
    model = load_model_from_params(nn_config)
    print_model_params(model)




# __________________________________ Train model __________________________________ 

    losses, frames = train_model(
      dataloader=dataloader,
      model=model,
      config=training_config,
      test_sample=test_image,
    )
    i_utils.plot_losses(losses, training_config['n_epochs'], save_path=f'{session_path}/images')

    print("Saving weights")
    torch.save(model.state_dict(), nn_config['weights_path'])

    print('Creating learning video')
    i_utils.create_video('learning', frames, save_path=f'{session_path}/videos', to_horizontal=to_horizontal, limit_frames=500)




# __________________________________ Post process __________________________________

    if len(session['post_process']) == 0:
      print('No post processing was requested')
      exit() 

    if 'evaluate_and_interpolate' in session['post_process']:
      eval_and_interp(model, X_gt, y_gt, to_horizontal, session_path)
    
    if 'full_latent_analysis' in session['post_process']:
      latents_np = evaluate_latent_batches(model, X_gt, batches=16)
      i_utils.plot_interpolation(model, latents_np, f'{session_path}/images')
      run_full_analysis(latents_np, save_path=f'{session_path}/images')
    
    if 'test_diffusion' in session['post_process']:
      sample_plot_image(X_gt[0].shape, save_path=f'{session_path}/images')
          # model, X_gt, save_path=f'{session_path}/images')
