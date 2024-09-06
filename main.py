import os
import torch
from utils.session_utils import start_session, load_model_from_params, parse_args
from datasets.Loader import load_dataset
from processing.vision import process_vision_session
import warnings

warnings.filterwarnings("ignore")

os.system('clear')
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('\n\nDevice:', device, '\n\n')



if __name__ == '__main__':

  vision_example = 'run_config/default_vision.yaml'
  text_example = 'run_config/default_text.yaml'

  config = parse_args(default=vision_example).config

  
  # ------------------------ Init session and DB -----------------------

  session, is_vision, is_text = start_session(config)
  session_name = session['session_name']
  session_path = session['path']
  params = session['nn_params']
  n_epochs = params['n_epochs']
  
  dataset, dataloader = load_dataset(session)

  if is_vision:
    process_vision_session(session, dataset, dataloader)
