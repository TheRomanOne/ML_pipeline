import os
import torch
from utils.session_utils import start_session, load_config
from datasets.Loader import load_dataset
from processing.vision import process_vision_session
from processing.sequential import process_sequential_session
import warnings

warnings.filterwarnings("ignore")

os.system('clear')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('\n\nDevice:', device, '\n\n')

if str(device) != 'cuda':
  print("Cuda was unable to start. Since training on a CPU is a terrible idea, the process will now terminate.\nPlease consider restarting the computer ")
  exit()

torch.cuda.empty_cache()
torch.cuda.reset_max_memory_allocated()


if __name__ == '__main__':

  image_example = 'run_config/default_image.yaml'
  video_example = 'run_config/default_video.yaml'
  text_example = 'run_config/default_text.yaml'
  timeseries_example = 'run_config/default_timeseries.yaml'

  config = load_config(default=vision_example).config

  
  # ------------------------ Init session and DB -----------------------

  session, is_vision, is_equential = start_session(config)
  session_name = session['session_name']
  session_path = session['path']
  
  dataset = load_dataset(session)

  if is_vision:
    process_vision_session(session, *dataset)
  elif is_equential:
    process_sequential_session(session, *dataset)
