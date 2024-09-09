import os
import torch
from utils.session_utils import start_session, parse_args
from processing.vision import process_vision_session
from processing.sequential import process_sequential_session


def get_default(name):
  return {
    'image': 'run_config/default_image.yaml',
    'video': 'run_config/default_video.yaml',
    'text': 'run_config/default_text.yaml',
    'timeseries': 'run_config/default_timeseries.yaml',
    'diffusion': 'run_config/default_diffusion.yaml',
  }[name]

if __name__ == '__main__':

  # ------------------------ Load session -----------------------

  args = parse_args()

  if args.name and len(args.name) > 0:
    config = get_default(args.name)
  elif args.config and len(args.config) > 0:
    config = args.config
  else:
    # default
    config = get_default('diffusion')
  
  # ------------------------ Init session -----------------------

  session, is_vision, is_equential = start_session(config)
  

  # ------------------------ Run session -----------------------

  if is_vision:
    process_vision_session(session)
  elif is_equential:
    process_sequential_session(session)
