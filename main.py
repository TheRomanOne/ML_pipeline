import os
import torch
from utils.session_utils import start_session, load_config
from processing.vision import process_vision_session
from processing.sequential import process_sequential_session




if __name__ == '__main__':

  image_example = 'run_config/default_image.yaml'
  video_example = 'run_config/default_video.yaml'
  text_example = 'run_config/default_text.yaml'
  timeseries_example = 'run_config/default_timeseries.yaml'

  config = load_config(default=image_example).config

  
  # ------------------------ Init session and DB -----------------------

  session, is_vision, is_equential = start_session(config)
  

  if is_vision:
    process_vision_session(session)
  elif is_equential:
    process_sequential_session(session)
