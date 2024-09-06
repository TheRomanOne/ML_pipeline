
from torch.utils.data import DataLoader
from datasets.VideoDataset import VideoDataset
from datasets.ImageDataset import ImageDataset
from utils.text_utils import parse_text

def load_video_ds(video_path, id_from, id_to, max_size, ratio=4, zoom_pixels=0, to_horizontal=False):
  video_dataset = VideoDataset(video_path, id_from, id_to, max_size, ratio, zoom_pixels, to_horizontal)
  print('Video dataset length', video_dataset.X_gt.shape)
  video_loader = DataLoader(video_dataset, batch_size=16, shuffle=True)

  return video_dataset, video_loader

def load_image_ds(dir_path, max_size, ratio=4, to_horizontal=False):
  image_dataset = ImageDataset(dir_path, max_size, ratio, to_horizontal)
  print('Image dataset length', len(image_dataset))
  image_loader = DataLoader(image_dataset, batch_size=16, shuffle=True)

  return image_dataset, image_loader

def load_dataset(session):
    dataset = session['dataset']
    scene_type = dataset['type']

    if scene_type == 'videos':
        # dataset = (dataset, dataloader)
        dataset = load_video_ds(
            video_path = session['dataset_path'],
            id_from = 0,
            id_to=-1,
            max_size=dataset['max_size'],
            ratio=dataset['resize_ratio'],
            to_horizontal=dataset['is_horizontal']
        )
    elif scene_type == 'images':
        # dataset = (dataset, dataloader)
        dataset = load_image_ds(
            dir_path = session['dataset_path'],
            max_size=dataset['max_size'],
            ratio=dataset['resize_ratio'],
            to_horizontal=dataset['is_horizontal']
        )
    elif scene_type == 'text':
        with open(session['dataset_path'], 'r') as file:
            text = file.read()
        
        # dataset = (raw_data, word_to_index, index_to_word, words)
        dataset = parse_text(text)

    return dataset