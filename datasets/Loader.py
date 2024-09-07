
from torch.utils.data import DataLoader
from datasets.VideoDataset import VideoDataset
from datasets.ImageDataset import ImageDataset
from datasets.TextDataset import TextDataset


def load_video_ds(video_path, max_size, ratio=4, zoom_pixels=0, to_horizontal=False):
  video_dataset = VideoDataset(video_path, max_size, ratio, zoom_pixels, to_horizontal)
  print('Video dataset length', video_dataset.X_gt.shape)
  video_loader = DataLoader(video_dataset, batch_size=16, shuffle=True)

  return  video_dataset, video_loader 

def load_image_ds(dir_path, max_size, ratio=4, to_horizontal=False):
  image_dataset = ImageDataset(dir_path, max_size, ratio, to_horizontal)
  print('Image dataset length', len(image_dataset))
  image_loader = DataLoader(image_dataset, batch_size=16, shuffle=True)

  return  image_dataset, image_loader 

def load_text(text_path, seq_length):
    text_dataset = TextDataset(text_path, seq_length)
    print('Text dataset length', len(text_dataset))
    print('Dictionary length', len(text_dataset.dictionary))
    text_loader = DataLoader(text_dataset, batch_size=256, shuffle=True)

    return text_dataset, text_loader

def load_dataset(session):
    dataset = session['dataset']
    scene_type = dataset['type']

    if scene_type == 'videos':
        dataset = load_video_ds(
            video_path = session['dataset_path'],
            max_size=dataset['max_size'],
            ratio=dataset['resize_ratio'],
            to_horizontal=dataset['is_horizontal']
        )
    elif scene_type == 'images':
        dataset = load_image_ds(
            dir_path = session['dataset_path'],
            max_size=dataset['max_size'],
            ratio=dataset['resize_ratio'],
            to_horizontal=dataset['is_horizontal']
        )
    elif scene_type == 'text':
        dataset = load_text(
           text_path=session['dataset_path'],
           seq_length=session['nn_params']['seq_length']
        
        )

    return dataset