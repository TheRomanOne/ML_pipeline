
from torch.utils.data import DataLoader
from datasets.VideoDataset import VideoDataset
from datasets.ImageDataset import ImageDataset
from datasets.TextDataset import TextDataset
from datasets.TimeSeriesDataset import TimeSeriesDataset


def load_video_ds(video_path, batch_size, max_size, ratio, to_horizontal):
  video_dataset = VideoDataset(video_path, max_size, ratio, to_horizontal)
  dataloader = DataLoader(video_dataset, batch_size=batch_size, shuffle=True)
  print('Video dataset length', video_dataset.X_gt.shape)

  return  video_dataset, dataloader 

def load_image_ds(dir_path, batch_size, max_size, ratio, to_horizontal):
  image_dataset = ImageDataset(dir_path, max_size, ratio, to_horizontal)
  dataloader = DataLoader(image_dataset, batch_size=batch_size, shuffle=True)
  print('Image dataset length', len(image_dataset))

  return  image_dataset, dataloader 

def load_text(text_path, batch_size, seq_length):
    text_dataset = TextDataset(text_path, seq_length)
    dataloader = DataLoader(text_dataset, batch_size=batch_size, shuffle=True)
    print('Text dataset length', len(text_dataset))
    print('Vocabulary length', len(text_dataset.vocabulary))

    return text_dataset, dataloader

def load_csv(csv_path, batch_size, seq_length, target_column):
    csv_dataset = TimeSeriesDataset(csv_path, seq_length, target_column)
    dataloader = DataLoader(csv_dataset, batch_size=batch_size, shuffle=True)
    print('CSV dataset length', len(csv_dataset))

    return csv_dataset, dataloader

def load_dataset(session):
    dataset_config = session['dataset']
    scene_type = dataset_config['type']


    if scene_type == 'video':
        dataset = load_video_ds(
            video_path = dataset_config['dataset_path'],
            batch_size=dataset_config['batch_size'],
            max_size=dataset_config['max_size'],
            ratio=dataset_config['resize_ratio'],
            to_horizontal=dataset_config['is_horizontal']
        )
    elif scene_type == 'image':
        dataset = load_image_ds(
            dir_path = dataset_config['dataset_path'],
            batch_size=dataset_config['batch_size'],
            max_size=dataset_config['max_size'],
            ratio=dataset_config['resize_ratio'],
            to_horizontal=dataset_config['is_horizontal']
        )
    elif scene_type == 'text':
        dataset = load_text(
           text_path=dataset_config['dataset_path'],
           batch_size=dataset_config['batch_size'],
           seq_length=session['nn']['params']['seq_length']
        )
        session['nn']['params']['input_dim'] = len(dataset[0].word_to_index)
        session['nn']['params']['output_dim'] = len(dataset[0].word_to_index)

    elif scene_type == 'timeseries':
        dataset = load_csv(
           csv_path=dataset_config['dataset_path'],        
           batch_size=dataset_config['batch_size'],
           seq_length=session['nn']['params']['seq_length'],
           target_column=dataset_config['target_column']
        )
    
    
    return dataset