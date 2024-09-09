import torch
import numpy as np
from torch.utils.data import Dataset
from utils.text_utils import parse_text
from utils.utils import get_sequential_data

class TextDataset(Dataset):
    def __init__(self, text_path, seq_length):
        self.text_path = text_path
        self.seq_length = seq_length
        self._load_text()

    def _load_text(self):
        with open(self.text_path, 'r') as file:
            text = file.read()
        dataset, word_to_index, index_to_word, vocabulary = parse_text(text)
        sequences, labels = get_sequential_data(dataset, self.seq_length)

        X = np.array([[word_to_index[word] for word in seq] for seq in sequences])
        y = np.array([word_to_index[label] for label in labels])


        self.datase = dataset
        self.word_to_index = word_to_index
        self.index_to_word = index_to_word
        self.vocabulary = vocabulary
        
        self.X_gt = torch.tensor(X, dtype=torch.long)
        self.y_gt = torch.tensor(y, dtype=torch.long)


    def __len__(self):
        return len(self.X_gt)

    def __getitem__(self, idx):
        return self.X_gt[idx], self.y_gt[idx]
