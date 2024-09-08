import torch.nn as nn
import yaml, torch
from utils.architecture_parser import parse_architecture

class LSTMText(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(LSTMText, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        with open('custom_models/sequence_config/lstm.yaml', 'r') as yaml_file:
            lstm_architecture = yaml.safe_load(yaml_file)

        lstm_architecture['layers'][0]['params']['input_size'] = embedding_dim
        lstm_architecture['layers'][0]['params']['hidden_size'] = hidden_dim
        lstm_architecture['layers'][0]['params']['num_layers'] = num_layers
        seq, _ = parse_architecture(lstm_architecture)
        self.lstm = nn.Sequential(*seq)
        # self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=8, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(2 * hidden_dim, vocab_size)  # Adjust hidden_dim for bidirectional LSTM

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out

class LSTMTimeSeq(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_stacked_layers):
        super(LSTMTimeSeq, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_stacked_layers = n_stacked_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_stacked_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.register_buffer('device', None)

    def forward(self, x):
        batch_dim = x.size(0)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out
