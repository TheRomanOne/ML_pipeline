import torch.nn as nn
import yaml
from architecture_parser import parse_architecture

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()

        with open('models/text/config/lstm.yaml', 'r') as yaml_file:
            lstm_architecture = yaml.safe_load(yaml_file)

        self.lstm = parse_architecture(lstm_architecture)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=8, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(2 * hidden_dim, output_dim)  # Adjust hidden_dim for bidirectional LSTM

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out
