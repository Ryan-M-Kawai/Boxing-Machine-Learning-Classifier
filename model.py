import torch
import torch.nn as nn

class PunchClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_classes=8):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, dropout=0.3)
        self.fc1  = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.3)
        self.fc2  = nn.Linear(32, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc2(self.drop(self.relu(self.fc1(out[:, -1, :]))))
        return out