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
class StanceClassifier(nn.Module):
    def __init__(self, input_size=8, hidden_size=32, num_classes=2):
        super().__init__()
        self.fc1  = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.3)
        self.fc2  = nn.Linear(hidden_size, 16)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.3)
        self.fc3  = nn.Linear(16, num_classes)

    def forward(self, x):
        out = self.drop1(self.relu1(self.fc1(x)))
        out = self.drop2(self.relu2(self.fc2(out)))
        out = self.fc3(out)
        return out