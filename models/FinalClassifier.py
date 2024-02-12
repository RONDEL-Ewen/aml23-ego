import torch
from torch import nn


class Classifier(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        """
        [TODO]: the classifier should be implemented by the students and different variations of it can be tested
        in order to understand which is the most performing one """

    def forward(self, x):
        return self.classifier(x), {}

class MLPClassifier(nn.Module):

    def __init__(
        self,
        input_size = 1024,
        hidden_size = 512,
        dropout_ratio = 0.5,
        num_classes = 8
    ):
        super(MLPClassifier, self).__init__()
        #self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout_ratio)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(
        self,
        x
    ):
        # Max pooling over the frames
        #x = x.permute(0, 2, 1)  # Change shape to [batch_size, features, frames]
        #x = self.pool(x).squeeze(-1)  # Apply max pooling and remove the last dimension
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x, {}
    
class LSTMClassifier(nn.Module):

    def __init__(
        self,
        input_size = 1024,
        hidden_size = 512,
        num_layers = 2,
        dropout_ratio = 0.5,
        num_classes = 8
    ):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first = True
        )
        self.dropout = nn.Dropout(dropout_ratio)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(
        self,
        x
    ):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Initialiser les états cachés et cellulaires à zéro
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        #x = x.unsqueeze(1)

        # Avancer à travers le LSTM
        out, (hn, cn) = self.lstm(x, (h0, c0))

        out = self.dropout(out)
        out = self.relu(out)

        out = self.dropout(out[:, -1, :])
        out = self.relu(out)
        out = self.fc(out)

        return out, {}