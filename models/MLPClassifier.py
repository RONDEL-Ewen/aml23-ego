import torch
import torch.nn as nn

class MLPClassifier(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        num_classes
    ):
        super(MLPClassifier, self).__init__()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(
        self,
        x
    ):
        # Max pooling over the frames
        x = x.permute(0, 2, 1)  # Change shape to [batch_size, features, frames]
        x = self.pool(x).squeeze(-1)  # Apply max pooling and remove the last dimension
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x