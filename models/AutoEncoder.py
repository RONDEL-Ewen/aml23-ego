import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from types import SimpleNamespace

class AutoEncoder(nn.Module):

    def __init__(
        self,
        num_emg_features = 16,
        num_classes = 400,      # Arbitrary as not used here
        dropout_rate = 0.5
    ):
        
        super(AutoEncoder, self).__init__()

        # Dynamically import I3d here to avoid circular import problems
        from models.I3D import I3D

        with open('./configs/I3D_save_feat.yaml', 'r') as file:
            config = yaml.safe_load(file)
        model_config_dict = config['models']['RGB']
        model_config = SimpleNamespace(**model_config_dict)

        # Encoder
        self.encoder = I3D(
            num_class = num_classes,
            modality = 'RGB',
            model_config = model_config
        )

        # Freeze I3D model parameters
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.flatten = nn.Flatten()
        
        # Decoder
        self.lstm = nn.LSTM(
            input_size = 1024,
            hidden_size = 512,
            num_layers = 1,
            batch_first = True
        )
        self.bn = nn.BatchNorm1d(512)
        self.fc = nn.Linear(
            512,
            num_emg_features
        )
        
    def forward(
        self,
        x
    ):
        
        # Encoder
        x = self.encoder(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = self.flatten(x)
        
        # Decoder
        x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        x = self.bn(lstm_out)
        x = self.fc(x)

        return x