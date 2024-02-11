import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import pandas as pd

class FeaturesDataset(Dataset):

    def __init__(
        self,
        features_file,
        labels_file,
        label_to_index
    ):
        with open(features_file, 'rb') as f:
            data = pickle.load(f)
            df = pd.DataFrame(data)
        
        with open(labels_file, 'rb') as f:
            data2 = pickle.load(f)
            df2 = pd.DataFrame(data2)

        self.features = torch.tensor([item['features_RGB'] for item in df['features']]).float()
        self.labels = df2['narration'].values
        self.label_to_index = label_to_index
        self.labels = [self.label_to_index[label] for label in self.labels]

    def __len__(
        self
    ):
        return len(self.features)

    def __getitem__(
        self,
        idx
    ):
        return self.features[idx], self.labels[idx]