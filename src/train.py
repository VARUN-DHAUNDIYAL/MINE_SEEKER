# utils/dataset.py
import torch
import numpy as np
import os
from torch.utils.data import Dataset

class RFDataset(Dataset):
    def __init__(self, data_dir, label_file=None):
        self.data_dir = data_dir
        self.data_files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
        self.labels = np.load(label_file) if label_file else None

    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.data_files[idx])
        data = np.load(file_path)
        
        # Ensure correct shape (2, N)
        data = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
        
        # Get label if available
        label = torch.tensor(self.labels[idx]) if self.labels is not None else torch.tensor(0)

        return data, label
