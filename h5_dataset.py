import h5py
import torch
from torch.utils.data import Dataset

class H5PointCloudDataset(Dataset):
    def __init__(self, h5_path):
        with h5py.File(h5_path, 'r') as f:
            self.points = f['points'][:]
            self.labels = f['labels'][:]
    def __len__(self):
        return len(self.points)
    def __getitem__(self, idx):
        return torch.tensor(self.points[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)