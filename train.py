import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset

from model.pointnet3d import PointNetSeg3D, feature_transform_regularizer
# from h5_dataset import H5PointCloudDataset

# Purpose: Set random seeds for reproducibility of results.
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Purpose: Define batch size and number of points per point cloud.
BATCH_SIZE = 16
NUM_POINTS = 1024


class NpyPointCloudDataset(Dataset):
    def __init__(self, npy_path, num_points=1024):
        data = np.load(npy_path, allow_pickle=True)
        if isinstance(data, np.ndarray) and data.dtype == object:
            data = data.item()  
        data = np.array(data)
        self.points = data[:, :3]
        self.labels = data[:, 3].astype(np.uint8)
        self.labels[self.labels == 255] = 1
        self.num_points = num_points
        self.length = len(self.points) // self.num_points

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        start = idx * self.num_points
        end = start + self.num_points
        pts = self.points[start:end]
        lbl = self.labels[start:end]
        return torch.tensor(pts, dtype=torch.float32), torch.tensor(lbl, dtype=torch.long)

# Purpose: Load the dataset and create a DataLoader for batching and shuffling.
npy_path = '//home/internship/Documents/LabelledPoints_bin/centre_centre_lighting.npy'
dataset = NpyPointCloudDataset(npy_path, num_points=NUM_POINTS)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print("Label min:", dataset.labels.min(), "Label max:", dataset.labels.max())

# Purpose: Initialize the PointNet segmentation model, optimizer, and loss function.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PointNetSeg3D(num_classes=2).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop: Trains the model for a set number of epochs.
# Purpose: For each epoch, iterate over the data, compute loss (including regularization), update weights, and print metrics.
EPOCHS = 20
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, feature_transform = model(data) 
        output = output.permute(0, 2, 1)  # (B, 2, N)
        output = output.reshape(-1, output.shape[1])  # (B*N, 2)
        target = target.view(-1)  # (B*N,)

        # Regularization loss for feature transform
        reg_loss = feature_transform_regularizer(feature_transform)
        loss = criterion(output, target) + 0.001 * reg_loss

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        preds = output.argmax(dim=1)
        correct += (preds == target).sum().item()
        total += target.numel()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    print(f'Epoch {epoch+1}/{EPOCHS} Loss: {avg_loss:.4f} | Acc: {accuracy:.4f}')
