import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset

from model.pointnet3d import PointNetSeg3D, feature_transform_regularizer

# Purpose: Set random seeds for reproducibility of results.
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Purpose: Define batch size and number of points per point cloud.
BATCH_SIZE = 16
NUM_POINTS = 1024

# H5PointCloudDataset: Loads point cloud data and labels from an HDF5 file.
# Purpose: Custom Dataset for loading point clouds and their segmentation labels.
class H5PointCloudDataset(Dataset):
    def __init__(self, h5_path):
        with h5py.File(h5_path, 'r') as f:
            self.points = f['point'][:]  # shape: (N, num_points, 3)
            self.labels = f['labels'][:]  # shape: (N, num_points)

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        # Returns a tuple: (point cloud, labels) for a single sample
        return torch.tensor(self.points[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

# Purpose: Load the dataset and create a DataLoader for batching and shuffling.
h5_path = '/home/internship/Documents/pointnet-hrc-vision/LabelledPoints/centre_centre_lighting.h5'
dataset = H5PointCloudDataset(h5_path)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

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
        output = model(data)  # (B, N, 2)
        output = output.permute(0, 2, 1)  # (B, 2, N)
        output = output.reshape(-1, output.shape[1])  # (B*N, 2)
        target = target.view(-1)  # (B*N,)

        # Regularization loss: Encourages transformation matrix to be close to orthogonal.
        transform = model.input_transform(data.transpose(2, 1))
        reg_loss = feature_transform_regularizer(transform)
        loss = criterion(output, target) + 0.001 * reg_loss

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Calculate accuracy
        preds = output.argmax(dim=1)
        correct += (preds == target).sum().item()
        total += target.numel()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    print(f'Epoch {epoch+1}/{EPOCHS} Loss: {avg_loss:.4f} | Acc: {accuracy:.4f}')