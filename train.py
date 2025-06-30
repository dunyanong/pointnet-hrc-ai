import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model.pointnet import PointNetSeg

# Mock data
BATCH_SIZE = 16
NUM_POINTS = 1024

# Example dummy dataset: B x N x 3
points = torch.randn(100, NUM_POINTS, 3)
labels = torch.randint(0, 2, (100, NUM_POINTS))  # 0 = background, 1 = robot

dataset = TensorDataset(points, labels)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PointNetSeg(num_classes=2).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
EPOCHS = 20
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)  # B x N x 2
        output = output.reshape(-1, 2)
        target = target.view(-1)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f'Epoch {epoch+1}/{EPOCHS} Loss: {total_loss / len(dataloader):.4f}')
