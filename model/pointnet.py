import torch
import torch.nn as nn
import torch.nn.functional as F

# TNet: Learns a transformation matrix to align input point clouds or features.
# Purpose: Makes the network invariant to geometric transformations (like rotation).
class TNet(nn.Module):
    def __init__(self, k=3):
        super(TNet, self).__init__()
        self.k = k  # Input dimension (e.g., 3 for xyz coordinates)
        # 1D convolution layers to extract features from input points
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        # Fully connected layers to regress the transformation matrix
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)  # Output is a flattened k x k matrix

        # Batch normalization for stable training
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        B = x.size(0)  # Batch size

        # Pass through convolutional layers with ReLU and batch norm
        x = F.relu(self.bn1(self.conv1(x)))    # (B, 64, N)
        x = F.relu(self.bn2(self.conv2(x)))    # (B, 128, N)
        x = F.relu(self.bn3(self.conv3(x)))    # (B, 1024, N)
        # Global feature: max pooling across all points
        x = torch.max(x, 2)[0]                 # (B, 1024)

        # Pass through fully connected layers
        x = F.relu(self.bn4(self.fc1(x)))      # (B, 512)
        x = F.relu(self.bn5(self.fc2(x)))      # (B, 256)
        x = self.fc3(x)                        # (B, k*k)

        # Add identity to make the transformation close to identity at the start
        identity = torch.eye(self.k, device=x.device).view(1, self.k * self.k).repeat(B, 1)
        x = x + identity
        # Reshape to (B, k, k) transformation matrices
        return x.view(B, self.k, self.k)

# PointNetSeg: Main segmentation network for point clouds.
# Purpose: Segments each point in the input point cloud into one of the given classes.
class PointNetSeg(nn.Module):
    def __init__(self, num_classes=2):
        super(PointNetSeg, self).__init__()
        # Input transformation network for alignment
        self.input_transform = TNet(k=3)
        # Feature extraction layers
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        # Segmentation head: combines local and global features
        self.fc1 = nn.Conv1d(1088, 512, 1)
        self.fc2 = nn.Conv1d(512, 256, 1)
        self.fc3 = nn.Conv1d(256, num_classes, 1)

        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        B, N, _ = x.size()         # B: batch size, N: number of points
        x = x.transpose(2, 1)      # Change shape to (B, 3, N) for Conv1d

        # Input transformation (learned alignment)
        transform = self.input_transform(x)    # (B, 3, 3)
        x = torch.bmm(transform, x)            # Apply transformation to input points

        # Feature extraction
        x1 = F.relu(self.bn1(self.conv1(x)))   # (B, 64, N)
        x2 = F.relu(self.bn2(self.conv2(x1)))  # (B, 128, N)
        x3 = self.bn3(self.conv3(x2))          # (B, 1024, N)

        # Global feature: max pooling across all points
        global_feat = torch.max(x3, 2, keepdim=True)[0]  # (B, 1024, 1)
        global_feat = global_feat.repeat(1, 1, N)        # (B, 1024, N)

        # Concatenate local and global features
        concat = torch.cat([x1, global_feat], 1)         # (B, 1088, N)

        # Segmentation head
        x = F.relu(self.bn4(self.fc1(concat)))           # (B, 512, N)
        x = F.relu(self.bn5(self.fc2(x)))                # (B, 256, N)
        x = self.fc3(x)                                  # (B, num_classes, N)
        return x.transpose(2, 1)                         # (B, N, num_classes)

# feature_transform_regularizer: Regularization loss for transformation matrices.
# Purpose: Encourages the transformation matrix to be close to orthogonal (prevents overfitting).
def feature_transform_regularizer(trans):
    d = trans.size(1)  # Matrix dimension
    I = torch.eye(d, device=trans.device)[None, :, :]  # Identity matrix
    # Frobenius norm of (trans * trans^T - I)
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss
