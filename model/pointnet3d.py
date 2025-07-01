import torch
import torch.nn as nn
import torch.nn.functional as F

# TNet: Learns a transformation matrix to align input point clouds or features.
# Purpose: Makes the network invariant to geometric transformations (like rotation).
class TNet3D(nn.Module):
    def __init__(self, k=3):
        super(TNet3D, self).__init__()
        self.k = k  # Input dimension (e.g., 3 for xyz coordinates)
        # 1D convolution layers to extract features from input points (No of input channels, No of output channels[number of filters/kernels], Kernel size)
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        # Fully connected layers to regress the transformation matrix (input no of features, output no of features)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k) 

        # Batch normalization normalizes activations to stabilize and speed up training.
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
class PointNetSeg3D(nn.Module):
    def __init__(self, num_classes=2):
        super(PointNetSeg3D, self).__init__()
        self.input_transform = TNet3D(k=3)
        self.feature_transform = TNet3D(k=64)

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Conv1d(1088, 512, 1)
        self.fc2 = nn.Conv1d(512, 256, 1)
        self.fc3 = nn.Conv1d(256, num_classes, 1)

        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        B, N, _ = x.size()
        x = x.transpose(2, 1)

        transform = self.input_transform(x)
        x = torch.bmm(transform, x)

        x1 = F.relu(self.bn1(self.conv1(x)))  # (B, 64, N)

        # Feature transform
        feature_transform = self.feature_transform(x1)
        x1 = torch.bmm(feature_transform, x1)  # (B, 64, N)

        x2 = F.relu(self.bn2(self.conv2(x1)))  # (B, 128, N)
        x3 = F.relu(self.bn3(self.conv3(x2)))          # (B, 1024, N)

        global_feat = torch.max(x3, 2, keepdim=True)[0]
        global_feat = global_feat.repeat(1, 1, N)

        concat = torch.cat([x1, global_feat], 1)

        x = F.dropout(F.relu(self.bn4(self.fc1(concat))), p=0.3, training=self.training)
        x = F.dropout(F.relu(self.bn5(self.fc2(x))), p=0.3, training=self.training)

        x = self.fc3(x)
        return x.transpose(2, 1), feature_transform  # Return feature_transform for regularization

# feature_transform_regularizer: Regularization loss for transformation matrices.
# Purpose: Encourages the transformation matrix to be close to orthogonal (prevents overfitting).
def feature_transform_regularizer(trans):
    d = trans.size(1)  # Matrix dimension
    I = torch.eye(d, device=trans.device)[None, :, :]  # Identity matrix
    # Frobenius norm of (trans * trans^T - I)
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss
