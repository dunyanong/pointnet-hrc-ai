import torch
import torch.nn as nn
import torch.nn.functional as F

class TNet(nn.Module):
    """Transformation Network (T-Net) for learning transformation matrices"""
    def __init__(self, k=3):
        super(TNet, self).__init__()
        self.k = k
        
        # Shared MLPs
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Shared MLPs
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Max pooling
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        
        # Fully connected layers
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        
        # Initialize as identity matrix
        identity = torch.eye(self.k, dtype=torch.float32).view(1, self.k*self.k).repeat(batch_size, 1)
        if x.is_cuda:
            identity = identity.cuda()
        
        x = x + identity
        x = x.view(-1, self.k, self.k)
        return x
    

    

class PointNetEncoder(nn.Module):
    """PointNet encoder that extracts both local and global features"""
    def __init__(self):
        super(PointNetEncoder, self).__init__()
        
        # Input transform (3x3)
        self.input_transform = TNet(k=3)
        
        # First shared MLP
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        
        # Feature transform (64x64)
        self.feature_transform = TNet(k=64)
        
        # Second shared MLP
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)
        
    def forward(self, x):
        batch_size, _, num_points = x.size()
        
        # Input transform
        input_transform_matrix = self.input_transform(x)
        x = x.transpose(2, 1)  # (B, N, 3)
        x = torch.bmm(x, input_transform_matrix)  # Apply transformation
        x = x.transpose(2, 1)  # (B, 3, N)
        
        # First shared MLP (64, 64)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        # Store point features before feature transform
        point_features = x
        
        # Feature transform
        feature_transform_matrix = self.feature_transform(x)
        x = x.transpose(2, 1)  # (B, N, 64)
        x = torch.bmm(x, feature_transform_matrix)  # Apply transformation
        x = x.transpose(2, 1)  # (B, 64, N)
        
        # Second shared MLP (64, 128, 1024)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        
        # Max pooling to get global feature
        global_feature = torch.max(x, 2, keepdim=True)[0]  # (B, 1024, 1)
        
        return point_features, global_feature, feature_transform_matrix

class PointNetClassification(nn.Module):
    """PointNet for classification tasks"""
    def __init__(self, num_classes=2):  # 2 for robot/not-robot classification
        super(PointNetClassification, self).__init__()
        
        self.encoder = PointNetEncoder()
        
        # Classification head
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        point_features, global_feature, feature_transform = self.encoder(x)
        
        # Flatten global feature
        x = global_feature.view(-1, 1024)
        
        # Classification layers
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x, feature_transform

class PointNetSegmentation(nn.Module):
    """PointNet for segmentation tasks (robot removal)"""
    def __init__(self, num_classes=2):  # 2 for robot/not-robot per point
        super(PointNetSegmentation, self).__init__()
        
        self.encoder = PointNetEncoder()
        
        # Segmentation head
        self.conv1 = nn.Conv1d(1088, 512, 1)  # 64 (point features) + 1024 (global) = 1088
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.conv4 = nn.Conv1d(128, num_classes, 1)
        
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        batch_size, _, num_points = x.size()
        
        point_features, global_feature, feature_transform = self.encoder(x)
        
        # Expand global feature to match point dimensions
        global_feature_expanded = global_feature.repeat(1, 1, num_points)  # (B, 1024, N)
        
        # Concatenate point features with global feature
        combined_features = torch.cat([point_features, global_feature_expanded], dim=1)  # (B, 1088, N)
        
        # Segmentation layers
        x = F.relu(self.bn1(self.conv1(combined_features)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)  # (B, num_classes, N)
        
        x = x.transpose(2, 1).contiguous()  # (B, N, num_classes)
        x = x.view(-1, x.size(2))  # (B*N, num_classes)
        
        return x, feature_transform

def feature_transform_regularizer(transform):
    """
    Regularization term for feature transform matrix
    Encourages the transform to be close to orthogonal
    """
    d = transform.size(1)
    batch_size = transform.size(0)
    I = torch.eye(d)[None, :, :]
    if transform.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(transform, transform.transpose(2,1)) - I, dim=(1,2)))
    return loss