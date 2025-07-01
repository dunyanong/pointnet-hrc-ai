import torch
import torch.nn as nn
import torch.nn.functional as F

# Transformation Network (TNet)
class TNet9D(nn.Module):
    def __init__(self, k=9):
        super(TNet, self).__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        B = x.size(0)

        x = F.relu(self.bn1(self.conv1(x)))     # (B, 64, N)
        x = F.relu(self.bn2(self.conv2(x)))     # (B, 128, N)
        x = F.relu(self.bn3(self.conv3(x)))     # (B, 1024, N)

        x = torch.max(x, 2)[0]                  # (B, 1024)
        x = F.relu(self.bn4(self.fc1(x)))       # (B, 512)
        x = F.relu(self.bn5(self.fc2(x)))       # (B, 256)
        x = self.fc3(x)                         # (B, k*k)

        identity = torch.eye(self.k, device=x.device).view(1, self.k * self.k).repeat(B, 1)
        x = x + identity
        return x.view(B, self.k, self.k)

# PointNet for semantic segmentation with 9D input
class PointNetSeg9D(nn.Module):
    def __init__(self, num_classes=3):
        super(PointNetSeg9D, self).__init__()
        self.input_transform = TNet9D(k=9)
        self.feature_transform = TNet9D(k=64)

        self.conv1 = nn.Conv1d(9, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.conv4 = nn.Conv1d(1088, 512, 1)
        self.conv5 = nn.Conv1d(512, 256, 1)
        self.conv6 = nn.Conv1d(256, num_classes, 1)

        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        B, N, D = x.size()  # (B, N, 9)
        x = x.transpose(2, 1)  # (B, 9, N)

        t_input = self.input_transform(x)       # (B, 9, 9)
        x = torch.bmm(t_input, x)               # Apply transform: (B, 9, N)

        x1 = F.relu(self.bn1(self.conv1(x)))    # (B, 64, N)

        t_feat = self.feature_transform(x1)     # (B, 64, 64)
        x1 = torch.bmm(t_feat, x1)              # (B, 64, N)

        x2 = F.relu(self.bn2(self.conv2(x1)))   # (B, 128, N)
        x3 = F.relu(self.bn3(self.conv3(x2)))   # (B, 1024, N)

        global_feat = torch.max(x3, 2, keepdim=True)[0]  # (B, 1024, 1)
        global_feat = global_feat.repeat(1, 1, N)        # (B, 1024, N)

        concat = torch.cat([x1, global_feat], dim=1)     # (B, 64+1024=1088, N)

        x = F.dropout(F.relu(self.bn4(self.conv4(concat))), p=0.3, training=self.training)
        x = F.dropout(F.relu(self.bn5(self.conv5(x))), p=0.3, training=self.training)
        x = self.conv6(x)                                 # (B, num_classes, N)

        return x.transpose(2, 1), t_feat  # (B, N, num_classes), transform matrix

# Feature transform regularizer
def feature_transform_regularizer(trans):
    d = trans.size(1)
    I = torch.eye(d, device=trans.device)[None, :, :]
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss
