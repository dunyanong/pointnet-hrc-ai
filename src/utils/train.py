import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os 
import sys
import glob

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model.model import PointNetSegmentation, feature_transform_regularizer
from utils.extract_files import (
    load_point_cloud_data, 
    normalize_point_cloud, 
    resample_point_cloud,
    read_off_file,
    create_binary_labels
)

class PointCloudDataset(Dataset):
    """Dataset class for point cloud data in .off format"""
    def __init__(self, data_path, num_points=1024, split='train'):
        self.data_path = data_path
        self.num_points = num_points
        self.split = split
        self.current_epoch = 0
        self.adaptive_alpha = True
        
        # Load point cloud data from .off files
        self.point_clouds, self.labels = load_point_cloud_data(data_path, split)
    
    def set_epoch(self, epoch):
        """Set current epoch for adaptive alpha scheduling"""
        self.current_epoch = epoch
        
    def __len__(self):
        return len(self.point_clouds)
    
    def __getitem__(self, idx):
        point_cloud = self.point_clouds[idx].copy()
        label = self.labels[idx].copy()
        
        # Resample point cloud to target size
        point_cloud, label = resample_point_cloud(
            point_cloud, label, self.num_points, 
            self.current_epoch, self.adaptive_alpha
        )
        
        # Normalize point cloud to unit sphere
        point_cloud = normalize_point_cloud(point_cloud)
        
        return torch.FloatTensor(point_cloud).transpose(1, 0), torch.LongTensor(label)
    
def train_model():
    # Hyperparameters
    batch_size = 16 
    num_epochs = 200
    learning_rate = 0.0005 
    weight_decay = 1e-3 
    num_points = 1024
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Dataset paths
    data_path = 'Dataset/ModelNet40'
    
    # Create datasets
    train_dataset = PointCloudDataset(data_path, num_points=num_points, split='train')
    
    # Binary classification: 2 classes only
    num_classes = 2
    print(f"Binary classification with {num_classes} classes")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Model
    model = PointNetSegmentation(num_classes=num_classes).to(device)
    
    # Binary classification loss
    loss_type = nn.CrossEntropyLoss()
    # Use AdamW optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # Cosine annealing scheduler for better convergence
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    # Training loop
    best_acc = 0.0
    for epoch in range(num_epochs):
        # Update dataset epoch for adaptive alpha
        train_dataset.set_epoch(epoch)
        
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad() 
            
            # Forward pass
            output, feature_transform = model(data)
            
            # Reshape target to match output
            target = target.view(-1)
            
            # Binary classification loss
            loss = loss_type(output, target)
            
            # Adaptive feature transform regularization
            reg_weight = 0.01 * (1.0 - min(epoch / 200.0, 0.5))  # Reduce regularization over time
            reg_loss = feature_transform_regularizer(feature_transform)
            total_loss = loss + reg_weight * reg_loss
            
            # Backward pass
            total_loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Statistics
            train_loss += total_loss.item() 
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
            
            if batch_idx % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f'Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx}, '
                      f'Loss: {total_loss.item():.4f}, LR: {current_lr:.6f}')
        
        # Print epoch results
        train_acc = 100. * train_correct / train_total
        avg_loss = train_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {avg_loss:.4f}')
        print(f'Train Accuracy: {train_acc:.2f}%')
        print(f'Current LR: {optimizer.param_groups[0]["lr"]:.6f}')
        print('-' * 60)
        
        # Step scheduler
        scheduler.step()
        
        # Save best model
        if train_acc > best_acc:
            best_acc = train_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_acc': train_acc,
                'best_acc': best_acc,
            }, 'Checkpoints/best_model.pt')
        
        # Save model checkpoint
        if (epoch + 1) % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_acc': train_acc,
                'best_acc': best_acc,
            }, f'Checkpoints/checkpoint_epoch_{epoch+1}.pt')
    
    # Save final model
    torch.save(model.state_dict(), 'DeployModel/laptop_classifier.pt')
    print(f"Training completed! Best accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    train_model()