import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from model.model import PointNetSegmentation, feature_transform_regularizer

class PointCloudDataset(Dataset):
    """Dataset class for point cloud data"""
    def __init__(self, data_path, num_points=1024):
        self.data_path = data_path
        self.num_points = num_points
        self.point_clouds = []
        self.labels = []
        # TODO: Load your point cloud data and labels here
        
    def __len__(self):
        return len(self.point_clouds)
    
    def __getitem__(self, idx):
        point_cloud = self.point_clouds[idx]
        label = self.labels[idx]
        
        # Random sampling if more points than needed
        if point_cloud.shape[0] > self.num_points:
            choice = np.random.choice(point_cloud.shape[0], self.num_points, replace=False)
            point_cloud = point_cloud[choice, :]
            label = label[choice]
        
        # Padding if fewer points than needed
        elif point_cloud.shape[0] < self.num_points:
            padding = self.num_points - point_cloud.shape[0]
            point_cloud = np.concatenate([point_cloud, point_cloud[:padding]], axis=0)
            label = np.concatenate([label, label[:padding]], axis=0)
        
        return torch.FloatTensor(point_cloud).transpose(1, 0), torch.LongTensor(label)

def train_model():
    # Hyperparameters
    batch_size = 32
    num_epochs = 100
    learning_rate = 0.001
    num_points = 1024
    num_classes = 2  # robot/not-robot
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model
    model = PointNetSegmentation(num_classes=num_classes).to(device)
    
    # Loss and optimizer
    loss_type = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # TODO: Replace with your actual dataset
    # train_dataset = PointCloudDataset('path/to/train/data', num_points=num_points)
    # val_dataset = PointCloudDataset('path/to/val/data', num_points=num_points)
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Dummy data for testing
    train_loader = create_dummy_dataloader(batch_size, num_points, num_classes)
    val_loader = create_dummy_dataloader(batch_size, num_points, num_classes)
    
    # Training loop
    for epoch in range(num_epochs):
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
            
            # Main loss
            loss = loss_type(output, target)
            
            # Feature transform regularization
            reg_loss = feature_transform_regularizer(feature_transform)
            total_loss = loss + 0.001 * reg_loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += total_loss.item() 
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx}, '
                      f'Loss: {total_loss.item():.4f}')
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        #
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                
                output, feature_transform = model(data)
                target = target.view(-1)
                
                loss = loss_type(output, target)
                reg_loss = feature_transform_regularizer(feature_transform)
                total_loss = loss + 0.001 * reg_loss
                
                val_loss += total_loss.item()
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
        
        # Print epoch results
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')
        print('-' * 60)
        
        # Step scheduler
        scheduler.step()
        
        # Save model checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_acc': train_acc,
                'val_acc': val_acc,
            }, f'checkpoint_epoch_{epoch+1}.pth')
    
    # Save final model
    torch.save(model.state_dict(), 'pointnet_robot_removal.pth')
    print("Training completed!")

def create_dummy_dataloader(batch_size, num_points, num_classes):
    """Create dummy data for testing"""
    class DummyDataset(Dataset):
        def __init__(self, size=100):
            self.size = size
            
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            # Random point cloud (num_points x 3)
            points = torch.randn(3, num_points)
            # Random labels for each point
            labels = torch.randint(0, num_classes, (num_points,))
            return points, labels
    
    dataset = DummyDataset()
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

if __name__ == "__main__":
    train_model()