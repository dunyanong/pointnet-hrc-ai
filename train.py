import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os 
import glob
from model.model import PointNetSegmentation, feature_transform_regularizer

class PointCloudDataset(Dataset):
    """Dataset class for point cloud data in .off format"""
    def __init__(self, data_path, num_points=1024, split='train'):
        self.data_path = data_path
        self.num_points = num_points
        self.split = split
        self.point_clouds = []
        self.labels = []
        
        # Load point cloud data from .off files
        self.load_data()
        
    def load_data(self):
        """Load point cloud data from ModelNet40 .off files"""
        # categories = os.listdir(self.data_path)
        # categories = [cat for cat in categories if os.path.isdir(os.path.join(self.data_path, cat))]

        categories = ["laptop"]
        
        print(f"Binary classification for laptop detection")
        
        for category in categories:
            category_path = os.path.join(self.data_path, category, self.split)
            if not os.path.exists(category_path):
                continue
                
            off_files = glob.glob(os.path.join(category_path, "*.off"))
            
            for off_file in off_files:
                try:
                    points = self.read_off_file(off_file)
                    if points is not None and len(points) > 0:
                        self.point_clouds.append(points)
                        # Create binary segmentation labels
                        point_labels = self.create_binary_labels(points)
                        self.labels.append(point_labels)
                except Exception as e:
                    print(f"Error loading {off_file}: {e}")
                    continue
                    
        print(f"Loaded {len(self.point_clouds)} point clouds for {self.split}")
    
    def create_binary_labels(self, points):
        """Create binary labels for laptop segmentation"""
        num_points = len(points)
        
        # Simple heuristic: assume center points are more likely to be laptop
        center = np.mean(points, axis=0)
        distances = np.linalg.norm(points - center, axis=1)
        threshold = np.percentile(distances, 60)  # 60% of points as laptop
        
        labels = np.where(distances <= threshold, 1, 0).astype(np.int64)  # 1=laptop, 0=background
        return labels
    
    def read_off_file(self, file_path):
        """Read point cloud from .off file"""
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                
            # Check if first line is 'OFF'
            if lines[0].strip() != 'OFF':
                return None
                
            # Parse header
            header = lines[1].strip().split()
            num_vertices = int(header[0])
            num_faces = int(header[1])
            
            # Read vertices (points)
            points = []
            for i in range(2, 2 + num_vertices):
                vertex = lines[i].strip().split()
                if len(vertex) >= 3:
                    points.append([float(vertex[0]), float(vertex[1]), float(vertex[2])])
                    
            return np.array(points, dtype=np.float32)
            
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None
        
    def __len__(self):
        return len(self.point_clouds)
    
    def __getitem__(self, idx):
        point_cloud = self.point_clouds[idx].copy()
        label = self.labels[idx].copy()
        
        # Random sampling if more points than needed
        if point_cloud.shape[0] > self.num_points:
            choice = np.random.choice(point_cloud.shape[0], self.num_points, replace=False)
            point_cloud = point_cloud[choice, :]
            label = label[choice]
        
        # Padding if fewer points than needed
        elif point_cloud.shape[0] < self.num_points:
            padding = self.num_points - point_cloud.shape[0]
            indices = np.random.choice(point_cloud.shape[0], padding, replace=True)
            point_cloud = np.concatenate([point_cloud, point_cloud[indices]], axis=0)
            label = np.concatenate([label, label[indices]], axis=0)
        
        # Normalize point cloud to unit sphere
        point_cloud = self.normalize_point_cloud(point_cloud)
        
        return torch.FloatTensor(point_cloud).transpose(1, 0), torch.LongTensor(label)
    
    def normalize_point_cloud(self, points):
        """Normalize point cloud to unit sphere"""
        # Center the point cloud
        centroid = np.mean(points, axis=0)
        points = points - centroid
        
        # Scale to unit sphere
        max_dist = np.max(np.sqrt(np.sum(points**2, axis=1)))
        if max_dist > 0:
            points = points / max_dist
            
        return points

def train_model():
    # Hyperparameters
    batch_size = 64
    num_epochs = 500
    learning_rate = 0.001
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
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
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
            
            # Binary classification loss
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
        
        # Print epoch results
        train_acc = 100. * train_correct / train_total
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}')
        print(f'Train Accuracy: {train_acc:.2f}%')
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
            }, f'files/checkpoint_epoch_{epoch+1}.pt')
    
    # Save final model
    torch.save(model.state_dict(), 'pointnet_robot_removal.pt')
    print("Training completed!")

if __name__ == "__main__":
    train_model()