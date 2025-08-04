import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from model.model import PointNetSegmentation, feature_transform_regularizer
from train import PointCloudDataset
from src.Utils.extract_files import read_off_file, normalize_point_cloud, preprocess_single_point_cloud

class ModelValidator:
    """Class for validating the PointNet model"""
    
    def __init__(self, model_path, data_path, num_points=1024, batch_size=32):
        self.model_path = model_path
        self.data_path = data_path
        self.num_points = num_points
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load dataset
        self.dataset = PointCloudDataset(data_path, num_points=num_points, split='test')
        # Binary classification only
        self.num_classes = 2
        
        # Load model
        self.model = self.load_model()
        
        # Create data loader
        self.test_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False)
        
    def load_model(self):
        """Load the trained model"""
        model = PointNetSegmentation(num_classes=self.num_classes).to(self.device)
        
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
            else:
                model.load_state_dict(checkpoint)
                print("Loaded model state dict")
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
            
        model.eval()
        return model
    
    def validate(self):
        """Perform validation on test dataset"""
        if self.model is None:
            print("Model not loaded properly")
            return None
            
        print("Starting binary validation...")
        print(f"Device: {self.device}")
        print(f"Test dataset size: {len(self.dataset)}")
        print(f"Binary classification: background=0, laptop=1")
        
        all_predictions = []
        all_targets = [] 
        total_loss = 0.0
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                output, feature_transform = self.model(data)
                
                # Reshape target to match output
                target_flat = target.view(-1)
                
                # Calculate loss
                loss = criterion(output, target_flat)
                reg_loss = feature_transform_regularizer(feature_transform)
                total_loss += (loss + 0.001 * reg_loss).item()
                
                # Get predictions
                _, predicted = torch.max(output.data, 1)
                
                # Store predictions and targets
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target_flat.cpu().numpy())
                
                if batch_idx % 10 == 0:
                    print(f"Processed batch {batch_idx}/{len(self.test_loader)}")
        
        # Calculate binary metrics
        avg_loss = total_loss / len(self.test_loader)
        accuracy = accuracy_score(all_targets, all_predictions)
        precision = precision_score(all_targets, all_predictions, pos_label=1, zero_division=0)
        recall = recall_score(all_targets, all_predictions, pos_label=1, zero_division=0)
        f1 = f1_score(all_targets, all_predictions, pos_label=1, zero_division=0)
        
        # Calculate IoU for laptop class
        conf_matrix = confusion_matrix(all_targets, all_predictions)
        laptop_iou = self.calculate_iou(conf_matrix, class_idx=1)
        
        print("\n" + "="*60)
        print("BINARY CLASSIFICATION VALIDATION RESULTS")
        print("="*60)
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Laptop Precision: {precision:.4f}")
        print(f"Laptop Recall: {recall:.4f}")
        print(f"Laptop F1-Score: {f1:.4f}")
        print(f"Laptop IoU: {laptop_iou:.4f}")
        print("="*60)
        
        # Show binary statistics
        self.show_binary_stats(all_targets, all_predictions, conf_matrix)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'laptop_iou': laptop_iou,
            'predictions': all_predictions,
            'targets': all_targets
        }
    
    def calculate_iou(self, conf_matrix, class_idx):
        """Calculate IoU for a specific class"""
        if class_idx >= conf_matrix.shape[0]:
            return 0.0
        
        tp = conf_matrix[class_idx, class_idx]
        fp = conf_matrix[:, class_idx].sum() - tp
        fn = conf_matrix[class_idx, :].sum() - tp
        
        if tp + fp + fn == 0:
            return 0.0
        
        return tp / (tp + fp + fn)
    
    def show_binary_stats(self, targets, predictions, conf_matrix):
        """Show binary classification statistics"""
        print("\nBINARY STATISTICS:")
        print("-" * 60)
        
        # Class distribution
        target_counts = np.bincount(targets, minlength=2)
        pred_counts = np.bincount(predictions, minlength=2)
        
        print(f"Ground Truth - Background: {target_counts[0]}, Laptop: {target_counts[1]}")
        print(f"Predictions - Background: {pred_counts[0]}, Laptop: {pred_counts[1]}")
        
        print("\nConfusion Matrix:")
        print("               Predicted")
        print("             BG    Laptop")
        print(f"Actual BG    {conf_matrix[0,0]:<6} {conf_matrix[0,1]:<6}")
        print(f"     Laptop  {conf_matrix[1,0]:<6} {conf_matrix[1,1]:<6}")
        
    def validate_single_point_cloud(self, point_cloud_path):
        """Validate a single point cloud file"""
        if self.model is None:
            print("Model not loaded properly")
            return None
            
        # Load and preprocess the point cloud
        points = read_off_file(point_cloud_path)
        if points is None:
            print(f"Failed to load point cloud from {point_cloud_path}")
            return None
            
        # Preprocess using utility function
        points = preprocess_single_point_cloud(points, self.num_points, normalize=True)
        
        # Convert to tensor
        point_tensor = torch.FloatTensor(points).transpose(1, 0).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            output, _ = self.model(point_tensor)
            probabilities = F.softmax(output, dim=1)
            _, predicted_labels = torch.max(output, 1)
            
        # Convert back to point-wise predictions
        predicted_labels = predicted_labels.cpu().numpy().reshape(self.num_points)
        probabilities = probabilities.cpu().numpy().reshape(self.num_points, self.num_classes)
        
        print(f"\nBinary Segmentation Results for: {point_cloud_path}")
        print("-" * 60)
        
        background_points = np.sum(predicted_labels == 0)
        laptop_points = np.sum(predicted_labels == 1)
        
        print(f"Background points: {background_points} ({background_points/self.num_points*100:.2f}%)")
        print(f"Laptop points: {laptop_points} ({laptop_points/self.num_points*100:.2f}%)")
            
        return {
            'predicted_labels': predicted_labels,
            'probabilities': probabilities,
            'points': points,
            'background_points': background_points,
            'laptop_points': laptop_points
        }

def main():
    """Main validation function"""
    # Configuration
    model_path = 'DeployModel/laptop_classifier.pt'
    data_path = 'Dataset/ModelNet40'
    
    # Create validator
    validator = ModelValidator(
        model_path=model_path,
        data_path=data_path,
        num_points=1024,
        batch_size=32
    )
    
    # Run validation
    results = validator.validate()
    
    if results:
        print(f"\nValidation completed successfully!")
        print(f"Final Accuracy: {results['accuracy']*100:.2f}%")
        
        # Optional: Validate a single point cloud
        # single_result = validator.validate_single_point_cloud('path/to/single/file.off')

if __name__ == "__main__":
    main()