#!/usr/bin/env python3
"""
Convert the trained PointNet model to TorchScript for C++ inference
"""
import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path to import model
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model.model import PointNetSegmentation

def convert_model_to_torchscript():
    """Convert the trained PyTorch model to TorchScript"""
    
    # Model parameters
    num_classes = 2  # Binary classification (background=0, laptop=1)
    
    # Get the project root directory (two levels up from this script)
    project_root = os.path.join(os.path.dirname(__file__), '..', '..')
    model_path = os.path.join(project_root, 'DeployModel', 'laptop_classifier.pt')
    output_path = os.path.join(project_root, 'DeployModel', 'laptop_classifier_traced.pt')
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found!")
        return False
    
    # Load the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model instance
    model = PointNetSegmentation(num_classes=num_classes)
    
    # Load trained weights
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Loaded model from checkpoint")
        else:
            model.load_state_dict(checkpoint)
            print("Loaded model state dict directly")
            
    except Exception as e:
        print(f"Error loading model: {e}")
        return False
    
    # Set model to evaluation mode
    model.eval()
    model.to(device)
    
    # Create example input (batch_size=1, features=3, num_points=1024)
    example_input = torch.randn(1, 3, 1024).to(device)
    
    print("Converting model to TorchScript...")
    
    try:
        # Trace the model
        with torch.no_grad():
            traced_model = torch.jit.trace(model, example_input)
        
        # Save the traced model
        traced_model.save(output_path)
        print(f"Model successfully converted and saved to: {output_path}")
        
        # Test the traced model
        print("Testing traced model...")
        with torch.no_grad():
            original_output, _ = model(example_input)
            traced_output = traced_model(example_input)
            
            # Check if outputs are close (traced model might return tuple)
            if isinstance(traced_output, tuple):
                traced_output = traced_output[0]
                
            if torch.allclose(original_output, traced_output, atol=1e-5):
                print("✓ Traced model produces identical outputs")
            else:
                print("⚠ Warning: Traced model outputs differ from original")
                print(f"Max difference: {torch.max(torch.abs(original_output - traced_output))}")
        
        return True
        
    except Exception as e:
        print(f"Error during tracing: {e}")
        return False

if __name__ == "__main__":
    success = convert_model_to_torchscript()
    if success:
        print("Model conversion completed successfully!")
    else:
        print("Model conversion failed!")
