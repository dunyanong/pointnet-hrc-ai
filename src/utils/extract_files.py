import numpy as np
import os
import glob

def read_off_file(file_path):
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

def normalize_point_cloud(points):
    """Normalize point cloud to unit sphere"""
    # Center the point cloud
    centroid = np.mean(points, axis=0)
    points = points - centroid
    
    # Scale to unit sphere
    max_dist = np.max(np.sqrt(np.sum(points**2, axis=1)))
    if max_dist > 0:
        points = points / max_dist
        
    return points

def create_binary_labels(points):
    """Create binary labels for laptop segmentation"""
    num_points = len(points)
    
    # Simple heuristic: assume center points are more likely to be laptop
    center = np.mean(points, axis=0)
    distances = np.linalg.norm(points - center, axis=1)
    threshold = np.percentile(distances, 60)  # 60% of points as laptop
    
    labels = np.where(distances <= threshold, 1, 0).astype(np.int64)  # 1=laptop, 0=background
    return labels

def load_point_cloud_data(data_path, split='train', categories=None):
    """Load point cloud data from ModelNet40 .off files"""
    if categories is None:
        categories = ["laptop"]
    
    point_clouds = []
    labels = []
    
    print(f"Binary classification for laptop detection")
    
    for category in categories:
        category_path = os.path.join(data_path, category, split)
        if not os.path.exists(category_path):
            continue
            
        off_files = glob.glob(os.path.join(category_path, "*.off"))
        
        for off_file in off_files:
            try:
                points = read_off_file(off_file)
                if points is not None and len(points) > 0:
                    point_clouds.append(points)
                    # Create binary segmentation labels
                    point_labels = create_binary_labels(points)
                    labels.append(point_labels)
            except Exception as e:
                print(f"Error loading {off_file}: {e}")
                continue
                
    print(f"Loaded {len(point_clouds)} point clouds for {split}")
    return point_clouds, labels

def resample_point_cloud(point_cloud, label, num_points, current_epoch=0, adaptive_alpha=True):
    """Resample point cloud to target number of points"""
    # Random sampling if more points than needed
    if point_cloud.shape[0] > num_points:
        choice = np.random.choice(point_cloud.shape[0], num_points, replace=False)
        point_cloud = point_cloud[choice, :]
        label = label[choice]
    
    # Interpolation-based upsampling if fewer points than needed
    elif point_cloud.shape[0] < num_points:
        padding = num_points - point_cloud.shape[0]
        
        # Create interpolated points between existing points
        for _ in range(padding):
            # Pick two random points
            idx1, idx2 = np.random.choice(point_cloud.shape[0], 2, replace=False)
            
            # Use adaptive alpha for interpolation
            alpha = get_adaptive_alpha(current_epoch, adaptive_alpha)
            new_point = alpha * point_cloud[idx1] + (1 - alpha) * point_cloud[idx2]
            
            # Add adaptive noise based on training progress
            noise_scale = 0.01 * (1.0 - min(current_epoch / 200.0, 0.8))
            noise = np.random.normal(0, noise_scale, 3)
            new_point += noise
            
            # Interpolate label (take majority)
            new_label = label[idx1] if np.random.random() < alpha else label[idx2]
            
            # Add to arrays
            point_cloud = np.vstack([point_cloud, new_point])
            label = np.append(label, new_label)
    
    return point_cloud, label

def get_adaptive_alpha(current_epoch=0, adaptive_alpha=True):
    """Calculate adaptive alpha based on training progress"""
    if not adaptive_alpha:
        return np.random.uniform(0.3, 0.7)
    
    # Start with more conservative interpolation, become more aggressive over time
    base_alpha = 0.5
    epoch_factor = min(current_epoch / 100.0, 1.0)  # Normalize over 100 epochs
    
    # Reduce interpolation range as training progresses
    alpha_range = 0.4 * (1.0 - epoch_factor * 0.5)  # From 0.4 to 0.2 range
    alpha = base_alpha + np.random.uniform(-alpha_range/2, alpha_range/2)
    
    return np.clip(alpha, 0.1, 0.9)

def preprocess_single_point_cloud(points, num_points, normalize=True):
    """Preprocess a single point cloud for inference"""
    # Resample to target number of points
    if points.shape[0] > num_points:
        choice = np.random.choice(points.shape[0], num_points, replace=False)
        points = points[choice, :]
    elif points.shape[0] < num_points:
        padding = num_points - points.shape[0]
        indices = np.random.choice(points.shape[0], padding, replace=True)
        points = np.concatenate([points, points[indices]], axis=0)
    
    # Normalize if requested
    if normalize:
        points = normalize_point_cloud(points)
    
    return points
