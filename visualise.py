import open3d as o3d
import numpy as np

def save_labeled_pcd(points, labels, output_path):
    # Normalize labels to color map (0-1 range for RGB)
    max_label = labels.max() + 1
    cmap = plt.get_cmap('tab20')
    colors = cmap(labels / max_label)[:, :3]  # Get RGB, ignore alpha

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])  # XYZ
    pcd.colors = o3d.utility.Vector3dVector(colors)         # RGB (0–1)

    o3d.io.write_point_cloud(output_path, pcd)
    print(f"Saved: {output_path}")

points, labels = process_area(input_dir)
save_labeled_pcd(points, labels, "conference_room_labeled.pcd")