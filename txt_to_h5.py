import os
import numpy as np
import h5py
import open3d as o3d
import matplotlib.pyplot as plt

def read_txt_file(file_path):
    return np.loadtxt(file_path)

def process_area(area_path):
    points = []
    labels = []
    annotation_path = os.path.join(area_path, "Annotations")

    # Generate unique class names only
    class_names = sorted(set(f.split('_')[0] for f in os.listdir(annotation_path)))
    label_map = {cls: idx for idx, cls in enumerate(class_names)}
    print("Label map:", label_map)

    for file_name in os.listdir(annotation_path):
        file_path = os.path.join(annotation_path, file_name)
        class_name = file_name.split('_')[0]
        label = label_map[class_name]
        data = read_txt_file(file_path)
        points.append(data[:, :6])
        labels.append(np.full((data.shape[0],), label))

    return np.vstack(points), np.hstack(labels)

def save_to_h5(output_path, points, labels):
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('data', data=points)
        f.create_dataset('label', data=labels)

def convert_s3dis_to_h5(area_folder_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    points, labels = process_area(area_folder_path)
    area_name = os.path.basename(area_folder_path.rstrip('/'))
    output_path = os.path.join(output_dir, f"{area_name}.h5")
    save_to_h5(output_path, points, labels)
    print(f"Saved {output_path}")

# ✅ Use folder, not .txt file
input_dir = "/Users/ongdunyan/Downloads/LocalCodes/pointnet_hrc_vision/data/raw/Stanford3dDataset_v1.2_Aligned_Version/Area_1/conferenceRoom_1"
output_dir = "/Users/ongdunyan/Downloads/LocalCodes/pointnet_hrc_vision/data/processed"

convert_s3dis_to_h5(input_dir, output_dir)

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