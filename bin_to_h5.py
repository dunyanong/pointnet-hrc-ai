import numpy as np
import h5py

# File paths
bin_file = "/home/internship/Documents/LabelledPoints_bin/centre_centre_lighting.bin"
h5_file = "/home/internship/Documents/pointnet-hrc-vision/LabelledPoints/centre_centre_lighting.h5"

# Load the .bin as float32 and reshape into (N, 4): x, y, z, label
data = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)

# Split into coordinates and labels
points = data[:, :3]  # shape (N, 3)
labels = data[:, 3].astype(np.uint8)  # shape (N,), values like 0 or 1

# Save to HDF5
with h5py.File(h5_file, 'w') as f:
    f.create_dataset('point', data=points, compression='gzip')   # (N, 3)
    f.create_dataset('label', data=labels, compression='gzip')  # (N,)

print(f"Saved to {h5_file}")