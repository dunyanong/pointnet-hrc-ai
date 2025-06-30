import numpy as np
import h5py
import os

# ====== CONFIGURATION ======
bin_path = "/home/internship/Documents/LabelledPoints_bin/centre_centre_lighting.bin"  # Must be a file
h5_path = "/home/internship/Documents/pointnet-hrc-vision/LabelledPoints.h5"   # Full path to .h5 file
dtype = np.float32                    # Adjust this if your .bin file uses a different type
shape = None                 # Adjust this to your actual data shape, or set to None
dataset_name = "data"                 # Internal dataset name in HDF5
# ===========================

def bin_to_h5(bin_path, h5_path, dtype=np.float32, shape=None, dataset_name="data"):
    if not os.path.exists(bin_path):
        raise FileNotFoundError(f"Binary file not found: {bin_path}")

    data = np.fromfile(bin_path, dtype=dtype)

    if shape:
        try:
            data = data.reshape(shape)
        except ValueError:
            raise ValueError(f"Cannot reshape array of size {data.size} to shape {shape}")

    with h5py.File(h5_path, 'w') as h5f:
        h5f.create_dataset(dataset_name, data=data)

    print(f"Successfully converted:\n{bin_path} → {h5_path}")

if __name__ == "__main__":
    bin_to_h5(bin_path, h5_path, dtype=dtype, shape=shape, dataset_name=dataset_name)
