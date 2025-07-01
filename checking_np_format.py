import numpy as np

data = np.load('/home/internship/Documents/LabelledPoints_bin/centre_centre_lighting.npy', allow_pickle=True)
print(type(data), data.shape)
print(data[:5])  # Print first 5 rows