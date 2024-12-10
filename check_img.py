import numpy as np

# Paths to the .npy files
depth_image_path = 'TrainingData/depth_images/1.npy'
risk_image_path = 'TrainingData/risk_images/1.npy'

# Load the NumPy arrays
depth_array = np.load(depth_image_path)
risk_array = np.load(risk_image_path)

# Get array shapes
print(f"Depth Image Shape: {depth_array.shape}")
print(f"Risk Image Shape: {risk_array.shape}")

# Optionally, print the arrays (commented out if arrays are large)
# print(f"Depth Array:\n{depth_array}")
# print(f"Risk Array:\n{risk_array}")

# Find min and max values for depth image
depth_min = depth_array.min()
depth_max = depth_array.max()
print(f"Depth Image Min Pixel Value: {depth_min}")
print(f"Depth Image Max Pixel Value: {depth_max}")

# Find min and max values for risk image
risk_min = risk_array.min()
risk_max = risk_array.max()
print(f"Risk Image Min Pixel Value: {risk_min}")
print(f"Risk Image Max Pixel Value: {risk_max}")
