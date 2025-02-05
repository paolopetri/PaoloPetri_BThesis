import os
import cv2
import torch
import numpy as np
import re  # Import the re module for regular expressions

from depth_anything_v2.dpt import DepthAnythingV2

# Set the device for computation
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# Model configurations
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

# Choose the encoder
encoder = 'vitl'  # Options: 'vits', 'vitb', 'vitl', 'vitg'

# Initialize the model
model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
model = model.to(DEVICE).eval()

# Directories
input_folder = 'camera'
output_folder = 'depth'

# Create the output directory if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Get all image files in the input directory
image_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]

# Define a function to extract numerical value from filename
def extract_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else -1

# Sort the list of files numerically
image_files.sort(key=extract_number)

# First pass: find the maximum depth value across all images
max_depth_value = 0
total_images = len(image_files)

print("Starting first pass to find maximum depth value...")

for idx, image_file in enumerate(image_files, start=1):
    print(f"Processing {image_file} in first pass ({idx}/{total_images}).")
    image_path = os.path.join(input_folder, image_file)
    raw_img = cv2.imread(image_path)
    depth = model.infer_image(raw_img)  # HxW raw depth map in numpy
    current_max = depth.max()
    if current_max > max_depth_value:
        max_depth_value = current_max

print(f"First pass completed. Maximum depth value found: {max_depth_value}")

# Second pass: process and save normalized depth maps
print("Starting second pass to normalize and save depth maps...")

for idx, image_file in enumerate(image_files, start=1):
    print(f"Processing {image_file} in second pass ({idx}/{total_images}).")
    image_path = os.path.join(input_folder, image_file)
    raw_img = cv2.imread(image_path)
    depth = model.infer_image(raw_img)  # HxW raw depth map in numpy

    # Normalize the depth map using the maximum depth value
    depth_normalized = depth / max_depth_value  # Now values are between 0 and 1

    # Save the normalized depth map as a .npy file
    output_file = os.path.splitext(image_file)[0] + '.npy'  # Change extension to .npy
    output_path = os.path.join(output_folder, output_file)
    np.save(output_path, depth_normalized)

print("Second pass completed. All depth maps have been normalized and saved.")