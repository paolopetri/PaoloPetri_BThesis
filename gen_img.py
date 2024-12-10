import os
import numpy as np
from PIL import Image

# Paths to the .npy files
depth_image_path = 'TrainingData/depth_images/1.npy'
risk_image_path = 'TrainingData/risk_images/1.npy'

# Create the presentation folder if it doesn't exist
os.makedirs('presentation', exist_ok=True)

# Load the NumPy arrays
depth_array = np.load(depth_image_path)  # Normalized [0,1]
risk_array = np.load(risk_image_path)    # Normalized [0,1]

# Convert arrays to 8-bit
depth_uint8 = (depth_array * 255).astype(np.uint8)
risk_uint8 = (risk_array * 255).astype(np.uint8)

# Create PIL images
depth_image = Image.fromarray(depth_uint8, mode='L')
risk_image = Image.fromarray(risk_uint8, mode='RGB')

# Save images to the presentation folder
depth_image.save('presentation/depth_output.png')
risk_image.save('presentation/risk_output.png')
