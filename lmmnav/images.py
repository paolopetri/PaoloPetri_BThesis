import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

def ensure_three_channels(image):
    """
    Ensures the image has 3 channels. If the input is grayscale (2D), 
    it repeats the single channel three times along the channel dimension.
    """
    if image.ndim == 2:  # grayscale image
        image = np.stack([image] * 3, axis=-1)
    elif image.shape[-1] != 3:
        raise ValueError("Image does not have 1 or 3 channels.")
    return image

def resize_with_opencv(image, size):
    """
    Resizes the image using OpenCV.
    
    Args:
        image (np.ndarray): Image array of shape (H, W, C).
        size (tuple): Desired output size (height, width).

    Returns:
        np.ndarray: Resized image array.
    """
    # OpenCV expects size as (width, height)
    width = size[1]
    height = size[0]
    # cv2.resize expects the input shape as (width, height) 
    resized_image = cv2.resize(image, dsize=(width, height), interpolation=cv2.INTER_LINEAR)
    return resized_image

# Paths to your .npy files; adjust these as necessary
depth_path = 'TrainingData/Important/depth_images/220.npy'  # example: shape (1280, 1920)
risk_path = 'TrainingData/Important/risk_images/874.npy'    # example: shape (427, 640, 3)

# Load images from .npy files
depth_image = np.load(depth_path)
risk_image = np.load(risk_path)

print(f"depth_image shape: {depth_image.shape}")

depth_min = np.min(depth_image)
depth_max = np.max(depth_image)
depth_normalized = 255 * (depth_image - depth_min) / (depth_max - depth_min)
depth_normalized = depth_normalized.astype(np.uint8)

# Convert to PIL Image and save
depth_pil = Image.fromarray(depth_normalized, mode='L')  # 'L' mode for (8-bit pixels, black and white)
depth_pil.save('depth_image.png')  # Choose desired format

# --- Saving Risk Image (RGB) ---

# Since risk_image values are between 0 and 1, scale to 0-255
risk_scaled = (risk_image * 255).clip(0, 255).astype(np.uint8)

risk_image_rgb = risk_scaled[..., ::-1]  # Convert BGR to RGB

# Convert to PIL Image and save
risk_pil = Image.fromarray(risk_scaled, mode='RGB')
risk_pil.save('risk_image.png')  # Choose desired format

print("Images saved successfully using Pillow.")

# Ensure depth_image has three channels
depth_image = ensure_three_channels(depth_image)

# Resize both images to (360, 640)
target_size = (360, 640)  # (height, width)
depth_resized = resize_with_opencv(depth_image, target_size)
risk_resized = resize_with_opencv(risk_image, target_size)

# Visualization using Matplotlib
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(cv2.cvtColor(depth_resized, cv2.COLOR_BGR2RGB))
axes[0].set_title('Resized Depth Image')
axes[0].axis('off')

axes[1].imshow(cv2.cvtColor(risk_resized, cv2.COLOR_BGR2RGB))
axes[1].set_title('Resized Risk Image')
axes[1].axis('off')

plt.tight_layout()
plt.show()
