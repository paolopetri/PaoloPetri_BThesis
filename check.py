from PIL import Image

# Load the image
depth_image_path = 'TrainingData/depth_images/1.png'
depth_image = Image.open(depth_image_path)

risk_image_path = 'TrainingData/risk_images/1.png'
risk_image = Image.open(risk_image_path)

# Get image mode and size
print(f"Depth Image Mode: {depth_image.mode}")
print(f"Depth Image Size: {depth_image.size}")

print(f"Risk Image Mode: {risk_image.mode}")
print(f"Risk Image Size: {risk_image.size}")
