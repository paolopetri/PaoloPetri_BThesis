import os
import imageio
from PIL import Image
import numpy as np  # Import numpy

def create_compressed_gif(data_root, snippet_indices=range(1356, 1388), output_name="camera_sequence_compressed.gif", fps=1, resize_factor=0.5, colors=128):
    """
    Creates a compressed GIF by resizing images and reducing the color palette.

    Parameters:
    - data_root (str): Path to the root directory containing the 'camera' folder.
    - snippet_indices (iterable): Indices of images to include in the GIF.
    - output_name (str): Name of the output GIF file.
    - fps (int): Frames per second for the GIF.
    - resize_factor (float): Factor by which to resize images (e.g., 0.5 for half size).
    - colors (int): Number of colors in the GIF palette (max 256).
    """
    camera_dir = os.path.join(data_root, "camera")
    frames = []
    processed_frames = 0  # Counter for successfully processed frames

    for i in snippet_indices:
        img_path = os.path.join(camera_dir, f"{i}.png")
        try:
            with Image.open(img_path) as img:
                # Resize the image
                new_size = (int(img.width * resize_factor), int(img.height * resize_factor))
                img = img.resize(new_size, Image.ANTIALIAS)
                
                # Convert to palette-based image to reduce colors
                img = img.convert('P', palette=Image.ADAPTIVE, colors=colors)
                
                # Convert back to RGB for imageio compatibility
                img = img.convert('RGB')
                
                # Convert PIL Image to NumPy array
                frame = np.array(img)
                
                frames.append(frame)
                processed_frames += 1
        except FileNotFoundError:
            print(f"Warning: {img_path} not found. Skipping this frame.")
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    if processed_frames == 0:
        print("No frames were processed successfully. Please check your image paths and formats.")
        return

    # Ensure the output directory exists
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_name)

    try:
        # Save the GIF
        imageio.mimsave(output_path, frames, fps=fps, loop=0)
        print(f"Saved compressed GIF to: {output_path}")
    except Exception as e:
        print(f"Failed to save GIF: {e}")

if __name__ == "__main__":
    data_root = "TrainingData/Important"   # example root path
    create_compressed_gif(
        data_root,
        snippet_indices=range(1368, 1400),
        output_name="camera_sequence_compressed.gif",
        fps=1,                # Adjust FPS as needed
        resize_factor=0.5,    # Resize to 50% of original size
        colors=128            # Reduce to 128 colors
    )
