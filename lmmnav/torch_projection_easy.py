import torch
import matplotlib.pyplot as plt

def project_points(points_3d, P):
    """
    Projects 3D points onto a 2D image plane using projection matrix P.

    Parameters:
    - points_3d: (N, 3) tensor of 3D points in camera frame.
    - P: (3, 4) Projection matrix tensor.

    Returns:
    - points_2d: (N, 2) tensor of 2D pixel coordinates.
    """
    # Convert to homogeneous coordinates by adding a column of ones
    num_points = points_3d.shape[0]
    ones = torch.ones((num_points, 1), dtype=points_3d.dtype, device=points_3d.device)
    points_homogeneous = torch.cat((points_3d, ones), dim=1)  # Shape: (N, 4)

    # Apply projection matrix
    points_projected = P @ points_homogeneous.T  # Shape: (3, N)

    # Avoid division by zero by replacing zeros in the z-component
    z = points_projected[2, :]
    z_safe = torch.where(z == 0, torch.tensor(1e-6, dtype=z.dtype, device=z.device), z)

    # Normalize by the third (depth) component
    points_projected /= z_safe

    # Extract x and y coordinates
    points_2d = points_projected[:2, :].T  # Shape: (N, 2)

    return points_2d

# Example usage
if __name__ == "__main__":
    # Define the projection matrix P as a PyTorch tensor
    P = torch.tensor([
        859.6767270584465, 0.0, 916.2221759734231, 0.0,
        0.0, 970.6574047708339, 644.6715355648768, 0.0,
        0.0, 0.0, 1.0, 0.0
    ], dtype=torch.float32).reshape(3, 4)

    # Define the rotation matrix R (identity matrix for no rotation)
    R = torch.eye(3, dtype=torch.float32)  # Shape: (3, 3)

    # Define the 3D trajectory points as a PyTorch tensor
    trajectory_3d_user = torch.tensor([
        [0.0000, 0.0000, 0.0000],
        [0.3045, 0.3,    1.0473],
        [0.6455, 0.3,    2.0817],
        [1.0379, 0.3,    3.1124],
        [1.5833, 0.3,    4.0607],
        [2.4193, 0.3,    4.8332],
        [3.4528, 0.3,    5.4827],
        [4.6907, 0.3,    6.0076],
        [5.9715, 0.3,    6.4768],
        [7.0975, 0.3,    6.9660],
        [8.1853, 0.3,    7.4529]
    ], dtype=torch.float32)  # Shape: (11, 3)

    # Rotate points to align with the conventional camera coordinate system
    trajectory_3d_standard = (R @ trajectory_3d_user.T).T  # Shape: (11, 3)

    # Define the image path
    image_path = 'TrainingData/Important/camera/220.png'
    
    # Load the image using Matplotlib (remains a NumPy array)
    try:
        image = plt.imread(image_path)
        print(f"Image loaded successfully from {image_path}.")
    except FileNotFoundError:
        print(f"Image not found at path: {image_path}")
        # Create a blank white image for testing purposes
        image = torch.ones((1280, 1920, 3), dtype=torch.float32).numpy()  # Convert to NumPy for plotting
        print("Using a blank white image for plotting.")

    # Filter out points that are behind the camera (z <= 0)
    valid_indices = trajectory_3d_standard[:, 2] > 0
    trajectory_3d_standard_valid = trajectory_3d_standard[valid_indices]

    # Project the valid 3D points to 2D pixel coordinates
    trajectory_2d_valid = project_points(trajectory_3d_standard_valid, P)

    # Convert the projected 2D points to NumPy for plotting
    trajectory_2d_valid_np = trajectory_2d_valid.cpu().numpy()

    # Debugging: Print the projected 2D points
    print("Mirrored Projected 2D Points (Pixel Coordinates):")
    print(trajectory_2d_valid_np)

    # Plot the projected trajectory
    plt.figure(figsize=(12, 8))
    
    # Display the image
    plt.imshow(image)
    
    # Plot the trajectory points
    plt.plot(trajectory_2d_valid_np[:, 0], trajectory_2d_valid_np[:, 1], 'ro-', label='Trajectory')
    
    # Annotate each point with its index
    for idx, (x, y) in enumerate(trajectory_2d_valid_np):
        plt.annotate(f'{idx}', (x, y), textcoords="offset points", xytext=(0, 10), ha='center')
    
    plt.title('Projected Trajectory onto Image')
    plt.xlabel('Pixel X')
    plt.ylabel('Pixel Y')
    
    # Ensure the y-axis is correctly oriented
    plt.xlim(0, image.shape[1])
    plt.ylim(image.shape[0], 0)  # Invert y-axis for correct orientation
    plt.legend()
    plt.grid(True)
    
    # Save the plot to a file
    plt.savefig('projected_trajectory.png', bbox_inches='tight', dpi=300)
    
    # Display the plot
    plt.show()
