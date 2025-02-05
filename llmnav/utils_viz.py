"""
utils_viz.py

This module contains utility functions for visualization.

Author: [Paolo Petri]
Date: [07.02.2025]
"""
import os
import torch
import matplotlib.pyplot as plt
import imageio
import io
from PIL import Image
import math

def rotation_x(theta_degrees: float, device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """
    Generate a 3x3 rotation matrix around the x-axis, given an angle in degrees.

    Args:
        theta_degrees (float): The rotation angle around the x-axis in degrees.
        device (torch.device, optional): The device on which the resulting tensor
            will be allocated. Defaults to CPU.

    Returns:
        torch.Tensor: A 3x3 rotation matrix representing the specified rotation
        around the x-axis.
    """
    theta = math.radians(theta_degrees)
    return torch.tensor([
        [1, 0,           0          ],
        [0, math.cos(theta), -math.sin(theta)],
        [0, math.sin(theta),  math.cos(theta)]
    ], dtype=torch.float32, device=device)


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

def plot_single_waypoints_on_rgb(waypoints_batch: torch.tensor,
                                goal_positions_batch: torch.tensor,
                                idx: torch.tensor,
                                model_name: str,
                                frame: str,
                                show: bool = False,
                                save: bool = True, 
                                output_dir: str = 'output/image/single'
) -> list[plt.Figure]:
    """
    Projects and plots batched 3D waypoints and goal positions onto corresponding 2D images.

    Parameters:
    - waypoints_batch: (B, N, 3) tensor of 3D waypoints, where B is batch size.
    - goal_positions_batch: (B, 3) tensor of 3D goal positions, one per batch.
    - model_name: str, name of the model to include in saved filenames.
    - idx: list or tensor of length B, specifying image indices.
    - show: bool, whether to display the plots.
    - save: bool, whether to save the plots.
    - output_dir: str, directory to save the plots.

    Returns:
    - figures: list of Matplotlib Figure objects corresponding to each batch.

    Note:
    - The projection matrix P is hardcoded for the LLMNav camera.
    - The adaptations had to been made to allow for visualization in the iPlanner frame.
    """
    # Define the projection matrix P as a PyTorch tensor
    device = waypoints_batch.device
    P = torch.tensor([
        859.6767270584465, 0.0, 916.2221759734231, 0.0,
        0.0, 970.6574047708339, 644.6715355648768, 0.0,
        0.0, 0.0, 1.0, 0.0
    ], dtype=torch.float32).reshape(3, 4).to(device)

    print(f"Projection matrix P: {P}")

    # Define the rotation matrix R based on the model name
    if frame == "LLMNav":
        R = torch.eye(3, dtype=torch.float32).to(device)  # Shape (3,3)
    elif frame == "iPlanner":
        R = torch.tensor([
            [0, 1, 0],
            [0, 0, -1],
            [-1, 0, 0]
        ], dtype=torch.float32).to(device)  # Shape (3,3)
    else:
        raise ValueError(f"Unknown model_name: {model_name}. Please use 'LLMNav' or 'iPlanner'.")
    
    R_tilt = rotation_x(3, device=device) # accounting for camera tilt

    R_tilt_ip = rotation_x(4, device=device) # accounting for additional iPlanner tilt camera tilt
   

    # Ensure idx is a list
    if isinstance(idx, torch.Tensor):
        idx = idx.tolist()
    if len(waypoints_batch) != len(idx):
        raise ValueError("The number of waypoints batches must match the number of indices.")
    if len(waypoints_batch) != len(goal_positions_batch):
        raise ValueError("The number of waypoints batches must match the number of goal positions.")

    # Create the output directory if saving is enabled
    if save:
        os.makedirs(output_dir, exist_ok=True)

    figures = []  # List to store Figure objects

    # Iterate over each batch of waypoints, goal position, and corresponding index
    for batch_idx, (waypoints, goal_position, current_idx) in enumerate(zip(waypoints_batch, goal_positions_batch, idx)):
        # Rotate points to align with the conventional camera coordinate system
        trajectory_3d_standard = (R @ waypoints.T).T  # Shape: (N, 3)
        trajectory_3d_standard[0, 1] = 0.3 # set [::1] to 0.3 for iPlanner
        if model_name == "iPlanner":
            trajectory_3d_standard[:,1] = 0.3
            trajectory_3d_standard = (R_tilt_ip @ trajectory_3d_standard.T).T
        elif model_name == "LLMNav":
            trajectory_3d_standard = (R_tilt @ trajectory_3d_standard.T).T # R_tilt_ip for iPlanner

        # Rotate goal position
        goal_3d_standard = (R @ goal_position.unsqueeze(1)).squeeze(1)  # Shape: (3,)
       
        goal_3d_standard = (R_tilt @ goal_3d_standard.unsqueeze(1)).squeeze(1)
        if model_name == "iPlanner":
            trajectory_3d_standard[-1, 1] = goal_3d_standard[1] # for iPlanner, because its height estimation does not work!

        # Define the image path based on the index
        image_path = f'TrainingData/Important/camera/{current_idx}.png'
        
        # Load the image using Matplotlib (remains a NumPy array)
        try:
            image = plt.imread(image_path)
            print(f"[Batch {batch_idx}] Image loaded successfully from {image_path}.")
        except FileNotFoundError:
            print(f"[Batch {batch_idx}] Image not found at path: {image_path}")
            # Create a blank white image for testing purposes
            image = torch.ones((1280, 1920, 3), dtype=torch.float32).numpy()  # Convert to NumPy for plotting
            print(f"[Batch {batch_idx}] Using a blank white image for plotting.")

        print(f"Image shape: {image.shape}") # output: (1280, 1920, 3)


        # Project the valid 3D waypoints to 2D pixel coordinates
        trajectory_2d_valid = project_points(trajectory_3d_standard, P)

        # Project the goal position to 2D pixel coordinates
        goal_2d = project_points(goal_3d_standard.unsqueeze(0), P) 
        goal_2d_np = goal_2d.cpu().numpy()[0] 
        trajectory_2d_valid_np = trajectory_2d_valid.cpu().numpy()
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(image)
        
        # Plot the trajectory points
        ax.plot(trajectory_2d_valid_np[:, 0], trajectory_2d_valid_np[:, 1], 'ro-', label='Trajectory')
        
        # Plot the goal position
        ax.plot(goal_2d_np[0], goal_2d_np[1], 'bs', label='Goal Position')  # Blue square
        
        # Annotate each trajectory point with its index
        for idx_pt, (x, y) in enumerate(trajectory_2d_valid_np):
            ax.annotate(f'{idx_pt}', (x, y), textcoords="offset points", xytext=(0, 10), ha='center')
        
        
        ax.set_title(f'{model_name}: Projected Trajectory and Goal onto RGB Image')
        
        # Ensure the y-axis is correctly oriented
        ax.set_xlim(0, image.shape[1])
        ax.set_ylim(image.shape[0], 0)  # Invert y-axis for correct orientation
        ax.axis('off')
        ax.legend()
        
        # Save the plot if required
        if save:
            save_filename = f"{model_name}_idx{current_idx}.png"
            save_path = os.path.join(output_dir, save_filename)
            fig.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"[Batch {batch_idx}] Plot saved to {save_path}.")
        
        # Show the plot if required
        if show:
            plt.show()
        else:
            plt.close(fig)  # Close the figure to free memory

        # Append the figure to the list
        figures.append(fig)

    return figures


def plot_waypoints_on_depth_risk(
    waypoints_batch: torch.tensor,
    goal_positions_batch: torch.tensor,
    depth_images_batch: torch.tensor,
    risk_images_batch: torch.tensor,
    idx: torch.tensor,
    model_name: str,
    frame: str,
    show: bool = False,
    save: bool = True,
    output_dir: str = 'output/image/depth_risk'
    )-> list[plt.Figure]:
    """
    Projects and plots batched 3D waypoints and goal positions onto corresponding depth and risk images.

    Parameters:
    - waypoints_batch: (B, N, 3) tensor of 3D waypoints, where B is batch size.
    - goal_positions_batch: (B, 3) tensor of 3D goal positions, one per batch.
    - depth_images_batch: (B, 360, 640, 3) tensor or NumPy array of depth images normalized between 0 and 1.
    - risk_images_batch: (B, 360, 640, 3) tensor or NumPy array of risk images normalized between 0 and 1.
    - idx: list or tensor of length B, specifying image indices.
    - model_name: str, name of the model to include in saved filenames.
    - frame: str, either "LLMNav" or "iPlanner", specifying the frame of reference.
    - show: bool, whether to display the plots.
    - save: bool, whether to save the plots.
    - output_dir: str, directory to save the plots.

    Returns:
    - figures: list of Matplotlib Figure objects corresponding to each batch.
    """
    # Define the projection matrix P as a PyTorch tensor
    device = waypoints_batch.device
    P = torch.tensor([
        286.5589, 0.0, 305.4074, 0.0,
        0.0, 272.9974, 181.3139, 0.0,
        0.0, 0.0, 1.0, 0.0
    ], dtype=torch.float32).reshape(3, 4).to(device)

    # Define the rotation matrix R based on the frame
    if frame == "LLMNav":
        R = torch.eye(3, dtype=torch.float32).to(device)  # Shape (3,3)
    elif frame == "iPlanner":
        R = torch.tensor([
            [0, 1, 0],
            [0, 0, -1],
            [-1, 0, 0]
        ], dtype=torch.float32).to(device)  # Shape (3,3)
    else:
        raise ValueError(f"Unknown frame: {frame}. Please use 'LLMNav' or 'iPlanner'.")
    
    R_tilt = rotation_x(3, device=device) # accounting for camera tilt
    R_tilt_ip = rotation_x(4, device=device) # accounting for additional iPlanner tilt camera tilt
    
     # Ensure idx is a list
    if isinstance(idx, torch.Tensor):
        idx = idx.tolist()
    if len(waypoints_batch) != len(idx):
        raise ValueError("The number of waypoints batches must match the number of indices.")
    if len(waypoints_batch) != len(goal_positions_batch):
        raise ValueError("The number of waypoints batches must match the number of goal positions.")
    if len(waypoints_batch) != len(depth_images_batch) or len(waypoints_batch) != len(risk_images_batch):
        raise ValueError("The number of waypoints batches must match the number of depth and risk images.")

    # Create the output directory if saving is enabled
    if save:
        os.makedirs(output_dir, exist_ok=True)

    figures = []  # List to store Figure objects

    # Iterate over each batch of waypoints, goal position, depth image, risk image, and corresponding index
    for batch_idx, (waypoints, goal_position, depth_image, risk_image, current_idx) in enumerate(zip(
        waypoints_batch, goal_positions_batch, depth_images_batch, risk_images_batch, idx
    )):
        # Rotate points to align with the conventional camera coordinate system
        trajectory_3d_standard = (R @ waypoints.T).T  # Shape: (N, 3)
        trajectory_3d_standard[0, 1] = 0.3 # set [::1] to 0.3 for iPlanner
        if model_name == "iPlanner":
            trajectory_3d_standard[:,1] = 0.3
            trajectory_3d_standard = (R_tilt_ip @ trajectory_3d_standard.T).T
        elif model_name == "LLMNav":
            trajectory_3d_standard = (R_tilt @ trajectory_3d_standard.T).T # R_tilt_ip for iPlanner

        # Rotate goal position
        goal_3d_standard = (R @ goal_position.unsqueeze(1)).squeeze(1)  # Shape: (3,)
       
        goal_3d_standard = (R_tilt @ goal_3d_standard.unsqueeze(1)).squeeze(1)
        if model_name == "iPlanner":
            trajectory_3d_standard[-1, 1] = goal_3d_standard[1] # for iPlanner, because its height estimation does not work!

        # Project the valid 3D waypoints to 2D pixel coordinates
        trajectory_2d_valid = project_points(trajectory_3d_standard, P)

        # Project the goal position to 2D pixel coordinates
        goal_2d = project_points(goal_3d_standard.unsqueeze(0), P)  # Shape: (1, 2)
        goal_2d_np = goal_2d.cpu().numpy()[0]  # Convert to (2,)

        trajectory_2d_valid_np = trajectory_2d_valid.cpu().numpy()

        # Ensure depth_image and risk_image are NumPy arrays and rearrange dimensions if necessary
        if torch.is_tensor(depth_image):
            # Convert from (3, 360, 640) to (360, 640, 3)
            depth_image_np = depth_image.cpu().numpy().transpose(1, 2, 0)
        else:
            # If already a NumPy array, ensure it has the correct shape
            if depth_image.ndim == 4 and depth_image.shape[0] == 3:
                depth_image_np = depth_image.transpose(1, 2, 0)
            else:
                raise ValueError(f"Depth image has incorrect shape: {depth_image.shape}")

        if torch.is_tensor(risk_image):
            # Convert from (3, 360, 640) to (360, 640, 3)
            risk_image_np = risk_image.cpu().numpy().transpose(1, 2, 0)
        else:
            # If already a NumPy array, ensure it has the correct shape
            if risk_image.ndim == 4 and risk_image.shape[0] == 3:
                risk_image_np = risk_image.transpose(1, 2, 0)
            else:
                raise ValueError(f"Risk image has incorrect shape: {risk_image.shape}")

        # Create the figure with two subplots side by side
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # Increased width for better spacing

        # Plot Depth Image on the left
        ax_depth = axes[0]
        ax_depth.imshow(depth_image_np)
        ax_depth.plot(trajectory_2d_valid_np[:, 0], trajectory_2d_valid_np[:, 1], 'ro-', label='Trajectory')
        ax_depth.plot(goal_2d_np[0], goal_2d_np[1], 'bs', label='Goal Position')  # Blue square
        ax_depth.set_title('Depth Image', fontsize=14)
        ax_depth.set_xlim(0, depth_image_np.shape[1])
        ax_depth.set_ylim(depth_image_np.shape[0], 0)  # Invert y-axis for correct orientation
        ax_depth.axis('off')
        ax_depth.legend(loc='upper right')

        # Annotate each trajectory point with its index on Depth Image
        for idx_pt, (x, y) in enumerate(trajectory_2d_valid_np):
            ax_depth.annotate(f'{idx_pt}', (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8, color='white')

        # Plot Risk Image on the right
        ax_risk = axes[1]
        ax_risk.imshow(risk_image_np)
        ax_risk.plot(trajectory_2d_valid_np[:, 0], trajectory_2d_valid_np[:, 1], 'ro-', label='Trajectory')
        ax_risk.plot(goal_2d_np[0], goal_2d_np[1], 'bs', label='Goal Position')  # Blue square
        ax_risk.set_title('Risk Image', fontsize=14)
        ax_risk.set_xlim(0, risk_image_np.shape[1])
        ax_risk.set_ylim(risk_image_np.shape[0], 0)  # Invert y-axis for correct orientation
        ax_risk.axis('off')
        ax_risk.legend(loc='upper right')

        # Annotate each trajectory point with its index on Risk Image
        for idx_pt, (x, y) in enumerate(trajectory_2d_valid_np):
            ax_risk.annotate(f'{idx_pt}', (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)

        # Adjust layout to accommodate the main title and avoid overlap
        plt.tight_layout(rect=[0, 0, 1, 0.93])

        # Save the plot if required
        if save:
            save_filename = f"{model_name}_idx{current_idx}.png"
            save_path = os.path.join(output_dir, save_filename)
            fig.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"[Batch {batch_idx}] Plot saved to {save_path}.")

        # Show the plot if required
        if show:
            plt.show()
        else:
            plt.close(fig)  # Close the figure to free memory

        # Append the figure to the list
        figures.append(fig)

    return figures



def plot_traj_batch_on_map(
    start_idxs: torch.Tensor, 
    waypoints_idxs: torch.Tensor, 
    goal_idxs: torch.Tensor, 
    grid_maps: torch.Tensor,
    save: bool = False,
    output_dir: str = 'output/map'
)-> list[plt.Figure]:
    """
    Plots the Traversability and Risk map for each item in a batch.
    Returns a list of Matplotlib Figure objects, one figure per batch item.

    Args:
        start_idxs: shape (B, 2)
        waypoints_idxs: shape (B, N, 2)
        goal_idxs: shape (B, 2)
        grid_maps: shape (B, 2, H, W) 
                   (the first '2' is for [traversability_map, risk_map]).

    Returns:
        figs: List of length B, each being a Matplotlib Figure.
    """
    B = start_idxs.shape[0]
    figs = []

    for i in range(B):
        # Extract the i-th item
        start_idx = start_idxs[i].squeeze(0)      # shape (2,)
        wp_idx = waypoints_idxs[i]         # shape (N, 2)
        goal_idx = goal_idxs[i].squeeze(0)         # shape (2,)
        grid_map = grid_maps[i]           # shape (2, H, W)

        # Convert them to numpy
        traversability_map = grid_map[0].cpu().numpy()  # shape (H, W)
        risk_map           = grid_map[1].cpu().numpy()  # shape (H, W)

        start = start_idx.detach().cpu().numpy()   # shape (2,)
        waypoints = wp_idx.detach().cpu().numpy()    # shape (N, 2)
        goal = goal_idx.detach().cpu().numpy()       # shape (2,)

        # Swap axes for plotting
        start_x, start_y = start[1], start[0]
        waypoints_x, waypoints_y = waypoints[:, 1], waypoints[:, 0]
        goal_x, goal_y = goal[1], goal[0]

        # Create figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Plot the traversability map
        ax1.imshow(traversability_map, cmap='plasma', origin='upper')
        ax1.plot(start_x, start_y, 'go', label='Start')
        ax1.plot(waypoints_x, waypoints_y, '.-', color='silver', label='Waypoints')
        ax1.plot(goal_x, goal_y, 'ro', label='Goal')
        ax1.set_title('Traversability Map')
        ax1.set_xlabel('Y-Index')
        ax1.set_ylabel('X-Index')
        ax1.axis('off')
        ax1.legend()

        # Plot the risk map
        ax2.imshow(risk_map, cmap='plasma', origin='upper')
        ax2.plot(start_x, start_y, 'go', label='Start')
        ax2.plot(waypoints_x, waypoints_y, '.-', color='silver', label='Waypoints')
        ax2.plot(goal_x, goal_y, 'ro', label='Goal')
        ax2.set_title('Risk Map')
        ax2.set_xlabel('Y-Index')
        ax2.set_ylabel('X-Index')
        ax2.axis('off')
        ax2.legend()

        plt.tight_layout()

        figs.append(fig)

    return figs

def combine_figures(fig_img: plt.Figure, fig_map: plt.Figure) -> plt.Figure:
    """
    Combines two existing Matplotlib Figure objects (fig_img, fig_map)
    by converting each to a PIL image and placing them in a new 2-row figure.

    Args:
        fig_img: Matplotlib Figure object for the image plot.
        fig_map: Matplotlib Figure object for the map plot.

    Returns:
        fig: Matplotlib Figure object with two subplots, one for each input figure.
    """
    # 1) Convert fig_img to a PIL Image
    buf_img = io.BytesIO()
    fig_img.savefig(buf_img, format='png', dpi=100, bbox_inches='tight')
    buf_img.seek(0)
    pil_img = Image.open(buf_img)

    # 2) Convert fig_map to a PIL Image
    buf_map = io.BytesIO()
    fig_map.savefig(buf_map, format='png', dpi=100, bbox_inches='tight')
    buf_map.seek(0)
    pil_map = Image.open(buf_map)

    # Good practice: close the original figures to free resources
    plt.close(fig_img)
    plt.close(fig_map)

    # 3) Create a new figure with two rows for the two PIL Images
    fig, axes = plt.subplots(2, 1, figsize=(12, 12))

    axes[0].imshow(pil_img)
    axes[0].axis('off')
    axes[0].set_title("Trajectory plotted on Depth and Risk Images")

    axes[1].imshow(pil_map)
    axes[1].axis('off')
    axes[1].set_title("Trajectory plotted on Traversability and Risk Maps")

    plt.tight_layout()
    return fig


def create_gif_from_figures(
    figures: list,
    output_path: str,
    fps: int = 1
)-> None:
    """
    Takes a list of Matplotlib Figures, converts each to a PIL image, 
    and saves them all as a GIF at output_path.

    Args:
        figures: List of Matplotlib Figure objects.
        output_path: Path to save the GIF file.
        fps: Frames per second

    Returns:
        None
    """
    frames = []
    for fig in figures:
        # Convert fig -> PNG bytes -> PIL Image
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        frames.append(Image.open(buf))

        # Close the figure to free up memory
        plt.close(fig)

    # Save frames as a GIF
    imageio.mimsave(output_path, frames, fps=fps, loop=0)

   
# ---------------------------------------------------------------------
# for comparison.py

def comparison_plot_on_map(
    start_idxs: torch.Tensor,
    waypoints1_idxs: torch.Tensor,
    waypoints2_idxs: torch.Tensor,
    goal_idxs: torch.Tensor,
    grid_maps: torch.Tensor,
    model_name1: str,
    model_name2: str
    )-> list[plt.Figure]:
        """
        Plots traversability and risk maps for each batch item, overlaying two trajectories from different models.
        
        Args:
            start_idxs: shape (B, 2)
            waypoints1_idxs: shape (B, N, 2) for the first model's trajectory
            waypoints2_idxs: shape (B, N, 2) for the second model's trajectory
            goal_idxs: shape (B, 2)
            grid_maps: shape (B, 2, H, W)
                    (the first '2' is for [traversability_map, risk_map]).
            model_name1: Name/label for the first model.
            model_name2: Name/label for the second model.
        
        Returns:
            figs: List of Matplotlib Figure objects, one per batch item.
        """
        B = start_idxs.shape[0]
        figs = []

        for i in range(B):
            # Extract the i-th item for each trajectory and maps
            start_idx = start_idxs[i].squeeze(0)         # shape (2,)
            wp1_idx = waypoints1_idxs[i]                 # shape (N, 2)
            wp2_idx = waypoints2_idxs[i]                 # shape (N, 2)
            goal_idx = goal_idxs[i].squeeze(0)           # shape (2,)
            grid_map = grid_maps[i]                      # shape (2, H, W)

            # Convert maps to numpy arrays
            traversability_map = grid_map[0].cpu().numpy()  # shape (H, W)
            risk_map = grid_map[1].cpu().numpy()            # shape (H, W)

            # Convert indices to numpy
            start = start_idx.detach().cpu().numpy()   # shape (2,)
            waypoints1 = wp1_idx.detach().cpu().numpy()  # shape (N, 2)
            waypoints2 = wp2_idx.detach().cpu().numpy()  # shape (N, 2)
            goal = goal_idx.detach().cpu().numpy()       # shape (2,)

            # Swap axes for plotting
            start_x, start_y = start[1], start[0]
            wp1_x, wp1_y = waypoints1[:, 1], waypoints1[:, 0]
            wp2_x, wp2_y = waypoints2[:, 1], waypoints2[:, 0]
            goal_x, goal_y = goal[1], goal[0]

            # Create figure with two subplots side by side
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            # Define colors for each model trajectory
            color1 = 'white'
            color2 = 'cyan'

            # Plot on Traversability Map
            ax1.imshow(traversability_map, cmap='plasma', origin='upper')
            ax1.plot(start_x, start_y, 'go', label='Start')
            ax1.plot(wp1_x, wp1_y, '.-', color=color1, label=model_name1 + ' Waypoints')
            ax1.plot(wp2_x, wp2_y, '.-', color=color2, label=model_name2 + ' Waypoints')
            ax1.plot(goal_x, goal_y, 'ro', label='Goal')
            ax1.set_title('Traversability Map')
            ax1.set_xlabel('Y-Index')
            ax1.set_ylabel('X-Index')
            ax1.legend()

            # Plot on Risk Map
            ax2.imshow(risk_map, cmap='plasma', origin='upper')
            ax2.plot(start_x, start_y, 'go', label='Start')
            ax2.plot(wp1_x, wp1_y, '.-', color=color1, label=model_name1 + ' Waypoints')
            ax2.plot(wp2_x, wp2_y, '.-', color=color2, label=model_name2 + ' Waypoints')
            ax2.plot(goal_x, goal_y, 'ro', label='Goal')
            ax2.set_title('Risk Map')
            ax2.set_xlabel('Y-Index')
            ax2.set_ylabel('X-Index')
            ax2.legend()

            plt.tight_layout()
            figs.append(fig)

        return figs

def plot_loss_comparison(x_losses: list,
                         y_losses: list,
                         metric_name: str,
                         model1_label: str = "Model 1",
                         model2_label: str = "Model 2",
                         save_dir: str = "output/Comparison"
                         ) -> None:
    """
    Plots the losses for a specific metric of two models against each other for each batch and saves the plot.
    
    Args:
        x_losses (list): Losses of metric from the first model per batch.
        y_losses (list): Losses of metric from the second model per batch.
        metric_name (str): Name of the metric being plotted.
        model1_label (str): Label for the first model.
        model2_label (str): Label for the second model.
        save_dir (str): Directory where plots will be saved.

    Returns:
        None
    """
    import os
    import matplotlib.pyplot as plt

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(x_losses, y_losses, alpha=0.7)
    
    # Set larger font sizes for title and labels
    plt.title(f"{metric_name.capitalize()} Comparison", fontsize=24)
    plt.xlabel(f"{model1_label} {metric_name}", fontsize=22)
    plt.ylabel(f"{model2_label} {metric_name}", fontsize=22)
    
    plt.grid(True)
    
    # Plot reference line y = x
    min_val = min(min(x_losses), min(y_losses))
    max_val = max(max(x_losses), max(y_losses))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='y=x')
    plt.legend(fontsize=14)
    
    # Save the figure
    filename = f"{model1_label.replace(' ', '')}2{model2_label.replace(' ', '')}_{metric_name}.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path)
    plt.close()  # Close the figure to free memory
