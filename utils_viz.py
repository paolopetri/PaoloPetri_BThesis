import torch
import numpy as np
import matplotlib.pyplot as plt
import imageio
import io
from PIL import Image
from utils import TransformPoints2Grid, Pos2Ind

# ---------------------------------------------------------------------------
# 1) Define and scale the original camera intrinsics
# ---------------------------------------------------------------------------
K_ORIG = torch.tensor([
    [612.18396,   0.0,       1016.18363],
    [  0.0,     600.29548,    671.98658],
    [  0.0,       0.0,          1.0    ]
], dtype=torch.float32)

H_ORIG, W_ORIG = 1280, 1920
H_RESIZE, W_RESIZE = 360, 640

scale_x = W_RESIZE / W_ORIG  # 640 / 1920 = 0.3333...
scale_y = H_RESIZE / H_ORIG  # 360 / 1280 = 0.28125

K_RESIZED = K_ORIG.clone()
# Scale focal lengths
K_RESIZED[0, 0] *= scale_x  # fx
K_RESIZED[1, 1] *= scale_y  # fy
# Scale principal points
K_RESIZED[0, 2] *= scale_x  # cx
K_RESIZED[1, 2] *= scale_y  # cy


# ---------------------------------------------------------------------------
# 2) image_projection
# ---------------------------------------------------------------------------
def image_projection(K: torch.Tensor, points_3d: torch.Tensor) -> torch.Tensor:
    """
    Project 3D points (in camera frame) onto the 2D image plane using pinhole projection.

    Args:
        K (torch.Tensor): Shape (3, 3). The camera intrinsic matrix.
        points_3d (torch.Tensor): Shape (B, N, 3). 3D points in the camera frame [X, Y, Z].

    Returns:
        torch.Tensor: Shape (B, N, 2). The 2D points in the image plane [u, v].
    """

    # points_3d: (B, N, 3)
    x = points_3d[..., 0]  # (B, N)
    y = points_3d[..., 1]
    z = points_3d[..., 2]

    # Check for negative/zero Z
    if (z <= 0).any():
        # Print how many are negative or zero
        num_bad_z = (z <= 0).sum().item()
        print(f"[WARNING] {num_bad_z} points have z <= 0. Their projections may be invalid.")

    # Extract intrinsics
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Pinhole projection: (u, v) = (fx * (x / z) + cx, fy * (y / z) + cy)
    u = fx * (x / z) + cx
    v = fy * (y / z) + cy

    # Stack into (B, N, 2)
    projected_2d = torch.stack([u, v], dim=-1)

    # Debug: check min/max of resulting coordinates
    min_u, max_u = projected_2d[..., 0].min().item(), projected_2d[..., 0].max().item()
    min_v, max_v = projected_2d[..., 1].min().item(), projected_2d[..., 1].max().item()
    print(f"[DEBUG] Projected coords range: u in [{min_u:.2f}, {max_u:.2f}], "
          f"v in [{min_v:.2f}, {max_v:.2f}]")

    return projected_2d


# ---------------------------------------------------------------------------
# 3) plot_trajectory_on_images
# ---------------------------------------------------------------------------
def plot_trajectory_on_images(
    waypoints_cam: torch.Tensor,
    depth: torch.Tensor,
    risk: torch.Tensor,
    K: torch.Tensor = K_RESIZED
) -> list:
    """
    For each sample in the batch:
      1) Projects 3D camera-frame points to 2D pixel coords.
      2) Overlays them on the corresponding depth and risk images (side by side).
      3) Returns a list of Matplotlib Figure objects (one per batch element).

    Args:
        waypoints_cam (torch.Tensor): (B, N, 3) batch of 3D points in camera frame.
        depth (torch.Tensor): (B, 3, H, W) batch of depth images (channels-first).
        risk (torch.Tensor): (B, 3, H, W) batch of risk maps (channels-first).
        K (torch.Tensor): (3, 3) camera intrinsic matrix (default = K_RESIZED).

    Returns:
        list: A list of matplotlib.figure.Figure objects, one per batch sample.
    """

    # 1) Project the entire batch to 2D using the provided K
    #    projected_2d: (B, N, 2)
    projected_2d = image_projection(K, waypoints_cam)

    B, C, H, W = depth.shape  # e.g. (B, 3, 360, 640)
    figs = []

    for i in range(B):
        # 2) Extract the i-th sample's data
        points_i = projected_2d[i]  # (N, 2)
        depth_i  = depth[i]         # (3, H, W) for this sample
        risk_i   = risk[i]          # (3, H, W)

        # Convert each image to NumPy for plotting => shape (H, W, 3)
        depth_i_np = depth_i.detach().cpu().numpy().transpose(1, 2, 0)
        risk_i_np  = risk_i.detach().cpu().numpy().transpose(1, 2, 0)

        # 3) Create a figure with two side-by-side subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # -- Depth image subplot --
        axes[0].imshow(depth_i_np)
        axes[0].scatter(points_i[:, 0].cpu().numpy(),
                        points_i[:, 1].cpu().numpy(),
                        c='red', s=5)
        axes[0].set_title(f"Depth Sample {i} (size: {H}x{W})")
        axes[0].axis('off')

        # -- Risk image subplot --
        axes[1].imshow(risk_i_np)
        axes[1].scatter(points_i[:, 0].cpu().numpy(),
                        points_i[:, 1].cpu().numpy(),
                        c='red', s=5)
        axes[1].set_title(f"Risk Sample {i} (size: {H}x{W})")
        axes[1].axis('off')

        figs.append(fig)

    return figs


def plot2grid(start_idx, waypoints_idxs, goal_idx, grid_map):
    """
    Plots the Traversability Map and Risk Map side by side with the starting point, waypoints, and goal position.

    Args:
        start_idx (torch.Tensor): Starting index in the grid map of shape (2,).
        waypoints_idxs (torch.Tensor): Waypoints indices in the grid map of shape (num_waypoints, 2).
        goal_idx (torch.Tensor): Goal index in the grid map of shape (2,).
        grid_map (torch.Tensor): Grid map tensor of shape (2, height, width).
    """

    # Extract the traversability and risk maps
    traversability_map = grid_map[0].cpu().numpy()  # Shape: (height, width)
    risk_map = grid_map[1].cpu().numpy()            # Shape: (height, width)

    # Extract start, waypoints, and goal indices
    start = start_idx.cpu().numpy()               # Shape: (2,)
    waypoints = waypoints_idxs.cpu().numpy()      # Shape: (num_waypoints, 2)
    goal = goal_idx.cpu().numpy()                 # Shape: (2,)

    # Swap the axes for the points (since matplotlib uses (x, y) indexing)
    start_x, start_y = start[1], start[0]
    waypoints_x, waypoints_y = waypoints[:, 1], waypoints[:, 0]
    goal_x, goal_y = goal[1], goal[0]

    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the traversability map
    ax1.imshow(traversability_map, cmap='plasma', origin='upper')
    ax1.plot(start_x, start_y, 'go', label='Start')           # Green circle
    ax1.plot(waypoints_x, waypoints_y, '.-', color = 'silver', label='Waypoints')  
    ax1.plot(goal_x, goal_y, 'ro', label='Goal')              # Red circle
    ax1.set_title('Traversability Map')
    ax1.set_xlabel('Y-Index')
    ax1.set_ylabel('X-Index')
    ax1.legend()

    # Plot the risk map
    ax2.imshow(risk_map, cmap='plasma', origin='upper')
    ax2.plot(start_x, start_y, 'go', label='Start')
    ax2.plot(waypoints_x, waypoints_y, '.-', color = 'silver', label='Waypoints')  
    ax2.plot(goal_x, goal_y, 'ro', label='Goal')
    ax2.set_title('Risk Map')
    ax2.set_xlabel('Y-Index')
    ax2.set_ylabel('X-Index')
    ax2.legend()

    # Adjust layout
    plt.tight_layout()
    # Return the figure object instead of showing it
    return fig


def fig_to_pil(fig):
    """
    Convert a Matplotlib figure to a PIL Image in memory.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    pil_img = Image.open(buf)
    return pil_img


def create_gif_from_batch(
    grid_map,
    center_position,
    t_world_to_grid_SE3,
    t_cam_to_world_SE3,
    goal_position,
    waypoints,
    grid_idxs,
    batch_size,
    device,
    voxel_size,
    length_x,
    length_y,
    fps=1,
    gif_name="trajectory.gif"
):
    """
    Creates a GIF for a batch of samples, returning the path to the GIF.
    """
    # Collect frames here
    frames = []

    # Start position is consistent or can vary from sample to sample if desired
    start_position = torch.tensor([0.0, 0.0, 0.0], device=device)


    for i in range(batch_size):
        # Extract the ith sample
        grid_map_sample = grid_map[i]
        center_position_sample = center_position[i]
        t_world_to_grid_SE3_sample = t_world_to_grid_SE3[i]
        t_cam_to_world_SE3_sample = t_cam_to_world_SE3[i]
        goal_position_sample = goal_position[i]
        waypoints_sample = waypoints[i]
        grid_idxs_sample = grid_idxs[i]

        # Expand dimensions to match your transform functions
        start_position_expanded = start_position.unsqueeze(0).unsqueeze(0)
        t_cam_to_world_SE3_expanded = t_cam_to_world_SE3_sample.unsqueeze(0)
        t_world_to_grid_SE3_expanded = t_world_to_grid_SE3_sample.unsqueeze(0)
        center_position_expanded = center_position_sample.unsqueeze(0)

        # Compute start index
        transformed_start = TransformPoints2Grid(
            start_position_expanded, 
            t_cam_to_world_SE3_expanded, 
            t_world_to_grid_SE3_expanded
        )
        start_idx = Pos2Ind(
            transformed_start, length_x, length_y, center_position_expanded, voxel_size, device
        ).squeeze(0).squeeze(0)

        # Compute goal index
        goal_position_expanded = goal_position_sample.unsqueeze(0).unsqueeze(0)
        transformed_goal = TransformPoints2Grid(
            goal_position_expanded, 
            t_cam_to_world_SE3_expanded, 
            t_world_to_grid_SE3_expanded
        )
        goal_idx = Pos2Ind(
            transformed_goal, length_x, length_y, center_position_expanded, voxel_size, device
        ).squeeze(0).squeeze(0)

        # Waypoints indices
        grid_idxs_squeezed = grid_idxs_sample  # (num_waypoints, 2)

        # Plot for this sample
        fig = plot2grid(
            start_idx.long(),
            grid_idxs_squeezed.long(),
            goal_idx.long(),
            grid_map_sample
        )

        # Convert to PIL image and store
        pil_img = fig_to_pil(fig)
        frames.append(pil_img)
        plt.close(fig)

    # Save frames to an actual GIF
    imageio.mimsave(gif_name, frames, fps=fps)

    return gif_name
