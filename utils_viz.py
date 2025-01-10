import torch
import numpy as np
import pypose as pp
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
K_RESIZED = K_RESIZED.to(device='cuda')


# ---------------------------------------------------------------------------
# 3) plot_trajectory_on_images
# ---------------------------------------------------------------------------
def plot_single_traj_on_img(
    waypoints_cam: torch.Tensor,
    depth: torch.Tensor,
    risk: torch.Tensor,
    K: torch.Tensor = K_RESIZED
) -> plt.Figure:
    """
    Plots a single set of 3D camera-frame waypoints over the given depth and risk images
    (side by side) and returns a single Matplotlib Figure.

    Args:
        waypoints_cam (torch.Tensor): (N, 3) 3D points in camera frame for this sample.
        depth (torch.Tensor): (3, H, W) depth image (channels-first).
        risk (torch.Tensor): (3, H, W) risk map (channels-first).
        K (torch.Tensor): (3, 3) camera intrinsic matrix (default = K_RESIZED).

    Returns:
        plt.Figure: A single Matplotlib Figure with two subplots (depth + risk).
    """

    # 1) Project the 3D points to 2D
    #    shape: (N, 2)
    projected_2d = pp.point2pixel(waypoints_cam, K, extrinsics=None)

    # 2) Convert the depth/risk images to NumPy
    #    shape: (H, W, 3)
    depth_np = depth.detach().cpu().numpy().transpose(1, 2, 0)
    risk_np  = risk.detach().cpu().numpy().transpose(1, 2, 0)

    # 3) Create a figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # -- Depth image subplot --
    axes[0].imshow(depth_np)
    axes[0].scatter(
        projected_2d[:, 0].cpu().numpy(),
        projected_2d[:, 1].cpu().numpy(),
        c='red', s=5
    )
    axes[0].set_title("Depth Image")
    axes[0].axis('off')

    # -- Risk image subplot --
    axes[1].imshow(risk_np)
    axes[1].scatter(
        projected_2d[:, 0].cpu().numpy(),
        projected_2d[:, 1].cpu().numpy(),
        c='red', s=5
    )
    axes[1].set_title("Risk Image")
    axes[1].axis('off')

    plt.tight_layout()
    return fig



def plot_single_traj_on_map(start_idx, waypoints_idxs, goal_idx, grid_map):
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


def combine_figures(fig_img: plt.Figure, fig_map: plt.Figure) -> plt.Figure:
    """
    Combines two existing Matplotlib Figure objects (fig_img, fig_map)
    by converting each to a PIL image and placing them in a new 2-row figure.
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
    axes[0].set_title("Figure 1 (Images)")

    axes[1].imshow(pil_map)
    axes[1].axis('off')
    axes[1].set_title("Figure 2 (Maps)")

    plt.tight_layout()
    return fig



# GIF creation

def create_gif_from_figures(
    figures: list,
    output_path: str,
    fps: int = 2
):
    """
    Takes a list of Matplotlib Figures, converts each to a PIL image, 
    and saves them all as a GIF at output_path.
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
    imageio.mimsave(output_path, frames, fps=fps)








# ---------------------------------------------------------------------------
# 4) fig_to_pil intendet for forward pass

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
    Creates a GIF for a batch of samples, returning the path to the GIF. It is meant to be used in the forwayrd pass of a model. Just for visualization purposes.
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
        fig = plot_single_traj_on_map(
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
