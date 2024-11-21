import torch
import pypose as pp
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

def CostofTraj(waypoints, goals, grid_maps, grid_idxs,
               length_x, length_y, device,
               alpha=1, epsilon=1.0, delta=1, is_map=True):
    """
    Calculates the total cost of trajectories for a batch based on traversability, risk, and goal proximity.

    Args:
        waypoints (torch.Tensor): Waypoints in camera frame. Shape: [batch_size, num_waypoints, 3]
        goals (torch.Tensor): Goal positions in grid frame. Shape: [batch_size, 3]
        grid_maps (torch.Tensor): Grid maps. Shape: [batch_size, 2, height, width]
        grid_idxs (torch.Tensor): Grid indices for waypoints. Shape: [batch_size, num_waypoints, 2]
        length_x (int): Grid map length in x-dimension.
        length_y (int): Grid map length in y-dimension.
        device (torch.device): The device to perform computations on.
        alpha (float): Weight for traversability loss.
        epsilon (float): Weight for risk loss.
        delta (float): Weight for goal loss.
        is_map (bool): Indicates if the map is initialized.

    Returns:
        torch.Tensor: Total cost (scalar).
    """
    batch_size, num_waypoints, _ = waypoints.shape

    if is_map:
        # Normalize grid indices
        norm_grid_idxs = normalize_grid_indices(grid_idxs, length_x, length_y) 

        # Ensure grid_maps is on the correct device
        grid_maps = grid_maps.to(device)  # Shape: [batch_size, 2, height, width]

        # Split grid_maps into traversability and risk maps
        traversability_arrays = grid_maps[:, 0:1, :, :]  # Shape: [batch_size, 1, height, width]
        risk_arrays = grid_maps[:, 1:2, :, :]            # Shape: [batch_size, 1, height, width]

        # Perform grid sampling for traversability
        t_loss_M = F.grid_sample(
            traversability_arrays,
            norm_grid_idxs,
            mode='nearest',
            padding_mode='border',
            align_corners=True
        ).squeeze(1).squeeze(2)  # Shape: [batch_size, num_waypoints]

        t_loss_M = t_loss_M.to(torch.float32)
        t_loss = torch.sum(t_loss_M, dim=1)  # Shape: [batch_size]

        # Perform grid sampling for risk
        r_loss_M = F.grid_sample(
            risk_arrays,
            norm_grid_idxs,
            mode='nearest',
            padding_mode='border',
            align_corners=True
        ).squeeze(1).squeeze(2)  # Shape: [batch_size, num_waypoints]

        r_loss_M = r_loss_M.to(torch.float32)
        r_loss = torch.sum(r_loss_M, dim=1)  # Shape: [batch_size]

    else:
        # If the map is not initialized
        t_loss = torch.zeros(batch_size, device=device)
        r_loss = torch.zeros(batch_size, device=device)

    # Goal Loss
    # waypoints[:, -1, :] is the last waypoint for each sample
    gloss = torch.norm(goals - waypoints[:, -1, :], dim=1)  # Shape: [batch_size]
    gloss = torch.log(gloss + 1.0)  # Shape: [batch_size]

    # Total Cost per sample
    total_cost_per_sample = alpha * t_loss + epsilon * r_loss + delta * gloss  # Shape: [batch_size]

    # Aggregate total cost over the batch
    total_cost = torch.mean(total_cost_per_sample)  # Scalar

    # Optionally, print individual losses
    print(f"Traversability Loss: {t_loss.mean().item()}")
    print(f"Risk Loss: {r_loss.mean().item()}")
    print(f"Goal Loss: {gloss.mean().item()}")

    return total_cost



def TransformPoints2Grid(waypoints, t_cam_to_odom, t_odom_to_grid):
    """
    Transforms waypoints from camera frame to grid frame using pp.SE3 and batched data.

    Args:
        waypoints (torch.Tensor): Waypoints in camera frame. Shape: [batch_size, num_waypoints, 3]
        t_cam_to_odom (torch.Tensor): Transformation from camera to odometry frame. Shape: [batch_size, 7]
        t_odom_to_grid (torch.Tensor): Transformation from odometry to grid frame. Shape: [batch_size, 7]

    Returns:
        torch.Tensor: Transformed waypoints in grid frame. Shape: [batch_size, num_waypoints, 3]
    """

    # Transform waypoints from camera to odometry frame
    waypoints_odom = t_cam_to_odom @ waypoints  # Shape: [batch_size, num_waypoints, 3]

    # Transform waypoints from odometry to grid frame
    waypoints_grid = t_odom_to_grid @ waypoints_odom  # Shape: [batch_size, num_waypoints, 3]

    return waypoints_grid  # Return as a tensor


def normalize_grid_indices(grid_idxs, length_x, length_y):
    """
    Normalizes grid indices from positive values to [-1, 1] range for grid_sample.

    Args:
        grid_idxs (torch.Tensor): Tensor of grid indices. Shape: [batch_size, num_waypoints, 2] (x, y)
        length_x (int): Length of the grid map in x-dimension (width).
        length_y (int): Length of the grid map in y-dimension (height).

    Returns:
        torch.Tensor: Normalized grid indices. Shape: [batch_size, num_waypoints, 1, 2]
    """
    batch_size, num_waypoints, _ = grid_idxs.shape

    # Separate x and y indices
    x_idxs = grid_idxs[:, :, 0]  # Shape: [batch_size, num_waypoints]
    y_idxs = grid_idxs[:, :, 1]  # Shape: [batch_size, num_waypoints]

    # Normalize x and y to [-1, 1]
    normalized_x = (x_idxs / (length_x - 1)) * 2 - 1  # Shape: [batch_size, num_waypoints]
    normalized_y = (y_idxs / (length_y - 1)) * 2 - 1  # Shape: [batch_size, num_waypoints]

    # Stack and reshape to match grid_sample's expected input shape
    normalized_grid = torch.stack([normalized_y, normalized_x], dim=2)  # Shape: [batch_size, num_waypoints, 2]
    grid = normalized_grid.view(batch_size, num_waypoints, 1, 2)  # Shape: [batch_size, num_waypoints, 1, 2]

    return grid


def Pos2Ind(points, length_x, length_y, center_xy, voxel_size, device):
    """
    Converts a list of points to indices in the map array.

    Args:
        points (torch.Tensor): Tensor of points. Shape: [batch_size, num_points, 2] or [batch_size, num_points, 3]
        length_x (int): Length of the map in the x-dimension.
        length_y (int): Length of the map in the y-dimension.
        center_xy (torch.Tensor): Tensor of center coordinates. Shape: [batch_size, 2]
        voxel_size (float): Size of each voxel in the map.
        device (torch.device): The device to perform computations on.

    Returns:
        torch.Tensor: Indices in the map array. Shape: [batch_size, num_points, 2]
    """
    # Calculate center indices (broadcasted over batch_size)
    # TODO: Could add a batch dimention to center_xy to avoid broadcasting
    center_idx = torch.tensor([(length_x - 1) / 2, (length_y - 1) / 2], device=device)
    print(f"Center Indices: {center_idx}")

    # Extract x and y coordinates
    points_xy = points[..., :2]  # Shape: [batch_size, num_points, 2]

    # Compute indices
    # center_xy is broadcasted over num_points
    indices = center_idx + (center_xy - points_xy) / voxel_size

    return indices

def plotting(start_idx, waypoints_idxs, goal_indx, grid_map):
    """
    Plots the Traversability Map and Risk Map side by side with the starting point, waypoints, and goal position.

    Args:
        start_idx (torch.Tensor): Starting index in the grid map of shape (batch_size, 2).
        waypoints_idxs (torch.Tensor): Waypoints indices in the grid map of shape (batch_size, num_waypoints, 2).
        goal_indx (torch.Tensor): Goal index in the grid map of shape (batch_size, 2).
        grid_map (torch.Tensor): Grid map tensor of shape (batch_size, 2, height, width).
    """

    # Determine the number of samples in the batch
    batch_size = grid_map.shape[0]

    for i in range(batch_size):
        # Extract traversability and risk maps for the current sample
        traversability_map = grid_map[i, 0].cpu().numpy()  # Shape: [height, width]
        risk_map = grid_map[i, 1].cpu().numpy()            # Shape: [height, width]

        # Extract start, waypoints, and goal indices for the current sample
        start = start_idx[i].cpu().numpy()          # Shape: [2] -> [x, y]
        waypoints = waypoints_idxs[i].cpu().numpy() # Shape: [num_waypoints, 2] -> [[x1, y1], [x2, y2], ...]
        goal = goal_indx[i].cpu().numpy()           # Shape: [2] -> [x, y]

        # Extract x and y coordinates
        start_x, start_y = start[0], start[1]
        waypoints_x, waypoints_y = waypoints[:, 0], waypoints[:, 1]
        goal_x, goal_y = goal[0], goal[1]

        # Get grid dimensions
        height, width = traversability_map.shape

        # Clip indices to ensure they lie within the grid boundaries
        start_x = np.clip(start_x, 0, height - 1)
        start_y = np.clip(start_y, 0, width - 1)
        waypoints_x = np.clip(waypoints_x, 0, height - 1)
        waypoints_y = np.clip(waypoints_y, 0, width - 1)
        goal_x = np.clip(goal_x, 0, height - 1)
        goal_y = np.clip(goal_y, 0, width - 1)

        # -----------------------------
        # Create a Single Figure with Two Subplots
        # -----------------------------
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # -----------------------
        # Plot Traversability Map
        # -----------------------
        ax1 = axes[0]
        # Adjust 'extent' to center the grid cells around their indices
        ax1.imshow(traversability_map, cmap='gray', origin='upper', extent=[-0.5, width - 0.5, height - 0.5, -0.5])
        # Plot start point
        ax1.plot(start_y, start_x, 'go', markersize=10, label='Start')  # Green circle
        # Plot waypoints line and individual markers
        if waypoints_x.size > 0:
            ax1.plot(waypoints_y, waypoints_x, 'b-', linewidth=2, label='Waypoints')  # Blue line
            ax1.plot(waypoints_y, waypoints_x, 'bo', markersize=6)  # Blue circles for waypoints
            # Add annotations for each waypoint
            for wp_idx, (wp_x, wp_y) in enumerate(zip(waypoints_x, waypoints_y)):
                ax1.text(wp_y + 0.1, wp_x + 0.1, f'WP{wp_idx+1}', color='blue', fontsize=9)
        # Plot goal point
        ax1.plot(goal_y, goal_x, 'ro', markersize=10, label='Goal')  # Red circle
        # Title and labels
        ax1.set_title(f'Sample {i+1} - Traversability Map')
        ax1.set_xlabel('Y Index (Left to Right)')
        ax1.set_ylabel('X Index (Top to Bottom)')
        ax1.legend()
        ax1.grid(False)
        # Set limits to align with grid boundaries
        ax1.set_xlim(-0.5, width - 0.5)
        ax1.set_ylim(height - 0.5, -0.5)  # Inverted to have x increase from top to bottom

        # -------------------
        # Plot Risk Map
        # -------------------
        ax2 = axes[1]
        # Adjust 'extent' similarly for risk map
        ax2.imshow(risk_map, cmap='hot', origin='upper', extent=[-0.5, width - 0.5, height - 0.5, -0.5])
        # Plot start point
        ax2.plot(start_y, start_x, 'go', markersize=10, label='Start')  # Green circle
        # Plot waypoints line and individual markers
        if waypoints_x.size > 0:
            ax2.plot(waypoints_y, waypoints_x, 'b-', linewidth=2, label='Waypoints')  # Blue line
            ax2.plot(waypoints_y, waypoints_x, 'bo', markersize=6)  # Blue circles for waypoints
            # Add annotations for each waypoint
            for wp_idx, (wp_x, wp_y) in enumerate(zip(waypoints_x, waypoints_y)):
                ax2.text(wp_y + 0.1, wp_x + 0.1, f'WP{wp_idx+1}', color='blue', fontsize=9)
        # Plot goal point
        ax2.plot(goal_y, goal_x, 'ro', markersize=10, label='Goal')  # Red circle
        # Title and labels
        ax2.set_title(f'Sample {i+1} - Risk Map')
        ax2.set_xlabel('Y Index (Left to Right)')
        ax2.set_ylabel('X Index (Top to Bottom)')
        ax2.legend()
        ax2.grid(False)
        # Set limits to align with grid boundaries
        ax2.set_xlim(-0.5, width - 0.5)
        ax2.set_ylim(height - 0.5, -0.5)  # Inverted to have x increase from top to bottom

        # Adjust layout for better spacing
        plt.tight_layout()

        # Display the combined figure
        plt.show()
