import torch
import pypose as pp
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import cv2
import open3d as o3d
import copy
from typing import List
import open3d.visualization.rendering as rendering

def CostofTraj(waypoints, desired_wp, goals, grid_maps, grid_idxs,
               length_x, length_y, device,
               alpha=1.0, beta = 1.0, epsilon=1.0, delta=1.0, is_map=True):
    """
    Calculates the total cost of trajectories for a batch based on traversability, risk, and goal proximity.

    Args:
        waypoints (torch.Tensor): Waypoints in camera frame. Shape: [batch_size, num_waypoints, 3]
        desired_wp (torch.Tensor): Desired direction of the waypoints. Shape: [batch_size, num_waypoints, 3]
        goals (torch.Tensor): Goal positions in grid frame. Shape: [batch_size, 3]
        grid_maps (torch.Tensor): Grid maps. Shape: [batch_size, 2, height, width]
        grid_idxs (torch.Tensor): Grid indices for waypoints. Shape: [batch_size, num_waypoints, 2]
        length_x (int): Grid map length in x-dimension.
        length_y (int): Grid map length in y-dimension.
        device (torch.device): The device to perform computations on.
        alpha (float): Weight for traversability loss.
        beta (float): Weight for risk loss.
        epsilon (float): Weight for motion loss.
        delta (float): Weight for goal loss.
        is_map (bool): Indicates if the map is initialized.

    Returns:
        total_cost (torch.Tensor): Scalar tensor representing the total cost averaged over the batch.
        t_loss_mean (torch.Tensor): Mean traversability loss over the batch.
        r_loss_mean (torch.Tensor): Mean risk loss over the batch.
        mloss_mean (torch.Tensor): Mean motion loss over the batch.
        gloss_mean (torch.Tensor): Mean goal loss over the batch.
    """
    batch_size, num_p, _ = waypoints.shape

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
        ).squeeze(1).squeeze(2)  # Shape: [batch_size, num_p]

        t_loss_M = t_loss_M.to(torch.float32)
        t_loss = torch.sum(t_loss_M, dim=1)  # Shape: [batch_size]

        # Perform grid sampling for risk
        r_loss_M = F.grid_sample(
            risk_arrays,
            norm_grid_idxs,
            mode='nearest',
            padding_mode='border',
            align_corners=True
        ).squeeze(1).squeeze(2)  # Shape: [batch_size, num_p]

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

    # Motion Loss
    desired_ds = torch.norm(desired_wp[:, 1:num_p, :] - desired_wp[:, 0:num_p-1, :], dim=2)
    wp_ds = torch.norm(waypoints[:, 1:num_p, :] - waypoints[:, 0:num_p-1, :], dim=2)
    mloss = torch.abs(desired_ds - wp_ds)
    mloss = torch.sum(mloss, axis=1)


    # Total Cost per sample
    total_cost_per_sample = alpha * t_loss + beta * r_loss + epsilon * mloss +  delta * gloss  # Shape: [batch_size]
    total_cost = torch.mean(total_cost_per_sample)  # Scalar

    # Compute means of individual losses
    t_loss_mean = t_loss.mean()
    r_loss_mean = r_loss.mean()
    mloss_mean = mloss.mean()
    gloss_mean = gloss.mean()

    return total_cost, t_loss_mean, r_loss_mean, mloss_mean, gloss_mean


def TransformPoints2Grid(waypoints, t_cam_to_world_tensor, t_world_to_grid_tensor):
    """
    Transforms waypoints from camera frame to grid frame using pp.SE3 and batched data.

    Args:
        waypoints (torch.Tensor): Waypoints in camera frame. Shape: [batch_size, num_waypoints, 3]
        t_cam_to_world_tensor (torch.Tensor): Transformation from camera to world frame. Shape: [batch_size, 7]
        t_world_to_grid_tensor (torch.Tensor): Transformation from world to grid frame. Shape: [batch_size, 7]

    Returns:
        torch.Tensor: Transformed waypoints in grid frame. Shape: [batch_size, num_waypoints, 3]
    """

    # Recreate pp.SE3 objects from tensors

    batch_size, num_waypoints, _ = waypoints.shape

    world_wp = pp.identity_SE3(batch_size, num_waypoints, device = waypoints.device, requires_grad = waypoints.requires_grad)
    world_wp.tensor()[:, :, 0:3] = waypoints

    t_cam_to_world = pp.SE3(t_cam_to_world_tensor)
    t_world_to_grid = pp.SE3(t_world_to_grid_tensor).Inv()

    # Apply the transformations using .Act()
    waypoints_world = t_cam_to_world[:, None, :] @ world_wp  # Shape: [batch_size, num_waypoints, 3]

    # Transform waypoints from world to grid frame
    waypoints_grid = t_world_to_grid[:, None, :] @ waypoints_world  # Shape: [batch_size, num_waypoints, 3]

    return waypoints_grid.tensor()[:, :, 0:3]


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
    
    center_idx = torch.tensor([(length_x - 1) / 2, (length_y - 1) / 2], device=device).view(1, 1, 2)

    # Extract x and y coordinates
    points_xy = points[..., :2]  # Shape: [batch_size, num_points, 2]

    center_xy = center_xy.unsqueeze(1)  # Shape: [batch_size, 1, 2]
    # Compute indices
    # center_xy is broadcasted over num_points
    indices = center_idx + (center_xy - points_xy) / voxel_size
    return indices


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
