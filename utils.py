import torch
import pypose as pp
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
    batch_size, num_waypoints, _ = waypoints.shape

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
    center_idx = torch.tensor([(length_x - 1) / 2, (length_y - 1) / 2], device=device)
    print(f"Center Indices: {center_idx}")

    # Ensure points and center_xy are on the correct device
    points = points.to(device)
    center_xy = center_xy.to(device)

    # Extract x and y coordinates (assuming points are at least 2D)
    points_xy = points[..., :2]  # Shape: [batch_size, num_points, 2]

    # Compute indices
    # center_xy is broadcasted over num_points
    indices = center_idx + (center_xy.unsqueeze(1) - points_xy) / voxel_size

    return indices


