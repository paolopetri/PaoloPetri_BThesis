"""
utils.py

Utility functions to run the LMM Nav.

Author: [Paolo Petri]
Date: [07.02.2025]
"""
import torch
import pypose as pp
import torch.nn.functional as F

from typing import Tuple, Union


def CostofTraj(
    waypoints: torch.Tensor,
    waypoints_grid: torch.Tensor,
    desired_wp: torch.Tensor,
    goals: torch.Tensor,
    grid_maps: torch.Tensor,
    grid_idxs: torch.Tensor,
    length_x: int,
    length_y: int,
    device: torch.device,
    ahead_dist: float = 2.0,
    trav_threshold: float = 0.5,
    risk_threshold: float = 0.5,
    alpha: float = 1.0,
    beta: float = 1.0,
    epsilon: float = 1.0,
    delta: float = 1.0,
    zeta: float = 1.0,
    is_map: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute a multi-component cost for predicted trajectories and generate fear labels
    (binary indicators of potential obstacle/risk). 

    The total cost combines:
      - Traversability loss (alpha * t_loss)
      - Risk loss (beta * r_loss)
      - Motion loss (epsilon * m_loss)
      - Goal loss (delta * g_loss)
      - Height/elevation loss (zeta * h_loss)

    Args:
        waypoints (torch.Tensor):
            Predicted waypoints in the camera frame. Shape: (batch_size, num_waypoints, 3).
        waypoints_grid (torch.Tensor):
            Predicted waypoints in the grid frame (used to calculate height loss).
            Shape: (batch_size, num_waypoints, 3).
        desired_wp (torch.Tensor):
            Desired/ideal waypoints for motion loss. 
            Shape: (batch_size, num_waypoints, 3).
        goals (torch.Tensor):
            Goal positions in camera frame or relevant frame. 
            Shape: (batch_size, 3).
        grid_maps (torch.Tensor):
            Stacked grid maps [traversability, risk, elevation], 
            Shape: (batch_size, 3, height, width).
        grid_idxs (torch.Tensor):
            Computed grid indices for each waypoint, 
            Shape: (batch_size, num_waypoints, 2).
        length_x (int):
            Width dimension of the grid map (x-axis size).
        length_y (int):
            Height dimension of the grid map (y-axis size).
        device (torch.device):
            Device on which computation is performed.
        ahead_dist (float, optional):
            Maximum distance from the start waypoint for evaluating "fear" labels 
            (threshold on obstacle/risk presence). Default is 2.0.
        trav_threshold (float, optional):
            Threshold above which traversability is considered non-traversable (obstacle fear).
        risk_threshold (float, optional):
            Threshold above which risk is considered dangerous (risk fear).
        alpha (float, optional):
            Weight for traversability loss. Defaults to 1.0.
        beta (float, optional):
            Weight for risk loss. Defaults to 1.0.
        epsilon (float, optional):
            Weight for motion loss. Defaults to 1.0.
        delta (float, optional):
            Weight for goal loss. Defaults to 1.0.
        zeta (float, optional):
            Weight for height/elevation loss. Defaults to 1.0.
        is_map (bool, optional):
            If False, skips map-based cost calculations (traversability, risk, height). 
            Defaults to True.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            - total_cost: Scalar averaged cost over the batch (tensor of shape ()).
            - t_loss_mean: Mean traversability loss (scalar).
            - r_loss_mean: Mean risk loss (scalar).
            - mloss_mean: Mean motion loss (scalar).
            - gloss_mean: Mean goal loss (scalar).
            - hloss_mean: Mean height/elevation loss (scalar).
            - fear_labels: Binary tensor of shape (batch_size, 1) indicating 
              if there's significant obstacle/risk within `ahead_dist`.
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
        elevation_arrays = grid_maps[:, 2:3, :, :]        # Shape: [batch_size, 1, height, width]

        # Perform grid sampling for traversability
        t_loss_M = F.grid_sample(
            traversability_arrays,
            norm_grid_idxs,
            mode='bicubic',
            padding_mode='border',
            align_corners=False
        ).squeeze(1).squeeze(2)  # Shape: [batch_size, num_p]

        t_loss_M = t_loss_M.to(torch.float32)#
        t_loss = torch.sum(t_loss_M, dim=1) / num_p  # Shape: [batch_size]

        # Perform grid sampling for risk
        r_loss_M = F.grid_sample(
            risk_arrays,
            norm_grid_idxs,
            mode='bicubic',
            padding_mode='border',
            align_corners=False
        ).squeeze(1).squeeze(2)  # Shape: [batch_size, num_p]

        r_loss_M = r_loss_M.to(torch.float32)
        r_loss = torch.sum(r_loss_M, dim=1) / num_p  # Shape: [batch_size]

        h_loss_M = F.grid_sample(
            elevation_arrays,
            norm_grid_idxs,
            mode='bicubic',
            padding_mode='border',
            align_corners=False
        ).squeeze(1).squeeze(2)

        h_loss_M = torch.abs(waypoints_grid[:, :, 2] - h_loss_M) # in  camera frame y is up
        h_loss = torch.sum(h_loss_M, dim=1) / num_p

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

    # Compute means of individual losses
    t_loss_mean = t_loss.mean()
    r_loss_mean = r_loss.mean()
    mloss_mean = mloss.mean()
    gloss_mean = gloss.mean()
    hloss_mean = h_loss.mean()

    # Compute total cost
    total_cost = alpha * t_loss_mean + beta * r_loss_mean + epsilon * mloss_mean + delta * gloss_mean + zeta * hloss_mean

    # fear labels
    distance_from_start = torch.cumsum(wp_ds, dim=1, dtype=wp_ds.dtype)
    floss_M = torch.clone(t_loss_M)[:, 1:]  # using traversability data
    floss_M[distance_from_start > ahead_dist] = 0.0
    obstacle_fear = torch.max(floss_M, 1, keepdim=True)[0]
    obstacle_fear = (obstacle_fear > trav_threshold).to(torch.float32)

    # Similar logic for risk
    rloss_M = torch.clone(r_loss_M)[:, 1:]  # using risk data
    rloss_M[distance_from_start > ahead_dist] = 0.0
    risk_fear = torch.max(rloss_M, 1, keepdim=True)[0]
    risk_fear = (risk_fear > risk_threshold).to(torch.float32)

    # Combine fear labels
    fear_labels = torch.clamp(obstacle_fear + risk_fear, max=1.0)

    return total_cost, t_loss_mean, r_loss_mean, mloss_mean, gloss_mean, hloss_mean, fear_labels


def TransformPoints2Grid(waypoints: torch.tensor, t_cam_to_world_tensor: torch.tensor, t_world_to_grid_tensor: torch.tensor) -> torch.tensor:
    """
    Transforms waypoints from camera frame to grid frame using pp.SE3 and batched data.

    Args:
        waypoints (torch.Tensor): Waypoints in camera frame. Shape: [batch_size, num_waypoints, 3]
        t_cam_to_world_tensor (torch.Tensor): Transformation from camera to world frame. Shape: [batch_size, 7]
        t_world_to_grid_tensor (torch.Tensor): Transformation from world to grid frame. Shape: [batch_size, 7]

    Returns:
        torch.Tensor: Transformed waypoints in grid frame. Shape: [batch_size, num_waypoints, 3]
    """


    batch_size, num_waypoints, _ = waypoints.shape

    world_wp = pp.identity_SE3(batch_size, num_waypoints, device = waypoints.device, requires_grad = waypoints.requires_grad)
    world_wp.tensor()[:, :, 0:3] = waypoints

    t_cam_to_world = pp.SE3(t_cam_to_world_tensor)
    t_world_to_grid = pp.SE3(t_world_to_grid_tensor).Inv()

    waypoints_world = t_cam_to_world[:, None, :] @ world_wp  # Shape: [batch_size, num_waypoints, 3]

    waypoints_grid = t_world_to_grid[:, None, :] @ waypoints_world  # Shape: [batch_size, num_waypoints, 3]

    return waypoints_grid.tensor()[:, :, 0:3]


def normalize_grid_indices(grid_idxs: torch.tensor, length_x: int, length_y: int) -> torch.tensor:
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


def Pos2Ind(points: torch.tensor,
            length_x: int,
            length_y: int,
            center_xy: torch.tensor,
            voxel_size: float,
            device: torch.device
            ) -> torch.tensor:
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
    center_idx = torch.tensor([(length_x - 1) / 2, (length_y - 1) / 2], device=device).view(1, 1, 2)

    points_xy = points[..., :2]  # Shape: [batch_size, num_points, 2]

    center_xy = center_xy.unsqueeze(1)  # Shape: [batch_size, 1, 2]

    indices = center_idx + (center_xy - points_xy) / voxel_size
    return indices

def prepare_data_for_plotting(
    waypoints: torch.Tensor,
    goal_position: torch.Tensor,
    center_position: torch.Tensor,
    grid_map: torch.Tensor,
    t_cam_to_world_SE3: Union[torch.Tensor, "pp.SE3"],
    t_world_to_grid_SE3: Union[torch.Tensor, "pp.SE3"],
    voxel_size: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Transform waypoints, the start position, and the goal position from the camera 
    coordinate system into the grid coordinate system, then compute their respective 
    2D indices in the grid map.

    Args:
        waypoints (torch.Tensor): Predicted or planned waypoints in the camera frame. 
        goal_position (torch.Tensor): The goal position in the camera frame. 
        center_position (torch.Tensor): The center of the grid map in the grid frame. 
        grid_map (torch.Tensor): The grid map tensor, typically containing traversability/risk/elevation data.
        t_cam_to_world_SE3 (Union[torch.Tensor, "pp.SE3"]):
            Camera-to-world transformation parameters or SE3 object (one per batch).
        t_world_to_grid_SE3 (Union[torch.Tensor, "pp.SE3"]):
            World-to-grid transformation parameters or SE3 object (one per batch).
        voxel_size (float):
            The size (in meters) of one grid cell, used in converting world-space 
            distances into grid indices.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - **start_idx_sqz** (torch.Tensor): 
                The 2D grid indices of the start position. Shape: (batch_size, 2).
            - **waypoints_idxs_sqz** (torch.Tensor): 
                The 2D grid indices of all waypoints. Shape: (batch_size, num_waypoints, 2).
            - **goal_idx_sqz** (torch.Tensor): 
                The 2D grid indices of the goal position. Shape: (batch_size, 2).

    Note:
        This function internally calls utility methods like TransformPoints2Grid and Pos2Ind 
        to convert 3D coordinates into the appropriate grid indices.
    """

    device = grid_map.device

    _, _, length_x, length_y = grid_map.shape

    start_position = torch.tensor([0.0, 0.0, 0.0], device=device)

    # prepare data for fuctions:
    start = start_position.unsqueeze(0).unsqueeze(0)
    t_cam_to_world_params = t_cam_to_world_SE3
    t_world_to_grid_params = t_world_to_grid_SE3
    goal = goal_position.unsqueeze(1)
    center_xy = center_position

    transformed_start = TransformPoints2Grid(start, t_cam_to_world_params, t_world_to_grid_params)
    transformed_waypoints = TransformPoints2Grid(waypoints, t_cam_to_world_params, t_world_to_grid_params)
    transformed_goal = TransformPoints2Grid(goal, t_cam_to_world_params, t_world_to_grid_params)
    
    start_idx = Pos2Ind(transformed_start, length_x, length_y, center_xy, voxel_size, device)
    waypoints_idxs = Pos2Ind(transformed_waypoints, length_x, length_y, center_xy, voxel_size, device)
    goal_idx = Pos2Ind(transformed_goal, length_x, length_y, center_xy, voxel_size, device)
    
    start_idx_sqz = start_idx.squeeze(0)
    waypoints_idxs_sqz = waypoints_idxs
    goal_idx_sqz = goal_idx.squeeze(0)

    return start_idx_sqz, waypoints_idxs_sqz, goal_idx_sqz