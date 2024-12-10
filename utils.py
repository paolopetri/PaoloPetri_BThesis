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


def plot2img(self, preds: torch.Tensor, waypoints: torch.Tensor, odom: torch.Tensor, goal: torch.Tensor, fear: torch.Tensor, images: torch.Tensor, visual_offset: float = 0.5, mesh_size: float = 0.3, is_shown: bool = True) -> List[np.ndarray]:
        """
        Visualize images of the trajectory.

        Parameters:
        preds (tensor): Predicted Key points
        waypoints (tensor): Trajectory waypoints
        odom (tensor): Odometry data
        goal (tensor): Goal position
        fear (tensor): Fear value per waypoint
        images (tensor): Image data
        visual_offset (float): Offset for visualizing images
        mesh_size (float): Size of mesh objects in images
        is_shown (bool): If True, show images; otherwise, return image list
        """
        batch_size, _, _ = waypoints.shape

        preds_ws = self.TransformPoints(odom, preds)
        wp_ws = self.TransformPoints(odom, waypoints)

        if goal.shape[-1] != 7:
            pp_goal = pp.identity_SE3(batch_size, device=goal.device)
            pp_goal.tensor()[:, 0:3] = goal
            goal = pp_goal.tensor()

        goal_ws  = pp.SE3(odom) @ pp.SE3(goal)

        # Detach to CPU
        preds_ws = preds_ws.tensor()[:, :, 0:3].cpu().detach().numpy()
        wp_ws    = wp_ws.tensor()[:, :, 0:3].cpu().detach().numpy()
        goal_ws  = goal_ws.tensor()[:, 0:3].cpu().detach().numpy()

        # Adjust height
        preds_ws[:, :, 2] -= visual_offset
        wp_ws[:, :, 2]    -= visual_offset
        goal_ws[:, 2]     -= visual_offset

        # Set material shader
        mtl = o3d.visualization.rendering.MaterialRecord()
        mtl.base_color = [1.0, 1.0, 1.0, 0.3]
        mtl.shader = "defaultUnlit"

        # Set meshes
        small_sphere      = o3d.geometry.TriangleMesh.create_sphere(mesh_size/20.0)  # trajectory points
        mesh_sphere       = o3d.geometry.TriangleMesh.create_sphere(mesh_size/5.0)  # successful predict points
        mesh_sphere_fear  = o3d.geometry.TriangleMesh.create_sphere(mesh_size/5.0)  # unsuccessful predict points
        mesh_box          = o3d.geometry.TriangleMesh.create_box(mesh_size, mesh_size, mesh_size)  # end points

        # Set colors
        small_sphere.paint_uniform_color([0.99, 0.2, 0.1])  # green
        mesh_sphere.paint_uniform_color([0.4, 1.0, 0.1])
        mesh_sphere_fear.paint_uniform_color([1.0, 0.64, 0.0])
        mesh_box.paint_uniform_color([1.0, 0.64, 0.1])

        # Init open3D render
        render = rendering.OffscreenRenderer(self.camera.width, self.camera.height)
        render.scene.set_background([0.0, 0.0, 0.0, 1.0])  # RGBA
        render.scene.scene.enable_sun_light(False)

        # compute veretx normals
        small_sphere.compute_vertex_normals()
        mesh_sphere.compute_vertex_normals()
        mesh_sphere_fear.compute_vertex_normals()
        mesh_box.compute_vertex_normals()
        
        wp_start_idx = 1
        cv_img_list = []

        for i in range(batch_size):
            # Add geometries
            gp = goal_ws[i, :]

            # Add goal marker
            goal_mesh = copy.deepcopy(mesh_box).translate((gp[0]-mesh_size/2.0, gp[1]-mesh_size/2.0, gp[2]-mesh_size/2.0))
            render.scene.add_geometry("goal_mesh", goal_mesh, mtl)

            # Add predictions
            for j in range(preds_ws[i, :, :].shape[0]):
                kp = preds_ws[i, j, :]
                if fear[i, :] > 0.5:
                    kp_mesh = copy.deepcopy(mesh_sphere_fear).translate((kp[0], kp[1], kp[2]))
                else:
                    kp_mesh = copy.deepcopy(mesh_sphere).translate((kp[0], kp[1], kp[2]))
                render.scene.add_geometry("keypose"+str(j), kp_mesh, mtl)

            # Add trajectory
            for k in range(wp_start_idx, wp_ws[i, :, :].shape[0]):
                wp = wp_ws[i, k, :]
                wp_mesh = copy.deepcopy(small_sphere).translate((wp[0], wp[1], wp[2]))
                render.scene.add_geometry("waypoint"+str(k), wp_mesh, mtl)

            # Set cameras
            self.CameraLookAtPose(odom[i, :], render, self.camera_tilt)

            # Project to image
            img_o3d = np.asarray(render.render_to_image())
            mask = (img_o3d < 10).all(axis=2)

            # Attach image
            c_img = images[i, :, :].expand(3, -1, -1)
            c_img = c_img.cpu().detach().numpy().transpose(1, 2, 0)
            c_img = (c_img * 255 / np.max(c_img)).astype('uint8')
            img_o3d[mask, :] = c_img[mask, :]
            img_cv2 = cv2.cvtColor(img_o3d, cv2.COLOR_RGBA2BGRA)
            cv_img_list.append(img_cv2)

            # Visualize image
            if is_shown: 
                cv2.imshow("Preview window", img_cv2)
                cv2.waitKey()

            # Clear render geometry
            render.scene.clear_geometry()        

        return cv_img_list