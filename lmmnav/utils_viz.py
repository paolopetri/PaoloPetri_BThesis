# ======================================================================
# Copyright (c) 2023 Fan Yang
# Robotic Systems Lab, ETH Zurich
# All rights reserved.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ======================================================================

import os
import cv2
import copy
import torch
import numpy as np
import pypose as pp
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import open3d.visualization.rendering as rendering
from typing import List, Optional
import matplotlib.pyplot as plt
import imageio
import io
from PIL import Image

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 360
MESH_SIZE = 0.5

class TrajViz:    
    def __init__(self, root_path: str, cameraTilt: float = 0.0):
        """
        Initialize TrajViz class.
        """
        intrinsic_path = os.path.join(root_path, "camera_p_resized.txt")
        self.SetCamera(intrinsic_path)
        self.camera_tilt = cameraTilt

    def TransformPoints(self, odom: torch.Tensor, points: torch.Tensor) -> pp.SE3:
        """
        Transform points in the trajectory.
        """
        batch_size, num_p, _ = points.shape
        world_ps = pp.identity_SE3(batch_size, num_p, device=points.device)
        world_ps.tensor()[:, :, 0:3] = points
        world_ps = pp.SE3(odom[:, None, :]) @ pp.SE3(world_ps)
        return world_ps


    def SetCamera(self, intrinsic_path: str, img_width: int = IMAGE_WIDTH, img_height: int = IMAGE_HEIGHT):
        """
        Set camera parameters.
        """
        with open(intrinsic_path) as f:
            lines = f.readlines()
            elems = np.fromstring(lines[0][1:-2], dtype=float, sep=', ')
        K = np.array(elems).reshape(-1, 4)
        self.camera = o3d.camera.PinholeCameraIntrinsic(img_width, img_height, K[0,0], K[1,1], K[0,2], K[1,2])


    def VizImages(self, preds: torch.Tensor, waypoints: torch.Tensor, odom: torch.Tensor, goal: torch.Tensor, fear: torch.Tensor, images: torch.Tensor, visual_offset: float = 0.5, mesh_size: float = 0.3, is_shown: bool = True) -> List[np.ndarray]:
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
            # print(f"wp_cam: {waypoints[i, :, :]}")
            # print(f"wp_ws: {wp_ws[i, :, :]}")
            # print(f"odom: {odom[i, :]}")
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
    

    def CameraLookAtPose(self, odom: torch.Tensor, render: o3d.visualization.rendering.OffscreenRenderer, tilt: float) -> None:
        """
        Set camera to look at at current odom pose.

        Parameters:
        odom (tensor): Odometry data
        render (OffscreenRenderer): Renderer object
        tilt (float): Tilt angle for camera
        """
        unit_vec = pp.identity_SE3(device=odom.device)
        unit_vec.tensor()[0] = -1.0
        tilt_vec = [0, 0, 0]
        tilt_vec.extend(list(R.from_euler('y', tilt, degrees=True).as_quat()))
        tilt_vec = torch.tensor(tilt_vec, device=odom.device, dtype=odom.dtype)
        target_pose = pp.SE3(odom) @ pp.SE3(tilt_vec) @ unit_vec
        camera_up = [0, 0, 1]  # camera orientation
        eye = pp.SE3(odom)
        eye = eye.tensor()[0:3].cpu().detach().numpy()
        target = target_pose.tensor()[0:3].cpu().detach().numpy()
        render.scene.camera.set_projection(self.camera.intrinsic_matrix, 0.1, 100.0, self.camera.width, self.camera.height)
        render.scene.camera.look_at(target, eye, camera_up)
        return

    def combinecv(self, listA, listB):
        """
        Combines two lists of cv2 images (listA, listB) side-by-side,
        assuming both images have identical shape (height, width, channels).
        Returns a new list of cv2 images, each horizontally stacked.
        """
        combined_list = []
        for imgA, imgB in zip(listA, listB):
            # Direct horizontal stack since shapes match
            combined = np.hstack((imgA, imgB))
            combined_list.append(combined)

        return combined_list

    def cv2fig(self, cv_image_list):
        """
        Converts each cv2 image in the list into a Matplotlib Figure.
        Returns a list of Figures.
        """
        fig_list = []
        for cv_img in cv_image_list:
            fig, ax = plt.subplots(figsize=(6, 3))  # Adjust figure size as desired
            # OpenCV is BGR(A); Matplotlib expects RGB(A).
            # If your images are BGRA, convert to RGBA:
            if cv_img.shape[2] == 4:  # BGRA
                cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2RGBA)
            else:
                cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            
            ax.imshow(cv_img_rgb)
            ax.axis('off')
            fig_list.append(fig)
        return fig_list


def plot_traj_batch_on_map(
    start_idxs: torch.Tensor, 
    waypoints_idxs: torch.Tensor, 
    goal_idxs: torch.Tensor, 
    grid_maps: torch.Tensor
):
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
        ax1.legend()

        # Plot the risk map
        ax2.imshow(risk_map, cmap='plasma', origin='upper')
        ax2.plot(start_x, start_y, 'go', label='Start')
        ax2.plot(waypoints_x, waypoints_y, '.-', color='silver', label='Waypoints')
        ax2.plot(goal_x, goal_y, 'ro', label='Goal')
        ax2.set_title('Risk Map')
        ax2.set_xlabel('Y-Index')
        ax2.set_ylabel('X-Index')
        ax2.legend()

        plt.tight_layout()

        figs.append(fig)

    return figs

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
    axes[0].set_title("Figure 1: Trajectory plotted on Depth and Risk Images")

    axes[1].imshow(pil_map)
    axes[1].axis('off')
    axes[1].set_title("Figure 2: Trajectory plotted on Traversability and Risk Maps")

    plt.tight_layout()
    return fig


def create_gif_from_figures(
    figures: list,
    output_path: str,
    fps: int = 1
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
    ):
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

def plot_loss_comparison(x_losses, y_losses, metric_name, model1_label="Model 1", model2_label="Model 2", save_dir="output/Comparison"):
    """
    Plots the losses for a specific metric of two models against each other for each batch and saves the plot.
    
    Args:
        x_losses (list): Losses of metric from the first model per batch.
        y_losses (list): Losses of metric from the second model per batch.
        metric_name (str): Name of the metric being plotted.
        model1_label (str): Label for the first model.
        model2_label (str): Label for the second model.
        save_dir (str): Directory where plots will be saved.
    """
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(x_losses, y_losses, alpha=0.7)
    plt.title(f"{metric_name.capitalize()} Comparison")
    plt.xlabel(f"{model1_label} {metric_name}")
    plt.ylabel(f"{model2_label} {metric_name}")
    plt.grid(True)
    
    # Plot reference line y = x
    min_val = min(min(x_losses), min(y_losses))
    max_val = max(max(x_losses), max(y_losses))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='y=x')
    plt.legend()
    
    # Save the figure
    filename = f"{model1_label.replace(' ', '')}2{model2_label.replace(' ', '')}_{metric_name}.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path)
    plt.close()  # Close the figure to free memory
