# utils_viz.py

import torch
import matplotlib.pyplot as plt
import numpy as np
import os

# For optional GIF creation
import imageio
import io
from PIL import Image

##############################################################################
# Hard-coded original intrinsics and original resolution
##############################################################################
# Suppose your camera was originally calibrated at resolution (height=1280, width=1920).
# And you had the intrinsics matrix K_ORIG:
K_ORIG = torch.tensor([
    [612.18396,   0.0,       1016.18363],
    [  0.0,     600.29548,    671.98658],
    [  0.0,       0.0,          1.0    ]
], dtype=torch.float32)

H_ORIG = 1280
W_ORIG = 1920

##############################################################################
# 1) scale_intrinsics
##############################################################################
def scale_intrinsics(K_orig, w_orig, h_orig, w_new, h_new):
    """
    Scales intrinsics matrix from the original size (h_orig, w_orig) to (h_new, w_new).

    Args:
        K_orig (torch.Tensor): The 3x3 intrinsics for the original resolution.
        w_orig (int): Original image width.
        h_orig (int): Original image height.
        w_new (int): New image width.
        h_new (int): New image height.
    Returns:
        torch.Tensor: A new 3x3 intrinsics matrix scaled to (h_new, w_new).
    """
    scale_x = w_new / w_orig
    scale_y = h_new / h_orig
    K_new = K_orig.clone()
    K_new[0, 0] *= scale_x  # fx
    K_new[1, 1] *= scale_y  # fy
    K_new[0, 2] *= scale_x  # cx
    K_new[1, 2] *= scale_y  # cy
    return K_new

##############################################################################
# 2) project_points
##############################################################################
def project_points(waypoints_cam, K, filter_negative_z=True, z_epsilon=1e-5):
    """
    Projects 3D points in camera frame (N, 3) OR (B, N, 3) onto 2D image coords.
    Returns (N, 2) or (B, N, 2).

    This function:
      - Respects a batch dimension if present (shape (B, N, 3)).
      - By default, sets Z <= z_epsilon => NaN for (u, v).
      - Uses a pinhole model: (u, v) = (fx * (x/z) + cx, fy * (y/z) + cy).

    Args:
        waypoints_cam (torch.Tensor): (N, 3) or (B, N, 3) in camera coordinates.
        K (torch.Tensor): (3, 3) intrinsics (scaled to match the image resolution).
        filter_negative_z (bool): If True, points with z <= z_epsilon become NaN.
        z_epsilon (float): Threshold for z being considered invalid.

    Returns:
        torch.Tensor: (N, 2) or (B, N, 2) with projected pixel coords.
    """
    # Check shape
    if waypoints_cam.ndim == 3:
        # shape: (B, N, 3)
        x = waypoints_cam[..., 0]
        y = waypoints_cam[..., 1]
        z = waypoints_cam[..., 2]
        batched = True
    else:
        # shape: (N, 3). Add dummy batch dimension
        waypoints_cam = waypoints_cam.unsqueeze(0)  # => (1, N, 3)
        x = waypoints_cam[..., 0]
        y = waypoints_cam[..., 1]
        z = waypoints_cam[..., 2]
        batched = False

    # Intrinsics
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Filter negative or near-zero z
    if filter_negative_z:
        valid_mask = (z > z_epsilon)
    else:
        valid_mask = torch.ones_like(z, dtype=torch.bool)

    z_safe = torch.where(z < z_epsilon, torch.tensor(z_epsilon, device=z.device), z)

    u = fx * (x / z_safe) + cx
    v = fy * (y / z_safe) + cy

    # Mark invalid z as NaN in (u,v)
    nan_val = torch.tensor(float('nan'), device=z.device)
    u = torch.where(valid_mask, u, nan_val)
    v = torch.where(valid_mask, v, nan_val)

    # Stack
    proj = torch.stack([u, v], dim=-1)  # => (B, N, 2)

    if not batched:
        # Squeeze back to (N, 2)
        proj = proj.squeeze(0)  # => (N, 2)

    return proj

##############################################################################
# 3) plot_trajectory_on_images
##############################################################################
def plot_trajectory_on_images(
    waypoints_cam,
    depth_img,
    risk_img,
    rgb_img,
    sample_idx=0,
    filter_negative_z=True
):
    """
    Plots the waypoints (in camera coords) onto depth, risk, and RGB images.
    Handles single or batched waypoints_cam.

    Args:
        waypoints_cam (torch.Tensor): (B, N, 3) or (N, 3).
        depth_img (torch.Tensor or np.ndarray): shape ~ (H_d, W_d).
        risk_img (torch.Tensor or np.ndarray): shape ~ (H_r, W_r).
        rgb_img  (torch.Tensor or np.ndarray): shape ~ (3, H_rgb, W_rgb) or (H_rgb, W_rgb, 3).
        sample_idx (int): If batched waypoints, which sample index to visualize.
        filter_negative_z (bool): If True, Z <= z_epsilon => NaN in projection.

    Returns:
        fig (plt.Figure): Matplotlib Figure with 3 subplots: depth, risk, RGB.
    """
    # 1) If batched, pick the chosen sample
    if waypoints_cam.ndim == 3:
        waypoints_cam = waypoints_cam[sample_idx]  # => (N, 3)

    def to_numpy(img):
        if isinstance(img, torch.Tensor):
            return img.detach().cpu().numpy()
        return img

    depth_np = to_numpy(depth_img)
    risk_np  = to_numpy(risk_img)
    rgb_np   = to_numpy(rgb_img)

    # If rgb is (3, H, W), reorder to (H, W, 3)
    if rgb_np.ndim == 3 and rgb_np.shape[0] == 3:
        rgb_np = np.transpose(rgb_np, (1, 2, 0))  # => (H, W, 3)

    # 2) Scale intrinsics for each image's resolution
    H_d, W_d = depth_np.shape[:2]
    K_depth = scale_intrinsics(K_ORIG, W_ORIG, H_ORIG, W_d, H_d)
    proj_depth = project_points(waypoints_cam, K_depth, filter_negative_z=filter_negative_z)

    H_r, W_r = risk_np.shape[:2]
    K_risk = scale_intrinsics(K_ORIG, W_ORIG, H_ORIG, W_r, H_r)
    proj_risk = project_points(waypoints_cam, K_risk, filter_negative_z=filter_negative_z)

    H_rgb, W_rgb = rgb_np.shape[:2]
    K_rgb = scale_intrinsics(K_ORIG, W_ORIG, H_ORIG, W_rgb, H_rgb)
    proj_rgb = project_points(waypoints_cam, K_rgb, filter_negative_z=filter_negative_z)

    # 3) Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    ax_d, ax_r, ax_rgb = axes

    # Depth
    ax_d.imshow(depth_np, cmap='gray')
    ax_d.scatter(proj_depth[..., 0], proj_depth[..., 1], c='red', s=5)
    ax_d.set_title("Depth + Trajectory")
    ax_d.set_axis_off()

    # Risk
    ax_r.imshow(risk_np, cmap='inferno')
    ax_r.scatter(proj_risk[..., 0], proj_risk[..., 1], c='lime', s=5)
    ax_r.set_title("Risk + Trajectory")
    ax_r.set_axis_off()

    # RGB
    ax_rgb.imshow(rgb_np)
    ax_rgb.scatter(proj_rgb[..., 0], proj_rgb[..., 1], c='magenta', s=5)
    ax_rgb.set_title("RGB + Trajectory")
    ax_rgb.set_axis_off()

    plt.tight_layout()
    return fig

##############################################################################
# 4) Optional: More advanced function to handle batch & separate RGB files
##############################################################################
def plot_trajectories_for_batch(
    waypoints_cam_batch,
    depth_batch,
    risk_batch,
    rgb_folder="TrainingData/camera",
    filter_negative_z=True,
    z_epsilon=1e-5
):
    """
    Example function showing how you might handle a batch of depth/risk images
    and load corresponding RGB images from a folder (e.g. "0.png", "1.png", etc.).
    Produces one figure per sample.

    Args:
        waypoints_cam_batch (torch.Tensor): (B, N, 3)
        depth_batch (torch.Tensor): (B, 3, H_d, W_d)  or (B, 1, H_d, W_d)
        risk_batch  (torch.Tensor): (B, 3, H_r, W_r)  or (B, 1, H_r, W_r)
        rgb_folder (str): path to folder with 0.png, 1.png, ...
        filter_negative_z (bool): if True, Z <= z_epsilon => NaN
        z_epsilon (float): threshold for near-zero Z

    Returns:
        list of matplotlib.figure.Figure, one per sample in the batch.
    """
    B, C_d, H_d, W_d = depth_batch.shape
    _, C_r, H_r, W_r = risk_batch.shape

    figs = []

    for i in range(B):
        # Single-sample
        depth_i = depth_batch[i]  # (3, H_d, W_d) or (1, H_d, W_d)
        risk_i  = risk_batch[i]
        waypoints_i = waypoints_cam_batch[i]  # (N, 3)

        # Convert to numpy
        depth_np = depth_i.detach().cpu().numpy()
        risk_np  = risk_i.detach().cpu().numpy()

        # If it's effectively single-channel repeated, pick the first channel
        depth_np = depth_np[0]  # => (H_d, W_d)
        risk_np  = risk_np[0]

        # Load corresponding RGB file
        rgb_path = os.path.join(rgb_folder, f"{i}.png")
        if not os.path.exists(rgb_path):
            print(f"[plot_trajectories_for_batch] No RGB found for {rgb_path}; using black.")
            rgb_img = np.zeros((H_ORIG, W_ORIG, 3), dtype=np.float32)
        else:
            rgb_img = plt.imread(rgb_path)  # shape (H, W, 3 or 4)
            if rgb_img.shape[-1] == 4:
                rgb_img = rgb_img[..., :3]  # drop alpha

        # Scale intrinsics for depth
        K_depth = scale_intrinsics(K_ORIG, W_ORIG, H_ORIG, W_d, H_d)
        proj_depth = project_points(waypoints_i, K_depth, filter_negative_z, z_epsilon)

        # Scale intrinsics for risk
        K_risk = scale_intrinsics(K_ORIG, W_ORIG, H_ORIG, W_r, H_r)
        proj_risk = project_points(waypoints_i, K_risk, filter_negative_z, z_epsilon)

        # Scale intrinsics for the actual size of the loaded RGB
        h_rgb, w_rgb = rgb_img.shape[:2]
        K_rgb = scale_intrinsics(K_ORIG, W_ORIG, H_ORIG, w_rgb, h_rgb)
        proj_rgb = project_points(waypoints_i, K_rgb, filter_negative_z, z_epsilon)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        ax_d, ax_r, ax_rgb = axes

        # Plot depth
        ax_d.imshow(depth_np, cmap='gray')
        ax_d.scatter(
            proj_depth[..., 0].cpu() if isinstance(proj_depth, torch.Tensor) else proj_depth[..., 0],
            proj_depth[..., 1].cpu() if isinstance(proj_depth, torch.Tensor) else proj_depth[..., 1],
            c='r', s=5
        )
        ax_d.set_title(f"Depth (sample {i})")
        ax_d.set_axis_off()

        # Plot risk
        ax_r.imshow(risk_np, cmap='inferno')
        ax_r.scatter(
            proj_risk[..., 0].cpu() if isinstance(proj_risk, torch.Tensor) else proj_risk[..., 0],
            proj_risk[..., 1].cpu() if isinstance(proj_risk, torch.Tensor) else proj_risk[..., 1],
            c='lime', s=5
        )
        ax_r.set_title(f"Risk (sample {i})")
        ax_r.set_axis_off()

        # Plot RGB
        ax_rgb.imshow(rgb_img)
        ax_rgb.scatter(
            proj_rgb[..., 0].cpu().numpy() if isinstance(proj_rgb, torch.Tensor) else proj_rgb[..., 0],
            proj_rgb[..., 1].cpu().numpy() if isinstance(proj_rgb, torch.Tensor) else proj_rgb[..., 1],
            c='magenta', s=5
        )
        ax_rgb.set_title(f"RGB (sample {i})")
        ax_rgb.set_axis_off()

        plt.tight_layout()
        figs.append(fig)

    return figs

##############################################################################
# 5) Utility: figs_to_gif to create a GIF from a list of figures
##############################################################################
def figs_to_gif(figs, gif_name="trajectory.gif", fps=2):
    """
    Converts a list of Matplotlib figures into a GIF using imageio.

    Args:
        figs (List[matplotlib.figure.Figure]): The figures to turn into frames.
        gif_name (str): Output filename for the GIF.
        fps (int): Frames per second in the GIF.

    Returns:
        str: The path to the GIF file created.
    """
    frames = []
    for fig in figs:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        frames.append(imageio.v2.imread(buf))
        plt.close(fig)  # close figure to free memory

    imageio.mimsave(gif_name, frames, fps=fps)
    return gif_name