"""
map_dataset.py

This module defines the MapDataset class, a custom PyTorch Dataset for loading
and processing data required for training a navigation model. The dataset
includes:
- Traversability, risk, and elevation grid maps
- Depth and risk images
- Positions and transformations (camera-to-world, world-to-grid)
- Goal positions (with optional random perturbations)

Overall, it provides a convenient interface to retrieve all necessary training
samples for learning navigation-related tasks.
"""
import os
import torch
from torch.utils.data import Dataset
import numpy as np
import pypose as pp
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from scipy.ndimage import gaussian_filter

class MapDataset(Dataset):
    """
    The MapDataset class loads and prepares data for a navigation model, including:
      - Traversability, risk, and elevation maps (stacked into one tensor).
      - Depth and risk images.
      - Center positions in the grid frame.
      - Camera-to-world and world-to-grid transformations.
      - Goal positions, optionally randomized.

    Args:
        data_root (str): Path to the root directory containing map and image data.
        random_goals (bool, optional): If True, applies random perturbations to goal positions.
        device (torch.device, optional): Device on which tensors will be placed (CPU/GPU).

    Attributes:
        data_root (str): Root directory for data.
        device (torch.device): Device for all loaded tensors.
        center_positions (torch.Tensor): (num_samples, 2) center coordinates in the grid frame.
        t_cam_to_world_SE3 (pp.SE3): (num_samples,) camera-to-world transformations as pypose SE3.
        goal_positions (torch.Tensor): (num_samples, 3) goal positions in the start frame.
        t_world_to_grid_SE3 (pp.SE3): (num_samples,) world-to-grid transformations as pypose SE3.
    """
    def __init__(self, data_root: str, random_goals: bool = False, transform = None, device: torch.device = None) -> None:

        self.data_root = data_root
      
        self.device = device if device is not None else torch.device('cpu')
    
        self.center_positions = self.load_center_positions()  # Shape: (num_maps, 2)
        
        self.t_cam_to_world_SE3 = self.load_t_cam_to_world_SE3()  # Shape: (num_maps, 7)
        
        self.goal_positions = self.load_goal_positions(random_goals)  # Shape: (num_maps, 3)

        self.t_world_to_grid_SE3 = self.load_world_to_grid_transforms_SE3()  # Shape: (num_maps, 7)

        
    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Total number of samples.
        """
        return len(self.center_positions)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Retrieves the sample at the given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the sample data:
                - 'grid_map': Tensor of shape (2, height, width), stacked traversability and risk maps.
                - 'center_position': Tensor of shape (2), center of grid in the grid frame.
                - 'image_pair': Tuple of transformed depth and risk images.
                - 't_cam_to_world_SE3': Transformation object, start position transformed to camera frame.
                - 'goal_positions': Tensor of shape (3), goal positions in the starting frame.
                - 't_world_to_grid_SE3': Tensor of shape (7), world to grid transformation parameters.
                - 'start_idx': Tensor containing the start index for waypoint generation.
        """
        traversability_map, risk_map, elevation_map = self.load_grid_map(idx)  # Shape: (height, width)
        grid_map = torch.stack([traversability_map, risk_map, elevation_map], dim=0)  # Shape: (2, H, W)

        center_position = self.center_positions[idx]
        image_pair = self.load_image_pair(idx)  # Shape: (2, C, 360, 640)
        t_cam_to_world_SE3 = self.t_cam_to_world_SE3[idx]
        goal_positions = self.goal_positions[idx]
        t_world_to_grid_SE3 = self.t_world_to_grid_SE3[idx]

        sample = {
            'grid_map': grid_map,
            'center_position': center_position,
            'image_pair': image_pair,
            't_cam_to_world_SE3': t_cam_to_world_SE3,
            'goal_positions': goal_positions,
            't_world_to_grid_SE3': t_world_to_grid_SE3,
            'start_idx': torch.tensor(idx, dtype=torch.long)
        }
        return sample

    def load_grid_map(self, idx: int, sigma: float = 1.0) -> tuple:
        """
        Load and return traversability, risk, and elevation maps for a given index.

        This also applies a Gaussian blur to the traversability and risk maps.

        Args:
            idx (int): Index of the sample to load.
            sigma (float, optional): Standard deviation for Gaussian blurring.

        Returns:
            tuple: (traversability_map, risk_map, elevation_map) as torch.Tensor objects of shape (num_samples, H, W).
        """
        elevation_dir = os.path.join(self.data_root, 'maps', 'elevation')
        traversability_dir = os.path.join(self.data_root, 'maps', 'traversability')
        risk_dir = os.path.join(self.data_root, 'maps', 'risk')

        elevation_file = os.path.join(elevation_dir, f'{idx}.txt')
        traversability_file = os.path.join(traversability_dir, f'{idx}.txt')
        risk_file = os.path.join(risk_dir, f'{idx}.txt')

        elevation_map = np.loadtxt(elevation_file, delimiter=',').T
        traversability_map = np.loadtxt(traversability_file, delimiter=',').T
        risk_map = np.loadtxt(risk_file, delimiter=',').T

        traversability_map = 1 - traversability_map

        # Blur after dilation
        trav_blurred = gaussian_filter(traversability_map, sigma=sigma)
        risk_blurred = gaussian_filter(risk_map, sigma=sigma)

        # Convert to tensors
        elevation_tensor = torch.tensor(elevation_map, dtype=torch.float32, device=self.device)
        traversability_tensor = torch.tensor(trav_blurred, dtype=torch.float32, device=self.device)
        risk_tensor = torch.tensor(risk_blurred, dtype=torch.float32, device=self.device)

        return traversability_tensor, risk_tensor, elevation_tensor


    
    def load_center_positions(self) -> torch.Tensor:
        """
        Loads the position of the center of the grid maps from a text file.
        This position is expressed in the grid frame.

        Returns:
            torch.Tensor: Tensor of shape (num_samples, 2).
        """
        filepath = os.path.join(self.data_root,'maps', 'center_positions.txt')

        center_positions = np.loadtxt(filepath, delimiter=',')
        
        center_positions_tensor = torch.tensor(center_positions, dtype=torch.float32, device=self.device) 
        
        return center_positions_tensor

    def load_image_pair(self, idx: int) -> tuple:
        """
        Loads and returns the depth and risk images for a given index.

        Args:
            idx (int): Index of the image pair to load.

        Returns:
            tuple: Tuple containing depth and risk images as torch tensors.
        """
        depth_dir = os.path.join(self.data_root, 'depth_images')
        risk_dir = os.path.join(self.data_root, 'risk_images')

        risk_files = [f for f in os.listdir(risk_dir) if f.endswith('.npy')]
        total_risk_images = len(risk_files)

        # Calculate the intended risk index (2 steps ahead of depth) missmatch in generation!
        intended_risk_idx = idx + 2
        # Clip the risk index so it does not exceed the maximum available index
        max_valid_index = total_risk_images - 1
        risk_idx = min(intended_risk_idx, max_valid_index)

        depth_path = os.path.join(depth_dir, f'{idx}.npy')
        risk_path = os.path.join(risk_dir, f'{risk_idx}.npy') # Risk image is 2 steps ahead

        depth_image = np.load(depth_path)  # Shape: (1280, 1920)
        risk_image = np.load(risk_path)    # Shape: (427, 640, 3)

        depth_image = self.ensure_three_channels(depth_image)  # Shape: (1280, 1920, 3)

        # Convert numpy arrays to torch tensors and permute to (C, H, W)
        depth_image = torch.from_numpy(depth_image).permute(2, 0, 1).float()  # Shape: (3, 1280, 1920)
        risk_image = torch.from_numpy(risk_image).permute(2, 0, 1).float()    # Shape: (3, 427, 640)

        depth_image = self.resize_image(depth_image, size=(360, 640))
        risk_image = self.resize_image(risk_image, size=(360, 640))

        return depth_image, risk_image
    
    def ensure_three_channels(self, image: np.ndarray) -> np.ndarray:
        """
        Ensures that the image has three channels.

        Args:
            image (np.ndarray): The image to check and modify.

        Returns:
            np.ndarray: The image with three channels.
        """
        if image.ndim == 2:
            # Single-channel image, stack to create three channels
            image = np.stack((image,)*3, axis=-1)
        elif image.shape[2] == 1:
            # Single-channel in last dimension, repeat to create three channels
            image = np.repeat(image, 3, axis=2)
        return image
    
    def resize_image(self, image, size: tuple) -> torch.Tensor:
        """
        Resizes the image tensor to the given size.

        Args:
            image (torch.Tensor): Image tensor of shape (C, H, W).
            size (tuple): Desired output size (height, width).

        Returns:
            torch.Tensor: Resized image tensor.
        """
        # The input image tensor is of shape (C, H, W)
        image = TF.resize(image, size=size, interpolation=TF.InterpolationMode.BILINEAR)
        return image
    

    def load_t_cam_to_world_SE3(self) -> pp.SE3:
        """
        Loads start positions and orientations of the base link from a text file,
        transforms them to the camera link frame, and returns as pp.SE3 tensors.

        Returns:
            pp.SE3: A batch of SE(3) transformations of shape [num_samples, 7].
        """
        filepath = os.path.join(self.data_root, 'base_to_world_transforms.txt')
        t_base_to_world = np.loadtxt(filepath, delimiter=',')  # Shape: (num_samples, 7)
        t_base_to_world_tensor = torch.tensor(t_base_to_world, dtype=torch.float32, device=self.device)
        t_base_to_world_SE3 = pp.SE3(t_base_to_world_tensor)  # Shape: [num_samples, 7]

        # Define the static transform from camera link to base link frame
        # TODO: Update this transform based on the actual camera-to-base link calibration
        t_cam_to_base = torch.tensor([-0.460, -0.002, 0.115, 0.544, 0.544, -0.453, -0.451], dtype=torch.float32, device=self.device) # base2cam

        # For comparison to iPlanner:
        # t_cam_to_base = torch.tensor([0.4, 0, 0, 0, 0, 0, 1], dtype=torch.float32, device=self.device) # Shape: [7] # for iPlanner Frame
        t_cam_to_base_batch = pp.SE3(t_cam_to_base.unsqueeze(0).repeat(t_base_to_world.shape[0], 1))  # Shape: [num_samples, 7]

        t_cam_to_world_SE3 = t_base_to_world_SE3 @ t_cam_to_base_batch  # Shape: [num_samples, 7]

        return t_cam_to_world_SE3


    def load_goal_positions(self, random_goals: bool = False) -> torch.Tensor:
        """
        Generate goal positions by taking a future frame from each trajectory, 
        transforming it into the current frame. Optionally apply random offsets.

        Note: iPlanner uses a different frame for planning, so the offsets are different.

        Args:
            random_goals (bool, optional): If True, randomly perturb the x/z coordinates.

        Returns:
            torch.Tensor: (num_samples, 3) positions in the camera frame.
        """

        t_cam_to_world_SE3 = self.t_cam_to_world_SE3  # Shape: [num_samples]

        num_samples = t_cam_to_world_SE3.shape[0]

        goal_indices = torch.arange(num_samples, device=self.device) + 30  # Shape: [num_samples] 

        goal_indices = torch.clamp(goal_indices, max=num_samples - 1)  # Shape: [num_samples]

        goal_positions_SE3 = t_cam_to_world_SE3[goal_indices]  # Shape: [num_samples]

        transformed_goal_SE3 = t_cam_to_world_SE3.Inv() @ goal_positions_SE3  # Shape: [num_samples]

        goal_xyz_SE3 = transformed_goal_SE3.translation()  # Extract positions (x, y, z) from SE3 objects

        if random_goals:
            x_offset = torch.empty(num_samples, device=self.device).uniform_(-5.0, 5.0)  # Shape: [num_samples, 1]
            z_offset = torch.normal(mean=0.0, std=0.3, size=(num_samples,), device=self.device)  # Shape: [num_samples, 1]
            goal_xyz_SE3[:, 0] += x_offset
            goal_xyz_SE3[:, 2] += z_offset

            # Optional: For iPlanner Frame:
            # x_offset = torch.normal(mean=0.0, std=0.3, size=(num_samples,), device=self.device)  # Shape: [num_samples, 1]
            # y_offset = torch.empty(num_samples, device=self.device).uniform_(-7.0, 7.0)  # Shape: [num_samples, 1]
            # goal_xyz_SE3[:, 0] += x_offset
            # goal_xyz_SE3[:, 1] += y_offset

        return goal_xyz_SE3  # Shape: [num_samples, 3]

        

    def load_world_to_grid_transforms_SE3(self) -> pp.SE3:
        """
        Loads world to grid transformations from a text file and returns them as pp.SE3 objects.

        Returns:
            pp.SE3: A batch of SE(3) transformations of shape [num_samples].
        """
        filepath = os.path.join(self.data_root, 'maps', 'world_to_grid_transforms.txt')
        
        transforms = np.loadtxt(filepath, delimiter=',')  # Shape: (num_samples, 7)
        
        transforms_tensor = torch.tensor(transforms, dtype=torch.float32, device=self.device)
        
        t_world_to_grid_SE3 = pp.SE3(transforms_tensor)  # Shape: [num_samples]
        
        return t_world_to_grid_SE3


