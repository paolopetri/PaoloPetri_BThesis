"""
This module defines the MapDataset class, a custom PyTorch Dataset for loading and processing data required
for training a navigation model. The dataset includes traversability and risk maps, image pairs (depth and risk images),
start positions transformed into the camera frame, and goal positions transformed into the frame of the starting position.
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
    """The MapDataset handles the loading of all necessary data from text and image files, performs necessary
        transformations using PyPose (pp), and prepares samples for training.

        Key functionalities include:
        - Loading traversability and risk grid maps and stacking them into a single tensor.
        - Loading center positions, image pairs, start positions, goal positions, and world to grid transformations.
        - Transforming start positions from the base link frame to the camera frame.
        - Generating goal positions based on future positions and transforming them into the frame of the starting position.
    """
    def __init__(self, data_root: str, transform=None, device=None) -> None:
        """
        Initializes the MapDataset by loading all necessary data.

        Args:
            data_root (str): Path to the TrainingData folder.
            transform (callable, optional): Optional transform to be applied to images.
            device (torch.device): Device on which tensors will be allocated.
        """

        self.data_root = data_root
      
        # Define default transform if none provided
        self.transform = transform if transform is not None else transforms.Compose([
            transforms.Resize((360, 640)),  # Resize to [360, 640]
            transforms.ToTensor()           # Convert PIL Image to Tensor and scale pixel values to [0, 1]
        ])
      
        self.device = device if device is not None else torch.device('cpu')
    
        self.center_positions = self.load_center_positions()  # Shape: (num_maps, 2)
        
        self.t_cam_to_world_SE3 = self.load_t_cam_to_world_SE3()  # Shape: (num_maps, 7)
        
        self.goal_positions = self.load_goal_positions()  # Shape: (num_maps, 3)

        self.t_world_to_grid_SE3 = self.load_world_to_grid_transforms_SE3()  # Shape: (num_maps, 7)

        
    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Total number of samples.
        """
        return len(self.center_positions)
    
    def __getitem__(self, idx):
        """
        Retrieves the sample at the given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the sample data:
                - 'grid_map': Tensor of shape (2, height, width), stacked traversability and risk maps.
                - 'center_position': Tensor of shape (2), center of grid in the grid frame.
                - 'images': Tuple of transformed depth and risk images.
                - 'start_position_SE3': pp.SE3 object, start position transformed to camera frame.
                - 'goal_positions': Tensor of shape (3), goal positions in the starting frame.
                - 't_world_to_grid': Tensor of shape (7), world to grid transformation parameters.
        """
        traversability_map, risk_map = self.load_grid_map(idx)  # Shape: (height), (width)
        
        grid_map = torch.stack([traversability_map, risk_map], dim=0)  # Shape: (2, 266, 266)

        center_position = self.center_positions[idx]
        
        image_pair = self.load_image_pair(idx)  # Shape: (2, C, 360, 640)]

        t_cam_to_world_SE3 = self.t_cam_to_world_SE3[idx]
        
        goal_positions = self.goal_positions[idx]

        t_world_to_grid_SE3 = self.t_world_to_grid_SE3[idx]

        sample = {
            'grid_map': grid_map,
            'center_position': center_position,
            'image_pair': image_pair,
            't_cam_to_world_SE3': t_cam_to_world_SE3,
            'goal_positions': goal_positions,
            't_world_to_grid_SE3': t_world_to_grid_SE3
        }
        return sample
    
    def load_grid_map(self, idx, sigma=1.0):
        """
        Loads traversability and risk grid map into tensors.

        Returns:
            tuple:
                - traversability_maps (torch.Tensor): Tensor of shape (num_samples, height, width).
                - risk_maps (torch.Tensor): Tensor of shape (num_samples, height, width).
        """

        traversability_dir = os.path.join(self.data_root, 'maps', 'traversability')
        risk_dir = os.path.join(self.data_root, 'maps', 'risk')

        traversability_file = os.path.join(traversability_dir, f'{idx}.txt')
        risk_file = os.path.join(risk_dir, f'{idx}.txt')

        traversability_map = np.loadtxt(traversability_file, delimiter=',').T
        risk_map = np.loadtxt(risk_file, delimiter=',').T

        traversability_map_smoothed = gaussian_filter(traversability_map, sigma=sigma)
        risk_map_smoothed = gaussian_filter(risk_map, sigma)

        traversability_tensor = torch.tensor(traversability_map_smoothed, dtype=torch.float32, device=self.device)
        risk_tensor = torch.tensor(risk_map_smoothed, dtype=torch.float32, device=self.device)

        return traversability_tensor, risk_tensor
    
    def load_center_positions(self):
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

    def load_image_pair(self, idx):
        """
        Loads and returns the depth and risk images for a given index.

        Args:
            idx (int): Index of the image pair to load.

        Returns:
            tuple: Tuple containing depth and risk images as torch tensors.
        """
        depth_dir = os.path.join(self.data_root, 'depth_images')
        risk_dir = os.path.join(self.data_root, 'risk_images')

        depth_path = os.path.join(depth_dir, f'{idx}.npy')
        risk_path = os.path.join(risk_dir, f'{idx}.npy')

        depth_image = np.load(depth_path)  # Shape: (1280, 1920)
        risk_image = np.load(risk_path)    # Shape: (427, 640, 3)

        depth_image = self.ensure_three_channels(depth_image)  # Shape: (1280, 1920, 3)

        # Convert numpy arrays to torch tensors and permute to (C, H, W)
        depth_image = torch.from_numpy(depth_image).permute(2, 0, 1).float()  # Shape: (3, 1280, 1920)
        risk_image = torch.from_numpy(risk_image).permute(2, 0, 1).float()    # Shape: (3, 427, 640)

        depth_image = self.resize_image(depth_image, size=(360, 640))
        risk_image = self.resize_image(risk_image, size=(360, 640))

        return depth_image, risk_image
    
    def ensure_three_channels(self, image):
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
    
    def resize_image(self, image, size):
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
    

    def load_t_cam_to_world_SE3(self):
        """
        Loads start positions and orientations of the base link from a text file,
        transforms them to the camera link frame, and returns as pp.SE3 tensors.

        Returns:
            pp.SE3: A batch of SE(3) transformations of shape [num_samples, 7].
        """
        filepath = os.path.join(self.data_root, 't_cam_to_world.txt')
        t_base_to_world = np.loadtxt(filepath, delimiter=',')  # Shape: (num_samples, 7)
        t_base_to_world_tensor = torch.tensor(t_base_to_world, dtype=torch.float32, device=self.device)
        t_base_to_world_SE3 = pp.SE3(t_base_to_world_tensor)  # Shape: [num_samples, 7]

        # Define the static transform from camera link to base link frame
        t_cam_to_base = torch.tensor([-0.460, -0.002, 0.115, 0.544, 0.544, -0.453, -0.451], dtype=torch.float32, device=self.device) # Shape: [7]
        # t_cam_to_base = torch.tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float32, device=self.device) # Shape: [7] # Identity transform
  
        t_cam_to_base_batch = pp.SE3(t_cam_to_base.unsqueeze(0).repeat(t_base_to_world.shape[0], 1))  # Shape: [num_samples, 7]

        t_cam_to_world_SE3 = t_base_to_world_SE3 @ t_cam_to_base_batch  # Shape: [num_samples, 7]

        return t_cam_to_world_SE3


    def load_goal_positions(self):
        """
        Generates goal positions based on the next position after the start position
        and transforms them into the frame of the start position.

        Returns:
            torch.Tensor: Transformed goal positions as a tensor of shape (num_samples, 3).
        """

        t_cam_to_world_SE3 = self.t_cam_to_world_SE3  # Shape: [num_samples]

        num_samples = t_cam_to_world_SE3.shape[0]

        goal_indices = torch.arange(num_samples, device=self.device) + 30  # Shape: [num_samples]

        goal_indices = torch.clamp(goal_indices, max=num_samples - 1)  # Shape: [num_samples]

        goal_positions_SE3 = t_cam_to_world_SE3[goal_indices]  # Shape: [num_samples]

        transformed_goal_SE3 = t_cam_to_world_SE3.Inv() @ goal_positions_SE3  # Shape: [num_samples]

        goal_xyz_SE3 = transformed_goal_SE3.translation()  # Extract positions (x, y, z) from SE3 objects

        return goal_xyz_SE3  # Shape: [num_samples, 3]

        

    def load_world_to_grid_transforms_SE3(self):
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


