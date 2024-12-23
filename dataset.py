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
from PIL import Image as PILImage

class MapDataset(Dataset):
    """The MapDataset handles the loading of all necessary data from text and image files, performs necessary
        transformations using PyPose (pp), and prepares samples for training.

        Key functionalities include:
        - Loading traversability and risk grid maps and stacking them into a single tensor.
        - Loading center positions, image pairs, start positions, goal positions, and odometry to grid transformations.
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
    
        # Load all data
       
        # Load all grid maps into a tensor
        self.traversability_maps, self.risk_maps = self.load_all_grid_maps()  # Shape: (num_maps, 266, 266)

        # Load center positions in grid frame
        self.center_positions = self.load_center_positions()  # Shape: (num_maps, 2)
        
        # Load start positions (starting frames or odometry)
        self.t_cam_to_grid_SE3 = self.load_t_cam_to_grid_SE3()  # Shape: (num_maps, 7)
        
        # Load goal positions (in camera frame)
        self.goal_positions = self.load_goal_positions()  # Shape: (num_maps,max_episodes, 3)


        
    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Total number of samples.
        """
        return len(self.traversability_maps)
    
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
                - 'goal_positions': Tensor of shape (max_episodes, 3), goal positions in the starting frame.
                - 't_odom_to_grid': Tensor of shape (7), odometry to grid transformation parameters.
        """
        # Retrieve the grid map for the given index
        traversability_map = self.traversability_maps[idx]  # Shape: (height), (width)
        risk_map = self.risk_maps[idx]                      # Shape: (height), (width)
        
        # Stack the maps into a single tensor
        grid_map = torch.stack([traversability_map, risk_map], dim=0)  # Shape: (2, 266, 266)

        # Retrieve the grid position
        center_position = self.center_positions[idx]
        
        # Retrieve the image pair
        depth_image, risk_image = self.load_image_pair(idx)  # Tuple or tensor: (image1, image2)
        
        # Apply transformations to images if necessary
        if self.transform:
            depth_image = self.transform(depth_image)  # Tensor of shape (C, 360, 640)
            risk_image = self.transform(risk_image)    # Tensor of shape (C, 360, 640)

        # Stack the images into a single tensor
        image_pair = (depth_image, risk_image)  # Shape: (2, C, 360, 640)]
        
        # Retrieve the start position
        t_cam_to_grid_SE3 = self.t_cam_to_grid_SE3[idx]
        
        # Retrieve the goal position
        goal_positions = self.goal_positions[idx]
        
        # Prepare and return the sample
        sample = {
            'grid_map': grid_map,
            'center_position': center_position,
            'image_pair': image_pair,
            't_cam_to_grid_SE3': t_cam_to_grid_SE3,
            'goal_positions': goal_positions,
        }
        return sample
    
    def load_all_grid_maps(self):
        """
        Loads all traversability and risk grid maps into tensors.

        Returns:
            tuple:
                - traversability_maps (torch.Tensor): Tensor of shape (num_samples, height, width).
                - risk_maps (torch.Tensor): Tensor of shape (num_samples, height, width).
        """
        traversability_maps = []
        risk_maps = []

        traversability_dir = os.path.join(self.data_root, 'maps', 'traversability')
        risk_dir = os.path.join(self.data_root, 'maps', 'risk')

        traversability_files = sorted(os.listdir(traversability_dir))
        risk_files = sorted(os.listdir(risk_dir))

        for t_file, r_file in zip(traversability_files, risk_files):
            if t_file.endswith('.txt') and r_file.endswith('.txt'):
                # Load traversability map
                t_filepath = os.path.join(traversability_dir, t_file)
                traversability_map = np.loadtxt(t_filepath, delimiter=',').T
                #traversability_map_flip = np.flip(traversability_map, axis=1)
                #traversability_map_rot = np.rot90(traversability_map_flip, k=-1)

                # Make a copy to ensure positive strides
                #ätraversability_map_final = traversability_map_flip.copy()

                traversability_tensor = torch.tensor(traversability_map, dtype=torch.float32, device=self.device)
                traversability_maps.append(traversability_tensor)

                # Load risk map
                r_filepath = os.path.join(risk_dir, r_file)
                risk_map = np.loadtxt(r_filepath, delimiter=',').T
                #risk_map_flip = np.flip(risk_map, axis=1)
                #risk_map_rot = np.rot90(risk_map_flip, k=-1)


                # Make a copy to ensure positive strides
                #risk_map_final = risk_map_flip.copy()

                risk_tensor = torch.tensor(risk_map, dtype=torch.float32, device=self.device)
                risk_maps.append(risk_tensor)

        # Stack the lists into tensors
        traversability_maps = torch.stack(traversability_maps)  # Shape: (num_samples, height, width)
        risk_maps = torch.stack(risk_maps)                      # Shape: (num_samples, height, width)

        return traversability_maps, risk_maps
    
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
            tuple: Tuple containing depth and risk images as PIL Images.
        """
        depth_dir = os.path.join(self.data_root, 'depth_images')
        risk_dir = os.path.join(self.data_root, 'risk_images')

        depth_path = os.path.join(depth_dir, f'{idx}.png')
        risk_path = os.path.join(risk_dir, f'{idx}.png')

        depth_image = PILImage.open(depth_path).convert('RGB')
        risk_image = PILImage.open(risk_path).convert('RGB')
        
        return depth_image, risk_image
    

    def load_t_cam_to_grid_SE3(self):
        """
        Loads start positions and orientations of the base link from a text file,
        transforms them to the camera link frame, and returns as pp.SE3 tensors.

        Returns:
            pp.SE3: A batch of SE(3) transformations of shape [num_samples, 7].
        """
        filepath = os.path.join(self.data_root, 't_cam_to_grid.txt')
        t_cam_to_grid = np.loadtxt(filepath, delimiter=',')  # Shape: (num_samples, 7)
        t_cam_to_grid_tensor = torch.tensor(t_cam_to_grid, dtype=torch.float32, device=self.device)

        # Convert to pp.SE3 (batch of transformations)
        t_cam_to_grid_SE3 = pp.SE3(t_cam_to_grid_tensor)  # Shape: [num_samples, 7]

        return t_cam_to_grid_SE3


    def load_goal_positions(self):
        """
        Generates goal positions based on the next position after the start position
        and transforms them into the frame of the start position.

        Returns:
            torch.Tensor: Transformed goal positions as a tensor of shape (num_samples, 3).
        """

        # Load start positions as SE(3) transformations
        t_cam_to_grid_SE3 = self.t_cam_to_grid_SE3  # Shape: [num_samples]

        num_samples = t_cam_to_grid_SE3.shape[0]

        # Create indices for the next positions
        goal_indices = torch.arange(num_samples, device=self.device) + 30  # Shape: [num_samples]

        # Clip goal_indices to handle the last index (avoid out-of-bounds)
        goal_indices = torch.clamp(goal_indices, max=num_samples - 1)  # Shape: [num_samples]

        # Get goal positions as SE(3) transformations
        goal_positions_SE3 = t_cam_to_grid_SE3[goal_indices]  # Shape: [num_samples]
        
        # Transform goal positions into the frame of the starting position
        transformed_goal_SE3 = t_cam_to_grid_SE3.Inv() @ goal_positions_SE3  # Shape: [num_samples]

        goal_xyz_SE3 = transformed_goal_SE3.translation()  # Extract positions (x, y, z) from SE3 objects

        return goal_xyz_SE3  # Shape: [num_samples, 3]



