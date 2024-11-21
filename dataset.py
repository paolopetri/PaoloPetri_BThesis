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
from PIL import Image

class MapDataset(Dataset):
    """The MapDataset handles the loading of all necessary data from text and image files, performs necessary
        transformations using PyPose (pp), and prepares samples for training.

        Key functionalities include:
        - Loading traversability and risk grid maps and stacking them into a single tensor.
        - Loading center positions, image pairs, start positions, goal positions, and odometry to grid transformations.
        - Transforming start positions from the base link frame to the camera frame.
        - Generating goal positions based on future positions and transforming them into the frame of the starting position."""
    def __init__(self, data_root: str, transform=None, device=None) -> None:
        """
        Initializes the MapDataset by loading all necessary data.

        Args:
            data_root (str): Path to the TrainingData folder.
            transform (callable, optional): Optional transform to be applied to images.
            device (torch.device): Device on which tensors will be allocated.
        """

        self.data_root = data_root
      
        self.transform = transform
      
        self.device = device if device is not None else torch.device('cpu')
    
        # Load all data
       
        # Load all grid maps into a tensor
        self.traversability_maps, self.risk_maps = self.load_all_grid_maps()  # Shape: (num_maps, 266, 266)

        # Load center positions in grid frame
        self.center_positions = self.load_center_positions()  # Shape: (num_maps, 2)
        
        # Load all image pairs into a list or tensor
        #self.image_pairs = self.load_all_image_pairs()  # List or tensor of shape (num_maps, 2, channels, height, width)
        
        # Load start positions (starting frames or odometry)
        self.start_pos_ori_SE3 = self.load_start_pos_ori_SE3()  # Shape: (num_maps, 7)
        
        # Load goal positions (in camera frame)
        self.goal_positions = self.load_goal_positions()  # Shape: (num_maps,max_episodes, 3)

        # Load odometry to grid transforms
        self.t_odom_to_grid_SE3 = self.load_odom_to_grid_transforms_SE3()  # Shape: (num_maps, 7)
        
        # Any necessary transformations (e.g., normalization)
        #self.transform = self.define_transforms()  
    
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
        # image_pair = self.image_pairs[idx]  # Tuple or tensor: (image1, image2)
        
        # Apply transformations to images if necessary
        # if self.transform:
        #     image_pair = (self.transform(image_pair[0]), self.transform(image_pair[1]))
        
        # Retrieve the start position
        start_pos_ori_SE3 = self.start_pos_ori_SE3[idx]
        
        # Retrieve the goal position
        goal_positions = self.goal_positions[idx]

        # Retrieve the odometry to grid transform
        t_odom_to_grid_SE3 = self.t_odom_to_grid_SE3[idx]
        
        # Prepare and return the sample
        sample = {
            'grid_map': grid_map,
            'center_position': center_position,
            #'images': image_pair,
            'start_pos_ori_SE3': start_pos_ori_SE3,
            'goal_positions': goal_positions,
            't_odom_to_grid_SE3': t_odom_to_grid_SE3
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
                traversability_map = np.loadtxt(t_filepath, delimiter=',')
                traversability_tensor = torch.tensor(traversability_map, dtype=torch.float32, device=self.device)
                traversability_maps.append(traversability_tensor)

                # Load risk map
                r_filepath = os.path.join(risk_dir, r_file)
                risk_map = np.loadtxt(r_filepath, delimiter=',')
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

    # def load_all_image_pairs(self):
        
    #     Loads all image pairs (depth and risk images) into a list.

    #     #Returns:
    #         list: List containing tuples of (depth_image, risk_image) PIL Images.
       
    #     image_pairs = []
    #     depth_dir = os.path.join(self.data_root, 'depth')
    #     risk_dir = os.path.join(self.data_root, 'risk')
    #         list: List containing tuples of (depth_image, risk_image) PIL Images.
       
    #     image_pairs = []
    #     depth_dir = os.path.join(self.data_root, 'depth')
    #     risk_dir = os.path.join(self.data_root, 'risk')
    #     num_samples = len(self)  # Assuming the number of samples is determined by grid maps

    #     for idx in range(num_samples):
    #         depth_path = os.path.join(depth_dir, f'{idx}.png')
    #         risk_path = os.path.join(risk_dir, f'{idx}.png')

    #         depth_image = Image.open(depth_path).convert('RGB')
    #     num_samples = len(self)  # Assuming the number of samples is determined by grid maps

    #     for idx in range(num_samples):
    #         depth_path = os.path.join(depth_dir, f'{idx}.png')
    #         risk_path = os.path.join(risk_dir, f'{idx}.png')

    #         depth_image = Image.open(depth_path).convert('RGB')
    #         risk_image = Image.open(risk_path).convert('RGB')

    #         image_pairs.append((depth_image, risk_image))

    #     return image_pairs

    def load_start_pos_ori_SE3(self):
        """
        Loads start positions and orientations of the base link from a text file,
        transforms them to the camera link frame, and returns as pp.SE3 tensors.

        Returns:
            pp.SE3: A batch of SE(3) transformations of shape [num_samples, 7].
        """
        filepath = os.path.join(self.data_root, 'start_pos_ori.txt')
        start_positions = np.loadtxt(filepath, delimiter=',')  # Shape: (num_samples, 7)
        start_positions_tensor = torch.tensor(start_positions, dtype=torch.float32, device=self.device)

        # Convert to pp.SE3 (batch of transformations)
        start_positions_SE3 = pp.SE3(start_positions_tensor)  # Shape: [num_samples, 7]

        # Define the static transform from base link to camera link frame
        # TODO: Replace these values with the actual static transform values
        t_base_to_cam_params = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=torch.float32, device=self.device) # Shape: [7]
        
        # Repeat the static transform for the entire batch
        t_base_to_cam_batch = pp.SE3(t_base_to_cam_params.unsqueeze(0).repeat(start_positions_SE3.shape[0], 1))  # Shape: [num_samples, 7]
        

        # Apply the transformation to all start positions
        return start_positions_SE3 @ t_base_to_cam_batch  # Shape: [num_samples, 7]


    def load_goal_positions(self):
        """
        Generates goal positions based on the next position after the start position
        and transforms them into the frame of the start position.

        Returns:
            torch.Tensor: Transformed goal positions as a tensor of shape (num_samples, 3).
        """
        import pypose as pp

        # Load start positions as SE(3) transformations
        start_pos_ori_SE3 = self.start_pos_ori_SE3  # Shape: [num_samples]

        num_samples = start_pos_ori_SE3.shape[0]

        # Create indices for the next positions
        goal_indices = torch.arange(num_samples, device=self.device) + 1  # Shape: [num_samples]
        print(goal_indices)

        # Clip goal_indices to handle the last index (avoid out-of-bounds)
        goal_indices = torch.clamp(goal_indices, max=num_samples - 1)  # Shape: [num_samples]
        print(goal_indices)

        # Get goal positions as SE(3) transformations
        goal_positions_SE3 = self.start_pos_ori_SE3[goal_indices]  # Shape: [num_samples]

        # Transform goal positions into the frame of the starting position
        transformed_goal_SE3 = start_pos_ori_SE3.Inv() @ goal_positions_SE3  # Shape: [num_samples]

        # Extract positions (x, y, z) from transformed SE3 objects
        transformed_goal_positions = transformed_goal_SE3.translation()  # Shape: [num_samples, 3]
        print(transformed_goal_positions)

        return transformed_goal_positions.to(self.device)  # Shape: [num_samples, 3]

        

    def load_odom_to_grid_transforms_SE3(self):
        """
        Loads odometry to grid transformations from a text file and returns them as pp.SE3 objects.

        Returns:
            pp.SE3: A batch of SE(3) transformations of shape [num_samples].
        """
        filepath = os.path.join(self.data_root, 'maps', 'odom_to_grid_transforms.txt')
        
        transforms = np.loadtxt(filepath, delimiter=',')  # Shape: (num_samples, 7)
        
        transforms_tensor = torch.tensor(transforms, dtype=torch.float32, device=self.device)
        
        # Convert the tensor to pp.SE3 objects
        t_odom_to_grid_SE3 = pp.SE3(transforms_tensor)  # Shape: [num_samples]
        
        return t_odom_to_grid_SE3


