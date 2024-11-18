class MapDataset(torch.utils.data.Dataset):
    def _init_(self):
        # Load all grid maps into a tensor
        self.grid_maps = load_all_grid_maps()  # Shape: (num_maps, 266, 266)

        # Load grid positions in grid frame
        self.grid_positions = load_grid_positions()  # Shape: (num_maps, 2)
        
        # Load all image pairs into a list or tensor
        self.image_pairs = load_all_image_pairs()  # List or tensor of shape (num_maps, 2, channels, height, width)
        
        # Load start positions (starting frames or odometry)
        self.start_positions = load_start_positions()  # Shape: (num_maps, ...)
        
        # Load goal positions (in camera frame)
        self.goal_positions = load_goal_positions()  # Shape: (num_maps, ...)

        # Load odometry to grid transforms
        self.t_odom_to_grid = load_odom_to_grid_transforms()  # Shape: (num_maps, 4, 4)
        
        # Any necessary transformations (e.g., normalization)
        self.transform = define_transforms()  
    
    def _len_(self):
        # Returns the total number of samples
        return len(self.grid_maps)
    
    def _getitem_(self, idx):
        # Retrieve the grid map for the given index
        grid_map = self.grid_maps[idx]  # Shape: (266, 266)

        # Retrieve the grid position
        grid_position = self.grid_positions[idx]
        
        # Retrieve the image pair
        image_pair = self.image_pairs[idx]  # Tuple or tensor: (image1, image2)
        
        # Apply transformations to images if necessary
        if self.transform:
            image_pair = (self.transform(image_pair[0]), self.transform(image_pair[1]))
        
        # Retrieve the start position
        start_position = self.start_positions[idx]
        
        # Retrieve the goal position
        goal_position = self.goal_positions[idx]

        # Retrieve the odometry to grid transform
        t_odom_to_grid = self.t_odom_to_grid[idx]
        
        # Prepare and return the sample
        sample = {
            'grid_map': grid_map,
            'grid_position': grid_position,
            'images': image_pair,
            'start_position': start_position,
            'goal_position': goal_position,
            't_odom_to_grid': t_odom_to_grid
        }
        return sample