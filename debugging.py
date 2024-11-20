import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import traceback
import pypose as pp  # Import PyPose for SE3 transformations

# Import your dataset and utility functions
from dataset import MapDataset
from utils import CostofTraj, TransformPoints2Grid, Pos2Ind  # Adjust the import paths as necessary

def main():
    # Define the path to your debugging dataset
    data_root = 'TestData'  # Replace with the actual path

    # Initialize the MapDataset
    try:
        dataset = MapDataset(
            data_root=data_root,
            transform=None,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            max_episodes=1 
        )
        print("MapDataset initialized successfully.")
    except Exception as e:
        print("Error initializing MapDataset:")
        traceback.print_exc() 
        return

    # Check the length of the dataset
    print(f"Number of samples in the dataset: {len(dataset)}")

    # Create a DataLoader for batch processing
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    # Iterate over a few samples
    num_samples_to_check = 1
    for idx in range(num_samples_to_check):
        try:
            sample = dataset[idx]
            print(f"\nSample index: {idx}")

            # Extract data from the sample
            grid_map = sample['grid_map']  # Shape: (2, height, width)
            center_position = sample['center_position']  # Shape: [2]
            start_pos_ori_SE3 = sample['start_pos_ori_SE3']  # pp.SE3 object
            goal_position = sample['goal_positions']  # Shape: [3]
            t_odom_to_grid_SE3 = sample['t_odom_to_grid_SE3']  # pp.SE3 object

            device = grid_map.device

            # Print grid map shape
            print(f"Grid map shape: {grid_map.shape}")

            # Print grid position
            print(f"Center position: {center_position}")

            # Print start position SE3
            print(f"Start position SE3 tensor: {start_pos_ori_SE3.tensor()}")

            # Print goal positions
            print(f"Goal positions: {goal_position}")

            # Print odometry to grid transform SE3
            print(f"Odometry to grid SE3 tensor: {t_odom_to_grid_SE3.tensor()}")

            # Generate waypoints between start and goal positions
            num_waypoints = 3

            # Starting position
            start_position = torch.tensor([0.0, 0.0, 0.0], device=device)  # Assuming start position at origin

            # Get goal position
            goal_position = goal_position.to(device)  # Ensure device consistency

            # Generate waypoints (linear interpolation)
            waypoints = torch.linspace(0, 1, num_waypoints, device=device).unsqueeze(1) * (goal_position - start_position).unsqueeze(0) + start_position.unsqueeze(0)
            print(f"Waypoints shape: {waypoints.shape}")
            print(f"Waypoints: {waypoints}")

            # Prepare goals tensor
            goals = goal_position.unsqueeze(0)  # Shape: [1, 3]
            print(f"Goal postion shape: {goal_position.shape}")
            print(f"Goal postion: {goal_position}")

            # Prepare transformation parameters
            t_cam_to_odom_params = start_pos_ori_SE3.unsqueeze(0)
            t_odom_to_grid_params = t_odom_to_grid_SE3.unsqueeze(0)  # Shape: [1, 7]

            print(f"t_cam_to_odom_params shape: {t_cam_to_odom_params.shape}")

            # Get grid dimensions
            _, height, width = grid_map.shape  # grid_map shape is (2, H, W)

            # Prepare center_xy tensor
            center_xy = center_position.unsqueeze(0)  # Shape: [1, 2]

            # Define voxel size
            voxel_size = 1.0

            length_x = width
            length_y = height
            
            # Transform waypoints to grid coordinates
            transformed_waypoints = TransformPoints2Grid(waypoints, t_cam_to_odom_params, t_odom_to_grid_params)  # Shape: [1, num_waypoints, 3]
            print(f"Transformed waypoints to grid shape: {transformed_waypoints.shape}")
            print(f"Transformed waypoints to grid: {transformed_waypoints}")
    
            # Compute grid indices
            grid_idxs = Pos2Ind(transformed_waypoints, length_x, length_y, center_xy, voxel_size, device)

            # Calculate the trajectory cost
            total_cost = CostofTraj(
                waypoints=waypoints,
                goals=goals,
                grid_maps=grid_map.unsqueeze(0),
                grid_idxs=grid_idxs,
                length_x=width,
                length_y=height,
                device=device,
                alpha=1,
                epsilon=1.0,
                delta=1,
                is_map=True
            )

            print(f"Total cost: {total_cost.item()}")

            # Transform goal to grid frame
            transformed_goal = TransformPoints2Grid(goals, t_cam_to_odom_params, t_odom_to_grid_params)  # Shape: [1, 1, 3]
            print(f"Transformed goal shape: {transformed_goal.shape}")
            # Get starting position in grid frame
            t_cam_to_grid_SE3 = pp.SE3(t_odom_to_grid_params[0]) @ pp.SE3(t_cam_to_odom_params[0])
            start_position_grid = t_cam_to_grid_SE3.translation().unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, 3]

            # Prepare points for indexing
            points = torch.cat([start_position_grid, transformed_waypoints, transformed_goal], dim=1)  # Shape: [1, num_points, 3]

            # Convert positions to grid indices
            indices = Pos2Ind(points, width, height, center_xy, voxel_size, device)  # Shape: [1, num_points, 2]
            indices = indices.squeeze(0).cpu().numpy()  # Shape: [num_points, 2]

            # Extract indices for plotting
            start_index = indices[0]
            waypoints_indices = indices[1:-1]
            goal_index = indices[-1]

            # Adjust indices for image coordinate system (swap x and y)
            start_index = start_index[::-1]  # [y, x]
            waypoints_indices = waypoints_indices[:, ::-1]
            goal_index = goal_index[::-1]

            # Prepare the traversability map for plotting
            traversability_map = grid_map[0].cpu().numpy()  # Shape: [H, W]

            # Plotting
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(traversability_map, cmap='gray')

            # Plot starting position
            ax.plot(start_index[1], start_index[0], 'go', label='Start')

            # Plot waypoints
            ax.plot(waypoints_indices[:, 1], waypoints_indices[:, 0], 'b-', label='Waypoints')

            # Plot goal position
            ax.plot(goal_index[1], goal_index[0], 'ro', label='Goal')

            # Plot starting orientation arrow
            rotation_matrix = t_cam_to_grid_SE3.rotation().matrix().cpu().numpy()
            direction_vector = rotation_matrix[:2, 0]  # Get x and y components

            # Normalize and scale the direction vector for plotting
            direction_vector = direction_vector / np.linalg.norm(direction_vector) * 10  # Adjust the scale as needed

            # Plot the orientation arrow
            ax.arrow(start_index[1], start_index[0], direction_vector[1], direction_vector[0],
                     head_width=5, head_length=5, fc='green', ec='green')

            ax.legend()
            plt.title('Trajectory on Traversability Map')
            plt.gca().invert_yaxis()  # Invert y-axis to match image coordinate system
            plt.show()

        except Exception as e:
            print(f"Error processing sample index {idx}:")
            traceback.print_exc()
    """
    # Optionally, iterate over the DataLoader
    print("\nIterating over DataLoader:")
    try:
        for batch_idx, batch in enumerate(dataloader):
            print(f"\nBatch index: {batch_idx}")
            print(f"Batch grid map shape: {batch['grid_map'].shape}")
            print(f"Batch start positions SE3 shape: {len(batch['start_pos_ori_SE3'])}")
            print(f"Batch goal positions shape: {batch['goal_positions'].shape}")
            # Process only the first batch for debugging
            if batch_idx == 0:
                break
    except Exception as e:
        print("Error iterating over DataLoader:")
        traceback.print_exc()
    """
if __name__ == '__main__':
    main()
