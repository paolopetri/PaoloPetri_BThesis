import torch
from torch.utils.data import DataLoader
import traceback
import pypose as pp  # Import PyPose for SE3 transformations

# Import your dataset and utility functions
from dataset import MapDataset
from utils import CostofTraj, TransformPoints2Grid, Pos2Ind, plotting  # Adjust the import paths as necessary

def main(args):
    # Define the path to your debugging dataset
    data_root = 'TrainingData'  # Replace with the actual path

    # Initialize the MapDataset
    try:
        dataset = MapDataset(
            data_root=data_root,
            transform=None,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
        print("MapDataset initialized successfully.")
    except Exception as e:
        print("Error initializing MapDataset:")
        traceback.print_exc() 
        return

    # Check the length of the dataset
    print(f"Number of samples in the dataset: {len(dataset)}")

    # Create a DataLoader for batch processing
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    # Iterate over a few samples
    num_samples_to_check = 1000
    for idx in range(num_samples_to_check):
        try:
            offset = 0
            sample = dataset[idx + offset]
            print(f"\nSample index: {idx + offset}")

            # Extract data from the sample
            grid_map = sample['grid_map']  # Shape: (2, height, width)
            center_position = sample['center_position']  # Shape: [2]
            t_cam_to_grid_SE3 = sample['t_cam_to_grid_SE3']  # pp.SE3 object
            goal_position = sample['goal_positions']  # Shape: [3]

            device = grid_map.device

            # Print grid map shape
            print(f"Grid map shape: {grid_map.shape}")

            # Print grid position
            print(f"Center position: {center_position}")

            # Print start position SE3
            print(f"Start position SE3 tensor: {t_cam_to_grid_SE3}")

            # Print goal positions
            print(f"Goal positions: {goal_position}")

            # Generate waypoints between start and goal positions
            num_waypoints = 3

            # Starting position
            start_position = torch.tensor([0.0, 0.0, 0.0], device=device)  # Assuming start position at origin

            # Get goal position
            goal_position = goal_position.to(device)  # Ensure device consistency

            # Generate waypoints (linear interpolation)
            waypoints = torch.linspace(0, 1, num_waypoints, device=device).unsqueeze(1) * (goal_position - start_position).unsqueeze(0) + start_position.unsqueeze(0)
            waypoints = waypoints.unsqueeze(0)  # Add batch dimension
            print(f"Waypoints shape: {waypoints.shape}")
            print(f"Waypoints: {waypoints}")

            # Prepare goals tensor
            goal = goal_position.unsqueeze(0)  # Shape: [1, 3]
            print(f"Goal postion shape: {goal_position.shape}")
            print(f"Goal postion: {goal_position}")

            # Prepare transformation parameters
            t_cam_to_grid_SE3_batch = t_cam_to_grid_SE3.unsqueeze(0)

            print(f"t_cam_to_grid_SE3_batch shape: {t_cam_to_grid_SE3_batch.shape}")

            # Get grid dimensions
            _, height, width = grid_map.shape  # grid_map shape is (2, H, W)

            # Prepare center_xy tensor
            center_xy = center_position.unsqueeze(0)  # Shape: [1, 2]

            # Define voxel size
            voxel_size = 0.15

            length_x = height
            length_y = width
            # For hardcoded waypoints
            print("Hardcoded waypoints type:", type(waypoints))
            print("Hardcoded waypoints dtype:", waypoints.dtype)
            print("Hardcoded waypoints shape:", waypoints.shape)
                        
            # Transform waypoints to grid coordinates
            transformed_waypoints = TransformPoints2Grid(waypoints, t_cam_to_grid_SE3_batch)  # Shape: [1, num_waypoints, 3]
            print(f"Transformed waypoints to grid: {transformed_waypoints}")
            print(f"Transformed waypoints to grid shape: {transformed_waypoints.shape}")
    
            # Compute grid indices
            waypoints_idxs = Pos2Ind(transformed_waypoints, length_x, length_y, center_xy, voxel_size, device)
            print(f"Waypoints indices: {waypoints_idxs}")
            print(f"Waypoints indices shape: {waypoints_idxs.shape}")
            # Calculate the trajectory cost
            total_cost = CostofTraj(
                waypoints=waypoints,
                goals=goal,
                grid_maps=grid_map.unsqueeze(0),
                grid_idxs=waypoints_idxs,
                length_x=length_x,
                length_y=length_y,
                device=device,
                alpha=1,
                epsilon=1.0,
                delta=1,
                is_map=True
            )

            print(f"Total cost: {total_cost.item()}")

            transformed_start = TransformPoints2Grid(start_position.unsqueeze(0).unsqueeze(0), t_cam_to_grid_SE3_batch)  # Shape: [1, 3]
            print(f"Transformed start position: {transformed_start}")
            start_idx = Pos2Ind(transformed_start, length_x, length_y, center_xy, voxel_size, device)
            print(f"Start index: {start_idx}")
            transformed_goal = TransformPoints2Grid(goal.unsqueeze(0), t_cam_to_grid_SE3_batch)  # Shape: [1, 3]
            print(f"Transformed goal position: {transformed_goal}")
            goal_indx = Pos2Ind(transformed_goal, length_x, length_y, center_xy, voxel_size, device)
            print(f"Goal index: {goal_indx}")
            print(f"Waypoints indices: {waypoints_idxs}")


            start_idx_squeezed = start_idx.squeeze(1)
            goal_indx_squeezed = goal_indx.squeeze(1)

            plotting(start_idx_squeezed, waypoints_idxs, goal_indx_squeezed, grid_map.unsqueeze(0))

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
    main(args=None)
