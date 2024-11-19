import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import traceback

# Assuming MapDataset class is defined in map_dataset.py
from dataset import MapDataset

def main():
    # Define the path to your debugging dataset
    data_root = 'TestData'  # Replace with the actual path

    # Initialize the MapDataset
    try:
        dataset = MapDataset(
            data_root=data_root,
            transform=None,
            device=torch.device('cuda'),  # Change to 'cuda' if using GPU
            max_episodes=2  # Limit the number of episodes to load
        )
        print("MapDataset initialized successfully.")
    except Exception as e:
        print("Error initializing MapDataset:")
        traceback.print_exc()  # Print the full stack trace
        return

    # Check the length of the dataset
    print(f"Number of samples in the dataset: {len(dataset)}")

    # Create a DataLoader for batch processing (optional)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    # Iterate over a few samples
    num_samples_to_check = 2
    for idx in range(num_samples_to_check):
        try:
            sample = dataset[idx]
            print(f"\nSample index: {idx}")

            # Print grid map shape
            grid_map = sample['grid_map']  # Shape: (2, height, width)
            print(f"Grid map shape: {grid_map.shape}")

            # Print grid position
            center_position = sample['center_position']
            print(f"Center position: {center_position}")

            # Print start position SE3
            start_pos_ori_SE3 = sample['start_pos_ori_SE3']
            print(f"Start position SE3 tensor: {start_pos_ori_SE3.tensor()}")

            # Print goal positions
            goal_positions = sample['goal_positions']
            print(f"Goal positions shape: {goal_positions.shape}")
            print(f"Goal positions: {goal_positions}")

            # Print odometry to grid transform SE3
            t_odom_to_grid_SE3 = sample['t_odom_to_grid_SE3']
            print(f"Odometry to grid SE3 tensor: {t_odom_to_grid_SE3.tensor()}")

            # Visualize grid maps
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            axs[0].imshow(grid_map[0].cpu(), cmap='gray')
            axs[0].set_title('Traversability Map')
            axs[1].imshow(grid_map[1].cpu(), cmap='gray')
            axs[1].set_title('Risk Map')
            plt.show()

            # Visualize images
            # fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            # axs[0].imshow(np.transpose(depth_image.cpu().numpy(), (1, 2, 0)))
            # axs[0].set_title('Depth Image')
            # axs[1].imshow(np.transpose(risk_image.cpu().numpy(), (1, 2, 0)))
            # axs[1].set_title('Risk Image')
            # plt.show()

        except Exception as e:
            print(f"Error processing sample index {idx}: {e}")

    # Optionally, iterate over the DataLoader
    print("\nIterating over DataLoader:")
    try:
        for batch_idx, batch in enumerate(dataloader):
            print(f"\nBatch index: {batch_idx}")
            print(f"Batch grid map shape: {batch['grid_map'].shape}")
            #print(f"Batch images shapes: {batch['images'][0].shape}, {batch['images'][1].shape}")
            print(f"Batch start positions SE3 shape: {len(batch['start_pos_ori_SE3'])}")
            print(f"Batch goal positions shape: {batch['goal_positions'].shape}")
            # Process only the first batch for debugging
            if batch_idx == 0:
                break
    except Exception as e:
        print(f"Error iterating over DataLoader: {e}")

if __name__ == '__main__':
    main()
