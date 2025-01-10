import os
import torch
import traceback
import matplotlib.pyplot as plt

# Import your dataset and utility functions
from dataset import MapDataset
from planner_net import PlannerNet
from traj_opt import TrajOpt
from utils import prepare_data_for_plotting  # Adjust import paths as necessary
from utils_viz import plot_single_traj_on_map, plot_single_traj_on_img, combine_figures, create_gif_from_figures


def main(args):
    data_root = 'TrainingData'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        dataset = MapDataset(
            data_root=data_root,
            transform=None,
            device=device
        )
        print("MapDataset initialized successfully.")
    except Exception as e:
        print("Error initializing MapDataset:")
        traceback.print_exc()
        return

    model = PlannerNet(16, 5).to(device)
    traj_opt = TrajOpt()
    
    best_model_path = "checkpoints/best_model.pth"
    try:
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint)
        print(f"Loaded best model from {best_model_path}")
    except Exception as e:
        print("Error loading best model:")
        traceback.print_exc()
        return

    model.eval()

    voxel_size = 0.15

    frames_map = []
    frames_img = []
    frames_combined = []

    num_samples_to_check = 1000
    offset = 320
    skip = 10
    max_frames = 20
    fps = 1

    for i in range(num_samples_to_check):
        idx = i + offset

        if len(frames_map) >= max_frames:
            break

        if i % skip != 0:
            continue

        try:
            sample = dataset[idx]

            grid_map = sample['grid_map'].to(device)
            center_position = sample['center_position'].to(device)
            t_cam_to_world_SE3 = sample['t_cam_to_world_SE3'].to(device)
            goal_position = sample['goal_positions'].to(device)
            t_world_to_grid_SE3 = sample['t_world_to_grid_SE3'].to(device)
            depth_image, risk_image = sample['image_pair']
            depth_image = depth_image.to(device)
            risk_image = risk_image.to(device)

            with torch.no_grad():
                preds, fear = model(depth_image.unsqueeze(0), risk_image.unsqueeze(0), goal_position.unsqueeze(0))

            waypoints = traj_opt.TrajGeneratorFromPFreeRot(preds, step = 0.5)
            waypoints = waypoints.to(device)

            start_idx, waypoints_idxs, goal_idx = prepare_data_for_plotting(waypoints, goal_position, center_position, grid_map, t_cam_to_world_SE3, t_world_to_grid_SE3, voxel_size)
            
            fig_map = plot_single_traj_on_map(start_idx, waypoints_idxs, goal_idx, grid_map)
            fig_img = plot_single_traj_on_img(waypoints, depth_image, risk_image)
            fig_combined = combine_figures(fig_img, fig_map)

            frames_map.append(fig_map)
            frames_img.append(fig_img)
            frames_combined.append(fig_combined)

        except Exception as e:
            print(f"Error processing sample {idx}:")
            traceback.print_exc()
        
    img_gif_path = "output/trajectory_on_img.gif"
    map_gif_path = "output/trajectory_on_map.gif"
    combined_gif_path = "output/trajectory_combined.gif"

    create_gif_from_figures(frames_img, img_gif_path, fps)
    create_gif_from_figures(frames_map, map_gif_path, fps)
    create_gif_from_figures(frames_combined, combined_gif_path, fps)
    print(f"Saved GIFs to {img_gif_path}, {map_gif_path}, and {combined_gif_path}")


if __name__ == "__main__":
    main(None)  