import os
import torch
import traceback
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

# Import your dataset and utility functions
from dataset import MapDataset
from planner_net import PlannerNet
from traj_opt import TrajOpt
from utils import prepare_data_for_plotting  # Adjust import paths as necessary
from utils_viz import plot_single_traj_on_map, plot_single_traj_on_img_with_distortion, combine_figures, create_gif_from_figures, plot_rgb_with_distortion


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
    
    best_model_path = "checkpoints/base2cam.pth"
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
    frames_rgb = []
    frames_combined = []

    num_samples_to_check = 1000
    offset = 360
    skip = 10
    max_frames = 10
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


            transform = transforms.ToTensor()

            rgb_image_path = f'TrainingData/camera/{idx}.png'
            rgb_image = Image.open(rgb_image_path).convert('RGB')
            rgb_tensor = transform(rgb_image).to(device)


            with torch.no_grad():
                preds, fear = model(depth_image.unsqueeze(0), risk_image.unsqueeze(0), goal_position.unsqueeze(0))

            waypoints = traj_opt.TrajGeneratorFromPFreeRot(preds, step = 1.0)
            waypoints = waypoints.to(device)

            start_idx, waypoints_idxs, goal_idx = prepare_data_for_plotting(waypoints, goal_position, center_position, grid_map, t_cam_to_world_SE3, t_world_to_grid_SE3, voxel_size)
            

            fig_map = plot_single_traj_on_map(start_idx, waypoints_idxs, goal_idx, grid_map)
            fig_img = plot_single_traj_on_img_with_distortion(waypoints, depth_image, risk_image)
            fig_combined = combine_figures(fig_img, fig_map)
            fig_rgb = plot_rgb_with_distortion(waypoints, rgb_tensor)

            frames_map.append(fig_map)
            frames_img.append(fig_img)
            frames_rgb.append(fig_rgb)
            frames_combined.append(fig_combined)

        except Exception as e:
            print(f"Error processing sample {idx}:")
            traceback.print_exc()
        
    img_gif_path = "output/trajectory_on_img.gif"
    map_gif_path = "output/trajectory_on_map.gif"
    rgb_gif_path = "output/trajectory_on_rgb.gif"
    combined_gif_path = "output/trajectory_combined.gif"

    create_gif_from_figures(frames_img, img_gif_path, fps)
    create_gif_from_figures(frames_map, map_gif_path, fps)
    create_gif_from_figures(frames_rgb, rgb_gif_path, fps)
    create_gif_from_figures(frames_combined, combined_gif_path, fps)
    print(f"Saved GIFs to {img_gif_path}, {map_gif_path}, and {combined_gif_path}")


if __name__ == "__main__":
    main(None)  