"""
test_run.py

This script visualizes waypoints and trajectories produced by a trained PlannerNet model.
It loads a saved checkpoint of the model, takes a subset of data from MapDataset, and plots
the predicted waypoints on both the occupancy grid map and depth/risk images. Additionally,
it generates GIFs of the combined trajectory plots and RGB views for easier visualization.

Usage:
    python3 test_run.py

Author: [Paolo Petri]
Date: [06.02.2025]
"""
import os
import io
import torch
import imageio
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

from dataset import MapDataset
from utils_viz import plot_traj_batch_on_map, combine_figures, plot_waypoints_on_depth_risk, plot_single_waypoints_on_rgb
from utils import prepare_data_for_plotting
from planner_net import PlannerNet
from iplanner_planner_net import iPlannerPlannerNet
from traj_opt import TrajOpt

def main() -> None:
    """
    Main function to visualize model-predicted trajectories.

    Steps:
        1. Loads a subset of the MapDataset from `data_root`.
        2. Initializes PlannerNet (and optionally iPlannerPlannerNet) with a
           pre-trained checkpoint.
        3. Iterates through the data to predict waypoints using the loaded model.
        4. Plots waypoints on depth/risk images and occupancy grid maps, then
           saves and combines them into GIF animations for easy inspection.

    Returns:
        None
    """
    data_root = 'TrainingData/Important'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1) Create dataset
    dataset = MapDataset(
        data_root=data_root,
        device=device
    )

    snippet_indices = range(190, 194)
    
    subset_dataset = Subset(dataset, snippet_indices)

    viz_loader = DataLoader(
        subset_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0
    )

    traj_opt = TrajOpt()
    model = PlannerNet(32, 5).to(device)
    best_model_path = "checkpoints/best_model.pth"
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint)
    print(f"Loaded best model from {best_model_path}")

    model.eval()

    # Optional: Load iPlanner model (commented out)
    # iplanner_model = iPlannerPlannerNet(16, 5).to(device)
    # iplanner_model_path = "checkpoints/iplanner.pt"
    # iplanner_checkpoint_tuple = torch.load(iplanner_model_path, map_location=device)
    # loaded_model_instance, some_value = iplanner_checkpoint_tuple
    # state_dict = loaded_model_instance.state_dict()
    # iplanner_model.load_state_dict(state_dict)
    # print(f"Loaded iPlanner model from {iplanner_model_path}")
    # iplanner_model.eval()

    voxel_size = 0.15

    # Open a writer for the final GIF
    final_output_path = "output/trajectory_combined_final.gif"
    writer = imageio.get_writer(final_output_path, fps=1, loop=0)
    writer_rgb = imageio.get_writer("output/trajectory_rgb.gif", fps=1, loop=0)

    with torch.no_grad():
        for i, sample in enumerate(viz_loader):
            # Data preparation as before
            grid_map = sample['grid_map'].to(device)
            center_position = sample['center_position'].to(device)
            t_cam_to_world_SE3 = sample['t_cam_to_world_SE3'].to(device)
            goal_position = sample['goal_positions'].to(device)
            t_world_to_grid_SE3 = sample['t_world_to_grid_SE3'].to(device)
            depth_img, risk_img = sample['image_pair']
            depth_img = depth_img.to(device)
            risk_img = risk_img.to(device)
            idx = sample['start_idx']

            # Optional: permute goals in camera frame
            # goal_position[:, 0] += +7.0
            # goal_position[:, 2] += 0

            # Forward pass
            preds, fear = model(depth_img, risk_img, goal_position)
            waypoints = traj_opt.TrajGeneratorFromPFreeRot(preds, step=1.0)

            figs_img = plot_waypoints_on_depth_risk(waypoints, goal_position, depth_img, risk_img, idx, model_name='LLMNav', frame='LLMNav', show=False, save=False)

            start_idx, waypoints_idxs, goal_idx = prepare_data_for_plotting(
                waypoints, goal_position, center_position, grid_map, 
                t_cam_to_world_SE3, t_world_to_grid_SE3, voxel_size
            )

            figs_map = plot_traj_batch_on_map(start_idx, waypoints_idxs, goal_idx, grid_map)

            figs_rgb =plot_single_waypoints_on_rgb(waypoints, goal_position, idx, model_name='LLMNav', frame='LLMNAV', show=False, save=True)

            output_dir_combined = "output/image/combined"
            os.makedirs(output_dir_combined, exist_ok=True)

            # Combine figures and write frames directly to GIF
            for fig_img, fig_map in zip(figs_img, figs_map):
                fig_combined = combine_figures(fig_img, fig_map)

                i = idx[0]
                safe_path = os.path.join(output_dir_combined, f"{i}.png")
                fig_combined.savefig(safe_path, format='png', dpi=100, bbox_inches='tight')

                # Convert figure to image data
                buf = io.BytesIO()
                fig_combined.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                buf.seek(0)
                frame = imageio.imread(buf)

                writer.append_data(frame)

                # Close figure to free memory
                plt.close(fig_img)
                plt.close(fig_map)
                plt.close(fig_combined)

            for fig_rgb in figs_rgb:
                # Convert figure to image data
                buf = io.BytesIO()
                fig_rgb.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                buf.seek(0)
                frame_rgb = imageio.imread(buf)

                writer_rgb.append_data(frame_rgb)

                # Close figure to free memory
                plt.close(fig_rgb)

    

            print(f"[Batch {i}]")

    writer.close()

    print("Done visualizing snippet.")

    return None

if __name__ == "__main__":
    main()
    
