import torch
from torch.utils.data import DataLoader, Subset
import imageio
import io
import matplotlib.pyplot as plt

import numpy as np
import os
import pypose as pp

from dataset import MapDataset
from utils_viz import plot_single_waypoints_on_rgb, plot_waypoints_on_depth_risk, comparison_plot_on_map, plot_traj_batch_on_map, combine_figures
from utils import prepare_data_for_plotting
from planner_net import PlannerNet
from iplanner_planner_net import iPlannerPlannerNet
from traj_opt import TrajOpt  

def main():
    data_root = 'TrainingData/Important'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1) Create dataset
    dataset = MapDataset(
        data_root=data_root,
        transform=None,
        device=device
    )

    # snippet_indices = range(190, 194)
    snippet_indices = range(220, 221)
    subset_dataset = Subset(dataset, snippet_indices)

    viz_loader = DataLoader(
        subset_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    # 4) Load your model
    model = PlannerNet(32, 5).to(device)
    traj_opt = TrajOpt()

    best_model_path = "checkpoints/offset_32.pth"
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint)
    print(f"Loaded best model from {best_model_path}")
    model.eval()

    iplanner_model = iPlannerPlannerNet(16, 5).to(device)
    iplanner_model_path = "checkpoints/iplanner.pt"
    iplanner_checkpoint_tuple = torch.load(iplanner_model_path, map_location=device)
    loaded_model_instance, some_value = iplanner_checkpoint_tuple
    state_dict = loaded_model_instance.state_dict()
    iplanner_model.load_state_dict(state_dict)
    print(f"Loaded iPlanner model from {iplanner_model_path}")
    iplanner_model.eval()

    # Instantiate visualization
    voxel_size = 0.15

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
            idx = sample['start_idx'].to(device)
            print(f"Goal position: {goal_position}")

            goal_position[:,1] += 5.0

            batch_size = depth_img.size(0)

            # Forward pass
            preds, fear = model(depth_img, risk_img, goal_position)
            waypoints = traj_opt.TrajGeneratorFromPFreeRot(preds, step=1.0)
            
            #plot_single_waypoints_on_rgb(waypoints, goal_position, idx, model_name='LLMNav', frame = 'iPlanner', show=True, save=True)

            iplanner_preds, iplanner_fear = iplanner_model(depth_img, goal_position)
            iplanner_waypoints = traj_opt.TrajGeneratorFromPFreeRot(iplanner_preds, step=1.0)

            #plot_single_waypoints_on_rgb(iplanner_waypoints, goal_position, idx, model_name = 'iPlanner', frame = 'iPlanner', show=True, save=True)

            start_idxs, grid_idxs, goal_idxs = prepare_data_for_plotting(waypoints,
                                                                         goal_position,
                                                                         center_position,
                                                                         grid_map,
                                                                         t_cam_to_world_SE3,
                                                                         t_world_to_grid_SE3,
                                                                         voxel_size)
            
            _, iplanner_grid_idxs, _ = prepare_data_for_plotting(iplanner_waypoints,
                                                                                                     goal_position,
                                                                                                     center_position,
                                                                                                     grid_map,
                                                                                                     t_cam_to_world_SE3,
                                                                                                     t_world_to_grid_SE3,
                                                                                                     voxel_size)
            
            
            # figs_comparison = comparison_plot_on_map(start_idxs, grid_idxs, iplanner_grid_idxs, goal_idxs, grid_map, "LMMNav", "iPlanner")

            figs_depth_risk = plot_waypoints_on_depth_risk(waypoints, goal_position, depth_img, risk_img, idx, model_name='LMMNav', frame='iPlanner', show=True, save=True)

            plt.show()


            print(f"[Batch {i}] - Processed {len(batch_size)} images.")


    print("Done visualizing snippet.")

# def transform_goal_to_iplanner(goal_position, t_cam_to_world_SE3, idx, data_root, device):
#     # Define the fixed transformation from iPlanner to base
#     t_iplanner_to_base = pp.SE3([0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1]).to(device)

#     # Construct the file path
#     filepath = os.path.join(data_root, 't_cam_to_world.txt')
    
#     try:
#         # Use np.loadtxt with skiprows and max_rows to read only the desired line
#         # Assuming idx=0 corresponds to the first line in the file
#         t_base_to_world = np.loadtxt(filepath, delimiter=',', skiprows=idx, max_rows=1)
#     except IndexError:
#         raise ValueError(f"Index {idx} is out of bounds for the transforms file.")
#     except Exception as e:
#         raise IOError(f"An error occurred while reading the transform file: {e}")

#     # Convert the loaded transform to a Torch tensor
#     t_base_to_world_tensor = torch.tensor(t_base_to_world, dtype=torch.float32, device=device)
    
#     # Create the SE3 transformation
#     t_base_to_world_SE3 = pp.SE3(t_base_to_world_tensor)

#     t_iplanner_to_world = t_base_to_world_SE3 @ t_iplanner_to_base

#     t_cam_to_world = pp.SE3(t_cam_to_world_SE3)

#     t_cam_to_iplanner = t_iplanner_to_world.Inv() @ t_cam_to_world

#     iplanner_goal_position = t_cam_to_iplanner @ goal_position

#     return iplanner_goal_position


if __name__ == "__main__":
    main()
    
