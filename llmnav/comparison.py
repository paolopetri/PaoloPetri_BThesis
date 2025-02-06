"""
comparison.py

This script compares the performance of two trajectory planning models, the "LLM Nav" (PlannerNet)
and the "iPlanner Nav" (iPlannerPlannerNet). It calculates several cost components for each model,
visualizes their trajectories on an occupancy grid, and generates comparison plots and GIFs. 
Finally, it computes average and standard deviation statistics for each cost component and prints
a table of results.

Usage Example:
    python comparison.py

Author: [Paolo Petri]
Date: [07.02.2025]
"""
import os

import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd
from typing import NoReturn

from dataset import MapDataset
from utils_viz import comparison_plot_on_map, create_gif_from_figures, plot_loss_comparison
from utils import CostofTraj, prepare_data_for_plotting, TransformPoints2Grid
from planner_net import PlannerNet
from iplanner_planner_net import iPlannerPlannerNet
from traj_opt import TrajOpt  


def main() -> NoReturn:
    """
    Main function to compare trajectory costs between LLM Nav (PlannerNet) and iPlanner Nav
    (iPlannerPlannerNet).

    Steps:
        1. Loads a dataset from the specified root directory.
        2. Defines a subset of indices to evaluate.
        3. Initializes and loads checkpoints for both PlannerNet and iPlannerPlannerNet.
        4. Iterates over the data to compute trajectory predictions for each model.
        5. Calculates various cost components (Traversability, Risk, Motion, Goal, Height) 
           for both models.
        6. Visualizes and saves comparison plots and GIFs of both trajectories on the occupancy map.
        7. Aggregates and logs average and standard deviation of cost components, then prints 
           a formatted comparison table.

    Returns:
        None
    """
    # TODO: Update data_root to the correct path
    data_root = 'TrainingData/Important'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = MapDataset(
        data_root=data_root,
        random_goals=False,
        device=device
    )

    # TODO: Update snippet_indices to the desired range of snippets to evaluate
    # snippet_indices = range(132 + 34, 228 - 30)
    # snippet_indices = range(0, 1517 - 31)
    snippet_indices = range(220, 252)
    subset_dataset = Subset(dataset, snippet_indices)

    comp_loader = DataLoader(
        subset_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    model = PlannerNet(32, 5).to(device)
    traj_opt = TrajOpt()
    alpha = 1.0
    beta = 1.0
    epsilon = 1.0
    delta = 1.0
    zeta = 1.0

    best_model_path = "checkpoints/best_model_smallpermutation.pth" # does not work with planning in the camera frame!
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

    voxel_size = 0.15

    loss_components = ['Total Trajectory Cost', 'Traversability Cost', 'Risk Cost', 'Motion Cost', 'Goal Cost', 'Height Cost']

    model1_losses = { comp: [] for comp in loss_components }
    model2_losses = { comp: [] for comp in loss_components }


    with torch.no_grad():
        for i, sample in enumerate(comp_loader):

            grid_map = sample['grid_map'].to(device)
            center_position = sample['center_position'].to(device)
            t_cam_to_world_SE3 = sample['t_cam_to_world_SE3'].to(device)
            goal_position = sample['goal_positions'].to(device)
            t_world_to_grid_SE3 = sample['t_world_to_grid_SE3'].to(device)
            depth_img, risk_img = sample['image_pair']
            depth_img = depth_img.to(device)
            risk_img = risk_img.to(device)
            

            preds, fear = model(depth_img, risk_img, goal_position)
            waypoints = traj_opt.TrajGeneratorFromPFreeRot(preds, step=0.1)

            num_p = waypoints.shape[1]
            desired_wp = traj_opt.TrajGeneratorFromPFreeRot(goal_position[:, None, 0:3], step=1.0/(num_p-1))

            _, _, length_x, length_y = grid_map.shape

            transformed_waypoints = TransformPoints2Grid(waypoints, t_cam_to_world_SE3, t_world_to_grid_SE3)

            start_idxs, grid_idxs, goal_idxs = prepare_data_for_plotting(waypoints,
                                                                         goal_position,
                                                                         center_position,
                                                                         grid_map,
                                                                         t_cam_to_world_SE3,
                                                                         t_world_to_grid_SE3,
                                                                         voxel_size)
            
            # Calculate the trajectory cost
            total_loss, tloss, rloss, mloss, gloss, hloss, _ = CostofTraj(
                waypoints=waypoints,
                waypoints_grid=transformed_waypoints,
                desired_wp = desired_wp,
                goals=goal_position,
                grid_maps=grid_map,
                grid_idxs=grid_idxs,
                length_x=length_x,
                length_y=length_y,
                device=device,
                alpha=alpha,
                beta=beta,
                epsilon=epsilon,
                delta=delta,
                zeta=zeta,
                is_map=True
            )

            iplanner_preds, iplanner_fear = iplanner_model(depth_img, goal_position)
            iplanner_waypoints = traj_opt.TrajGeneratorFromPFreeRot(iplanner_preds, step=0.1)

            transformed_iplanner_waypoints = TransformPoints2Grid(iplanner_waypoints, t_cam_to_world_SE3, t_world_to_grid_SE3)
            _, iplanner_grid_idxs, _ = prepare_data_for_plotting(iplanner_waypoints, goal_position, center_position, grid_map, t_cam_to_world_SE3, t_world_to_grid_SE3, voxel_size)

            # Visualize both trajectories on the maps

            figs = comparison_plot_on_map(start_idxs, grid_idxs, iplanner_grid_idxs, goal_idxs, grid_map, "LMM Nav", "iPlanner Nav")
            output_path = f"output/Comparison/GIF/{i}.gif"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            create_gif_from_figures(figs, output_path, fps = 2)

            # Calculate the trajectory cost for iPlanner
            iplanner_total_loss, iplanner_tloss, iplanner_rloss, iplanner_mloss, iplanner_gloss, iplanner_hloss, _ = CostofTraj(
                waypoints=iplanner_waypoints,
                waypoints_grid=transformed_iplanner_waypoints,
                desired_wp = desired_wp,
                goals=goal_position,
                grid_maps=grid_map,
                grid_idxs=iplanner_grid_idxs,
                length_x=length_x,
                length_y=length_y,
                device=device,
                alpha=alpha,
                beta=beta,
                epsilon=epsilon,
                delta=delta,
                zeta=zeta,
                is_map=True
            )
    
            # Collect losses for Model 1
            model1_losses['Total Trajectory Cost'].append(total_loss.item())
            model1_losses['Traversability Cost'].append(tloss.item())
            model1_losses['Risk Cost'].append(rloss.item())
            model1_losses['Motion Cost'].append(mloss.item())
            model1_losses['Goal Cost'].append(gloss.item())
            model1_losses['Height Cost'].append(hloss.item())

            # Collect losses for iPlanner
            model2_losses['Total Trajectory Cost'].append(iplanner_total_loss.item())
            model2_losses['Traversability Cost'].append(iplanner_tloss.item())
            model2_losses['Risk Cost'].append(iplanner_rloss.item())
            model2_losses['Motion Cost'].append(iplanner_mloss.item())
            model2_losses['Goal Cost'].append(iplanner_gloss.item())
            model2_losses['Height Cost'].append(iplanner_hloss.item())


            print(f"[Batch iPlanner {i}] - Processed {len(figs)} images.")
    for comp in loss_components:
        save_dir = "output/Comparison/Losses"
        os.makedirs(save_dir, exist_ok=True)
        plot_loss_comparison(
            model1_losses[comp],
            model2_losses[comp],
            metric_name=comp,
            model1_label="LLM Nav",
            model2_label="iPlanner",
            save_dir = save_dir
        )
        results = {
        "LLM Nav": {},
        "iPlanner": {}
    }

    for comp in loss_components:
        avg_llm_nav = np.mean(model1_losses[comp])
        std_llm_nav = np.std(model1_losses[comp])
        avg_iplanner = np.mean(model2_losses[comp])
        std_iplanner = np.std(model2_losses[comp])
        
        results["LLM Nav"][comp] = f"{avg_llm_nav:.2f} ± {std_llm_nav:.2f}"
        results["iPlanner"][comp] = f"{avg_iplanner:.2f} ± {std_iplanner:.2f}"

    # Create a DataFrame with cost items as columns and models as rows
    df = pd.DataFrame(results).transpose()
    df = df[loss_components]  # Order the columns

    print("Trajectory Cost Comparison (Average ± Std):")
    print(df.to_string())
    # Optional: Create a LaTeX version:
    # print(df.to_latex())
    # print("Done with evaluation of LMM Nav to iPlanner Nav comparison.")

if __name__ == "__main__":
    main()