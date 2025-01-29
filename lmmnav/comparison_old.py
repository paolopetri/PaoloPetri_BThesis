import torch
from torch.utils.data import DataLoader, Subset
import pypose as pp

from dataset import MapDataset
from utils_viz import comparison_plot_on_map, create_gif_from_figures, plot_loss_comparison
from utils import CostofTraj, prepare_data_for_plotting
from planner_net import PlannerNet
from iplanner_planner_net import iPlannerPlannerNet
from traj_opt import TrajOpt  

def main():
    data_root = 'TrainingData/Important'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = MapDataset(
        data_root=data_root,
        transform=None,
        device=device
    )

    robot_path = dataset.t_cam_to_world_SE3

    # snippet_indices = range(132 + 48, 228 - 16)
    snippet_indices = range(0, 1517 - 31)
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

    best_model_path = "checkpoints/best_model.pth"
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint)
    print(f"Loaded best model from {best_model_path}")

    model.eval()

    voxel_size = 0.15

    loss_components = ['Total Loss', 'Traversability Loss', 'Risk Loss', 'Motion Loss', 'Goal Loss']

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
            start_idx_tensor = sample['start_idx']

            preds, fear = model(depth_img, risk_img, goal_position)
            waypoints = traj_opt.TrajGeneratorFromPFreeRot(preds, step=0.5)

            num_p = waypoints.shape[1]
            desired_wp = traj_opt.TrajGeneratorFromPFreeRot(goal_position[:, None, 0:3], step=1.0/(num_p-1))

            _, _, length_x, length_y = grid_map.shape

            start_idxs, grid_idxs, goal_idxs = prepare_data_for_plotting(waypoints,
                                                                         goal_position,
                                                                         center_position,
                                                                         grid_map,
                                                                         t_cam_to_world_SE3,
                                                                         t_world_to_grid_SE3,
                                                                         voxel_size)
            
            mission_waypoints = gen_mission_waypoints(start_idx_tensor, num_p, ahead = 30, complete_path = robot_path)

            transform = pp.SE3(t_cam_to_world_SE3).unsqueeze(1)

            mission_waypoints_SE3 = pp.SE3(mission_waypoints)

            mission_waypoints_cam_SE3 = transform.Inv() @ mission_waypoints_SE3

            mission_waypoints_cam = mission_waypoints_cam_SE3.translation()
            
            _, mission_grid_idxs, _ = prepare_data_for_plotting(mission_waypoints_cam,
                                                                 goal_position,
                                                                 center_position,
                                                                 grid_map, t_cam_to_world_SE3,
                                                                 t_world_to_grid_SE3,
                                                                 voxel_size)

            figs = comparison_plot_on_map(start_idxs, grid_idxs, mission_grid_idxs, goal_idxs, grid_map, "LMM Nav", "Mission Nav")
            output_path = f"output/Comparison/Mission/GIF/{i}.gif"
            create_gif_from_figures(figs, output_path, fps = 2)
            
            
            # Calculate the trajectory cost
            total_loss, tloss, rloss, mloss, gloss, _ = CostofTraj(
                waypoints=waypoints,
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
                is_map=True
            )
            
            mission_total_loss, mission_tloss, mission_rloss, mission_mloss, mission_gloss, _ = CostofTraj(
                waypoints=mission_waypoints_cam,
                desired_wp = desired_wp,
                goals=goal_position,
                grid_maps=grid_map,
                grid_idxs=mission_grid_idxs,
                length_x=length_x,
                length_y=length_y,
                device=device,
                alpha=alpha,
                beta=beta,
                epsilon=epsilon,
                delta=delta,
                is_map=True
            )

            # Collect losses for Model 1
            model1_losses['Total Loss'].append(total_loss.item())
            model1_losses['Traversability Loss'].append(tloss.item())
            model1_losses['Risk Loss'].append(rloss.item())
            model1_losses['Motion Loss'].append(mloss.item())
            model1_losses['Goal Loss'].append(gloss.item())

            # Collect losses for Model 2
            model2_losses['Total Loss'].append(mission_total_loss.item())
            model2_losses['Traversability Loss'].append(mission_tloss.item())
            model2_losses['Risk Loss'].append(mission_rloss.item())
            model2_losses['Motion Loss'].append(mission_mloss.item())
            model2_losses['Goal Loss'].append(mission_gloss.item())

            print(f"[Batch Mission {i}] - Processed {len(figs)} images.")
    for comp in loss_components:
        plot_loss_comparison(
            model1_losses[comp],
            model2_losses[comp],
            metric_name=comp,
            model1_label="LMM Nav",
            model2_label="Mission Nav",
            save_dir = "output/Comparison/Mission/Losses"
        )
    print("Done with evaluation of LMM Nav to Mission Nav comparison.")


    # Freeing up Hardware Resources
    del robot_path, model2_losses
    # start comparison to iPlanner
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    model2_losses = { comp: [] for comp in loss_components }

    # Load the iPlanner model
    iplanner_model = iPlannerPlannerNet(16, 5).to(device)
    iplanner_model_path = "checkpoints/iplanner.pt"
    iplanner_checkpoint_tuple = torch.load(iplanner_model_path, map_location=device)
    loaded_model_instance, some_value = iplanner_checkpoint_tuple
    state_dict = loaded_model_instance.state_dict()
    iplanner_model.load_state_dict(state_dict)
    print(f"Loaded iPlanner model from {iplanner_model_path}")

    iplanner_model.eval()

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
            waypoints = traj_opt.TrajGeneratorFromPFreeRot(preds, step=0.5)

            num_p = waypoints.shape[1]
            desired_wp = traj_opt.TrajGeneratorFromPFreeRot(goal_position[:, None, 0:3], step=1.0/(num_p-1))

            _, _, length_x, length_y = grid_map.shape

            start_idxs, grid_idxs, goal_idxs = prepare_data_for_plotting(waypoints, goal_position, center_position, grid_map, t_cam_to_world_SE3, t_world_to_grid_SE3, voxel_size)

            iplanner_preds, iplanner_fear = iplanner_model(depth_img, goal_position)
            iplanner_waypoints = traj_opt.TrajGeneratorFromPFreeRot(iplanner_preds, step=0.5)

            _, iplanner_grid_idxs, _ = prepare_data_for_plotting(iplanner_waypoints, goal_position, center_position, grid_map, t_cam_to_world_SE3, t_world_to_grid_SE3, voxel_size)

            # Visualize both trajectories on the maps

            figs = comparison_plot_on_map(start_idxs, grid_idxs, iplanner_grid_idxs, goal_idxs, grid_map, "LMM Nav", "iPlanner Nav")
            output_path = f"output/Comparison/iPlanner/GIF/{i}.gif"
            create_gif_from_figures(figs, output_path, fps = 2)

            # Calculate the trajectory cost for iPlanner
            iplanner_total_loss, iplanner_tloss, iplanner_rloss, iplanner_mloss, iplanner_gloss, _ = CostofTraj(
                waypoints=iplanner_waypoints,
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
                is_map=True
            )

            # Collect losses for iPlanner
            model2_losses['Total Loss'].append(iplanner_total_loss.item())
            model2_losses['Traversability Loss'].append(iplanner_tloss.item())
            model2_losses['Risk Loss'].append(iplanner_rloss.item())
            model2_losses['Motion Loss'].append(iplanner_mloss.item())
            model2_losses['Goal Loss'].append(iplanner_gloss.item())

            print(f"[Batch iPlanner {i}] - Processed {len(figs)} images.")
    for comp in loss_components:
        plot_loss_comparison(
            model1_losses[comp],
            model2_losses[comp],
            metric_name=comp,
            model1_label="LMM Nav",
            model2_label="iPlanner Nav",
            save_dir = "output/Comparison/iPlanner/Losses"
        )

    print("Done with evaluation of LMM Nav to iPlanner Nav comparison.")






def gen_mission_waypoints(start_idx_tensor, num_p, complete_path, ahead=30):
    """
    Generate mission waypoints.

    Args:
        start_idx_tensor (torch.Tensor): Start idx for each element in the batch. Shape: [batch_size]
        num_p (int): Number of waypoints.
        ahead (int): Number of steps ahead.
        complete_path_tensor (torch.Tensor): Complete path tensor. Shape: [num_samples, 7]
    Returns:
        torch.Tensor: Mission waypoints. Shape: [batch_size, num_p, 3]
    """
    batch_size = start_idx_tensor.shape[0]
    mission_waypoints = torch.zeros(batch_size, num_p, 7, device=complete_path.device)

    for i in range(batch_size):
        start_idx = int(start_idx_tensor[i].item())

        # Compute linearly spaced indices from start_idx to start_idx + ahead
        # Ensure we generate `num_p` indices
        indices = torch.linspace(start_idx, start_idx + ahead, steps=num_p)

        # Round indices to the nearest integer and convert to long for indexing
        indices = indices.round().long()

        # Use these indices to select points from complete_path_tensor and take the first 3 columns (x, y, z)
        mission_waypoints[i] = complete_path[indices, :]

    return mission_waypoints

if __name__ == "__main__":
    main()