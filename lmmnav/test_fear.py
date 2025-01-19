import torch
from torch.utils.data import DataLoader, Subset

from dataset import MapDataset
from utils_viz import plot_traj_batch_on_map
from utils import prepare_data_for_plotting
from planner_net import PlannerNet
from traj_opt import TrajOpt  

def main():
    data_root = 'TrainingData/seealpsee'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = MapDataset(
        data_root=data_root,
        transform=None,
        device=device
    )

    snippet_indices = range(350, 382)
    subset_dataset = Subset(dataset, snippet_indices)

    viz_loader = DataLoader(
        subset_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0
    )

    # 4) Load your model
    model = PlannerNet(16, 5).to(device)
    traj_opt = TrajOpt()

    best_model_path = "checkpoints/best_model.pth"
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint)
    print(f"Loaded best model from {best_model_path}")

    model.eval()

    voxel_size = 0.15

    with torch.no_grad():
        for i, sample in enumerate(viz_loader):
            # Data preparation as before
            grid_map = sample['grid_map'].to(device)
            center_position = sample['center_position'].to(device)
            goal_position = sample['goal_positions'].to(device)
            t_cam_to_world_SE3 = sample['t_cam_to_world_SE3'].to(device)
            t_world_to_grid_SE3 = sample['t_world_to_grid_SE3'].to(device)
            depth_img, risk_img = sample['image_pair']
            depth_img = depth_img.to(device)
            risk_img = risk_img.to(device)

            batch_size = center_position.shape[0]

            goal_position = gen_random_goal_positions(batch_size)

            # Forward pass
            preds, fear = model(depth_img, risk_img, goal_position)
            waypoints = traj_opt.TrajGeneratorFromPFreeRot(preds, step=0.5)

            print(f"Predictions: {preds}")

            start_idx, waypoints_idxs, goal_idx = prepare_data_for_plotting(
                waypoints, goal_position, center_position, grid_map, 
                t_cam_to_world_SE3, t_world_to_grid_SE3, voxel_size
            )

            figs_map = plot_traj_batch_on_map(start_idx, waypoints_idxs, goal_idx, grid_map)


if __name__ == "__main__":
    main()

def gen_random_goal_positions(batch_size, max_distance=12):
    """
    Generate random goal positions in the camera frame.

    The camera looks in the -x direction. Goals will be randomly placed 
    within a maximum distance.

    Args:
        batch_size (int): Number of goal positions to generate.
        max_distance (float): Maximum distance from the origin (in the -x direction).

    Returns:
        torch.Tensor: Tensor of shape (batch_size, 3) with goal positions in the camera frame.
    """
    # Generate random distances in the range [0, max_distance]
    distances = torch.rand(batch_size) * max_distance

    # Generate random angles within 90 degrees to either side of the -x direction (in radians)
    angles = torch.rand(batch_size) * torch.pi - torch.pi / 2

    # Compute goal positions in the camera frame
    x = -distances  # Move along the -x axis
    y = distances * torch.sin(angles)  # Randomly offset in the y direction
    z = torch.zeros(batch_size)  # Assume the ground plane (z = 0)

    # Combine into a (batch_size, 3) tensor
    goal_positions = torch.stack((x, y, z), dim=1)
    return goal_positions
