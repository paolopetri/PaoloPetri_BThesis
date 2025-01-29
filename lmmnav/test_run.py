import torch
from torch.utils.data import DataLoader, Subset
import imageio
import io
import matplotlib.pyplot as plt

from dataset import MapDataset
from utils_viz import plot_traj_batch_on_map, combine_figures, plot_waypoints_on_depth_risk, plot_single_waypoints_on_rgb
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
    # snippet_indices = range(1356, 1388)
    snippet_indices = range(132 + 34, 228 - 30)
    subset_dataset = Subset(dataset, snippet_indices)

    viz_loader = DataLoader(
        subset_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0
    )

    # 4) Load your model
    traj_opt = TrajOpt()
    # model = PlannerNet(32, 5).to(device)
    # best_model_path = "checkpoints/best_model.pth"
    # checkpoint = torch.load(best_model_path, map_location=device)
    # model.load_state_dict(checkpoint)
    # print(f"Loaded best model from {best_model_path}")

    # model.eval()

    iplanner_model = iPlannerPlannerNet(16, 5).to(device)
    iplanner_model_path = "checkpoints/iplanner.pt"
    iplanner_checkpoint_tuple = torch.load(iplanner_model_path, map_location=device)
    loaded_model_instance, some_value = iplanner_checkpoint_tuple
    state_dict = loaded_model_instance.state_dict()
    iplanner_model.load_state_dict(state_dict)
    print(f"Loaded iPlanner model from {iplanner_model_path}")
    iplanner_model.eval()

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

            goal_position[:, 1] += -0.5

            # Forward pass
            preds, fear = iplanner_model(depth_img, goal_position)
            waypoints = traj_opt.TrajGeneratorFromPFreeRot(preds, step=1.0)

            figs_img = plot_waypoints_on_depth_risk(waypoints, goal_position, depth_img, risk_img, idx, model_name='iPlanner', frame='iPlanner', show=False, save=False)

            start_idx, waypoints_idxs, goal_idx = prepare_data_for_plotting(
                waypoints, goal_position, center_position, grid_map, 
                t_cam_to_world_SE3, t_world_to_grid_SE3, voxel_size
            )

            figs_map = plot_traj_batch_on_map(start_idx, waypoints_idxs, goal_idx, grid_map)

            figs_rgb =plot_single_waypoints_on_rgb(waypoints, goal_position, idx, model_name='iPlanner', frame='iPlanner', show=False, save=True)

            # Combine figures and write frames directly to GIF
            for fig_img, fig_map in zip(figs_img, figs_map):
                fig_combined = combine_figures(fig_img, fig_map)

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

if __name__ == "__main__":
    main()
    
