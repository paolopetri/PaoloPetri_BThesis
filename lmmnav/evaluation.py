import torch
from torch.utils.data import DataLoader, Subset
import imageio
import io
import matplotlib.pyplot as plt

from dataset import MapDataset
from utils_viz import TrajViz, plot_traj_batch_on_map, combine_figures
from utils import prepare_data_for_plotting
from planner_net import PlannerNet
from traj_opt import TrajOpt  

def main():
    data_root = 'TrainingData/seealpsee'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1) Create dataset
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

    # Instantiate visualization
    viz = TrajViz(root_path=data_root, cameraTilt=-4.0)
    voxel_size = 0.15

    all_out_images = []

    # Open a writer for the final GIF
    final_output_path = "output/trajectory_combined_final.gif"
    writer = imageio.get_writer(final_output_path, fps=1)

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

            print(f"t_cam_to_world_SE3: {t_cam_to_world_SE3.type()}")

            # Forward pass
            preds, fear = model(depth_img, risk_img, goal_position)
            waypoints = traj_opt.TrajGeneratorFromPFreeRot(preds, step=0.5)

            # Visualization 
            list_img_depth = viz.VizImages(
                preds=preds,
                waypoints=waypoints,
                odom=t_cam_to_world_SE3,
                goal=goal_position,
                fear=fear,
                images=depth_img,
                visual_offset=0.4,
                mesh_size=0.5,
                is_shown=False
            )

            list_img_risk = viz.VizImages(
                preds=preds,
                waypoints=waypoints,
                odom=t_cam_to_world_SE3,
                goal=goal_position,
                fear=fear,
                images=risk_img,
                visual_offset=0.4,
                mesh_size=0.5,
                is_shown=False
            )

            list_img_combined = viz.combinecv(list_img_depth, list_img_risk)
            figs_img = viz.cv2fig(list_img_combined)

            start_idx, waypoints_idxs, goal_idx = prepare_data_for_plotting(
                waypoints, goal_position, center_position, grid_map, 
                t_cam_to_world_SE3, t_world_to_grid_SE3, voxel_size
            )

            figs_map = plot_traj_batch_on_map(start_idx, waypoints_idxs, goal_idx, grid_map)

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

            out_images = list_img_combined
            all_out_images.extend(out_images)

            print(f"[Batch {i}] - Processed {len(out_images)} images.")

    writer.close()

    print("Done visualizing snippet.")
    print(f"Total images collected: {len(all_out_images)}")

if __name__ == "__main__":
    main()
    
