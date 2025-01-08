import torch
from torch import optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import wandb
from tqdm import tqdm
# Import your dataset and other modules
from dataset import MapDataset
from planner_net import PlannerNet
from utils import CostofTraj, TransformPoints2Grid, Pos2Ind, create_gif_from_batch
from utils_viz import plot_trajectory_on_images
import matplotlib.pyplot as plt
from traj_opt import TrajOpt

def main():
    print("Training script started.")
    # Initialize wandb
    wandb.init(
        project="navigation_model",
        config={
            "num_epochs": 20,
            "batch_size": 32,
            "num_workers": 8,
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
            "encoder_channel": 16,
            "knodes": 5,
            "step": 1.0,
            "voxel_size": 0.15,
            "alpha": 0.5,
            "beta": 1.0,
            "epsilon": 1.0,
            "delta": 8.0
        }
    )
    config = wandb.config

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hyperparameters
    num_epochs = config.num_epochs
    batch_size = config.batch_size
    num_workers = config.num_workers
    learning_rate = config.learning_rate
    weight_decay = config.weight_decay
    encoder_channel = config.encoder_channel
    knodes = config.knodes
    step = config.step
    voxel_size = config.voxel_size
    alpha = config.alpha
    beta = config.beta
    epsilon = config.epsilon
    delta = config.delta

    # Initialize the dataset with transformations
    transform = transforms.Compose([
        transforms.Resize((360, 640)),
        transforms.ToTensor()
    ])
    full_dataset = MapDataset(data_root='TrainingData', transform=transform)

    # Define validation split ratio
    val_split = 0.3
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size

    # Split the dataset
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    # Initialize the model, optimizer, and scheduler
    model = PlannerNet(encoder_channel, knodes).to(device)
    traj_opt = TrajOpt()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Watch the model with wandb
    wandb.watch(model, log="all")

    # Initialize variables for checkpointing and early stopping
    best_val_loss = float('inf')
    patience = 5
    trigger_times = 0

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # Wrap the training DataLoader with tqdm for progress bar
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]")
        for batch_idx, sample in enumerate(train_bar):
            # Move data to GPU
            grid_map = sample['grid_map'].to(device)                # (batch, 2, 266, 266)
            center_position = sample['center_position'].to(device)  # (batch, 2)
            t_world_to_grid_SE3 = sample['t_world_to_grid_SE3'].to(device)  # (batch, 7)
            t_cam_to_world_SE3 = sample['t_cam_to_world_SE3'].to(device)  # (batch, 7)
            depth_img, risk_img = sample['image_pair']              
            depth_img = depth_img.to(device)
            risk_img = risk_img.to(device)                
            goal_position = sample['goal_positions'].to(device)      # (batch, max_episodes, 3)

            # Forward pass
            preds, fear = model(depth_img, risk_img, goal_position)  # (batch, num_waypoints, 3)
            waypoints = traj_opt.TrajGeneratorFromPFreeRot(preds, step=step)  # (batch, num_waypoints, 3)

            # For Motion loss
            _, num_p, _ = waypoints.shape
            desired_wp = traj_opt.TrajGeneratorFromPFreeRot(goal_position[:, None, 0:3], step=1.0/(num_p-1))

            _, _, length_x, length_y = grid_map.shape

            transformed_waypoints = TransformPoints2Grid(waypoints, t_cam_to_world_SE3, t_world_to_grid_SE3)  # (batch, num_waypoints, 3)
            grid_idxs = Pos2Ind(transformed_waypoints, length_x, length_y, center_position, voxel_size, device)  # (batch, num_waypoints)

            # Calculate the trajectory cost
            total_loss, tloss, rloss, mloss, gloss = CostofTraj(
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

            # Backward and optimize
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
            train_bar.set_postfix(loss=total_loss.item())

            wandb.log({
                "train_total_loss": total_loss.item(),
                "train_traversability_loss": tloss.item(),
                "train_risk_loss": rloss.item(),
                "train_motion_loss": mloss.item(),
                "train_goal_loss": gloss.item()
            })

        # Update learning rate
        scheduler.step()

        # Calculate average training loss for the epoch
        avg_train_loss = running_loss / len(train_loader)
        wandb.log({"epoch": epoch+1, "train_loss": avg_train_loss})

        # Validation phase
        model.eval()
        val_loss = 0.0

        # Disable gradient computation for validation
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]")
            for sample in val_bar:
                # Move data to GPU
                grid_map = sample['grid_map'].to(device)                # (batch, 2, 266, 266)
                center_position = sample['center_position'].to(device)  # (batch, 2)
                t_world_to_grid_SE3 = sample['t_world_to_grid_SE3'].to(device)  # (batch, 7)
                t_cam_to_world_SE3 = sample['t_cam_to_world_SE3'].to(device)  # (batch, 7)
                depth_img, risk_img = sample['image_pair']              
                depth_img = depth_img.to(device)
                risk_img = risk_img.to(device)                
                goal_position = sample['goal_positions'].to(device)      # (batch, max_episodes, 3)


                # Forward pass
                preds, fear = model(depth_img, risk_img, goal_position)  # (batch, num_waypoints, 3)
                waypoints = traj_opt.TrajGeneratorFromPFreeRot(preds, step=step)  # (batch, num_waypoints, 3)
                waypoints = waypoints.to(device)

                # For Motion loss
                _, num_p, _ = waypoints.shape
                desired_wp = traj_opt.TrajGeneratorFromPFreeRot(goal_position[:, None, 0:3], step=1.0/(num_p-1))

                _, _, length_x, length_y = grid_map.shape

                transformed_waypoints = TransformPoints2Grid(waypoints, t_cam_to_world_SE3, t_world_to_grid_SE3)  # (batch, num_waypoints, 3)
                grid_idxs = Pos2Ind(transformed_waypoints, length_x, length_y, center_position, voxel_size, device)  # (batch, num_waypoints)

                # Calculate the trajectory cost
                total_loss, tloss, rloss, mloss, gloss = CostofTraj(
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

                val_loss += total_loss.item()
                val_bar.set_postfix(loss=total_loss.item())

                wandb.log({
                    "validation_total_loss": total_loss.item(),
                    "validation_traversability_loss": tloss.item(),
                    "validation_risk_loss": rloss.item(),
                    "validation_motion_loss": mloss.item(),
                    "validation_goal_loss": gloss.item()
                })

                # For wandb, plot the waypoints and trajectory



            gif_path = create_gif_from_batch(
            grid_map,
            center_position,
            t_world_to_grid_SE3,
            t_cam_to_world_SE3,
            goal_position,
            waypoints,
            grid_idxs,
            batch_size=grid_map.shape[0],
            device=device,
            voxel_size=voxel_size,
            length_x=length_x,
            length_y=length_y,
            fps=1,
            gif_name="trajectory.gif"
            )

            figs = plot_trajectory_on_images(
                waypoints_cam=waypoints,
                depth=depth_img,
                risk=risk_img
            )

            # -------------------------------------------------
            # 3) Show them without saving
            # -------------------------------------------------
            # If you're in a GUI/desktop environment:
            for fig in figs:
                fig.show()  # Attempt to open each figure in a new window

            plt.show()  # Ensures all windows actually pop up

            # Log to wandb
            wandb.log({"Trajectory GIF": wandb.Video(gif_path, format="gif")})




        # Calculate average validation loss for the epoch
        avg_val_loss = val_loss / len(val_loader)
        wandb.log({"epoch": epoch+1, "validation_loss": avg_val_loss})

        print(f"Epoch [{epoch+1}/{num_epochs}] Training Loss: {avg_train_loss:.4f} Validation Loss: {avg_val_loss:.4f}")

        # Checkpointing: Save the model if validation loss has decreased
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pth")
            wandb.save("best_model.pth")
            trigger_times = 0
            print("Validation loss decreased, saving model.")
        else:
            trigger_times += 1
            print(f"No improvement in validation loss for {trigger_times} epoch(s).")
            if trigger_times >= patience:
                print("Early stopping triggered.")
                break

if __name__ == '__main__':
    main()