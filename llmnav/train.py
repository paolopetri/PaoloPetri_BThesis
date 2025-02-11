"""
train.py

This script is used to train the PlannerNet model for trajectory planning. 
It utilizes a combination of PyTorch for the neural network, wandb for logging,
and various utility functions for data processing, trajectory cost computation, and trajectory
optimization. The training loop supports checkpointing, early stopping, and configurable hyperparameters
via command-line arguments or a YAML file. 

Usage Example:
    python3 train.py --num_epochs 50 --batch_size 64 --optimizer adam

Author: [Paolo Petri]
Date: [05.02.2025]
"""
import argparse
import yaml
import torch
from torch import optim
from torch.utils.data import DataLoader, random_split, ConcatDataset
import torch.nn.functional as F

import wandb
from tqdm import tqdm
from pprint import pprint
from typing import Dict, Any

from dataset import MapDataset
from planner_net import PlannerNet
from utils import CostofTraj, TransformPoints2Grid, Pos2Ind
from traj_opt import TrajOpt

def load_config_from_yaml(yaml_path: str) -> Dict[str, Any]:
    """
    Load configuration parameters from a YAML file.

    Args:
        yaml_path (str): Path to the YAML configuration file.

    Returns:
        Dict[str, Any]: Dictionary containing key-value pairs from the YAML file.
    """
    with open(yaml_path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the training script.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Train PlannerNet with wandb sweeps.")

    # General training parameters
    parser.add_argument('--num_epochs', type=int, default=80,
                        help='Number of training epochs.')
    parser.add_argument('--patience', type=int, default=10,
                        help='Number of epochs to wait for improvement in validation loss.')
    parser.add_argument('--min_gamma', type=float, default=0,
                        help='Minimum improvement in validation loss to continue training.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training.')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of worker processes for data loading.')
    parser.add_argument('--optimizer', type=str, default='adamw',
                        help='Optimizer type (e.g., "adam", "adamw").')
    parser.add_argument('--learning_rate', type=float, default=0.0001187126946534174,
                        help='Learning rate for the optimizer.')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='Weight decay (L2 regularization).')
    parser.add_argument('--encoder_channel', type=int, default=32,
                        help='Number of channels for the encoder in PlannerNet.')
    parser.add_argument('--knodes', type=int, default=5,
                        help='Number of nodes in PlannerNet output trajectory.')

    # Additional hyperparameters
    parser.add_argument('--ahead_dist', type=float, default=2.0,
                        help='Distance ahead for fear loss.')
    parser.add_argument('--trav_threshold', type=float, default=0.9,
                        help='Traversability threshold for fear loss.')
    parser.add_argument('--risk_threshold', type=float, default=0.5,
                        help='Risk threshold for fear loss.')
    parser.add_argument('--fear_weight', type=float, default=1.0,
                        help='Weight factor for fear loss.')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Weight factor for traversability loss.')
    parser.add_argument('--beta', type=float, default=8.0,
                        help='Weight factor for risk loss.')
    parser.add_argument('--delta', type=float, default=1.8,
                        help='Weight factor for goal loss.')
    parser.add_argument('--epsilon', type=float, default=0.4,
                        help='Weight factor for motion loss.')
    parser.add_argument('--zeta', type=float, default=1.0,
                        help='Weight factor for height loss.')

    # Flag to optionally load best config
    parser.add_argument('--use_best_config', action='store_true',
                        help="Flag to override args with best configuration from YAML.")

    return parser.parse_args()

def main() -> None:
    """
    Main function to orchestrate the training process. Parses arguments, initializes wandb, 
    loads dataset, creates DataLoaders, sets up the PlannerNet model and optimizer, 
    then runs the training and validation loops with checkpointing and early stopping.
    
    Returns:
        None
    """
    # Initialize wandb
    args = parse_args()

    if args.use_best_config:
        best_config = load_config_from_yaml("config/best_config_32_64_4.yaml")
        for key, value in best_config.items():
            setattr(args, key, value)

    wandb.init(
        project="LLM_Nav",
        config = {
            "num_epochs": args.num_epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "optimizer": args.optimizer,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "encoder_channel": args.encoder_channel,
            "knodes": args.knodes,
            "ahead_dist": args.ahead_dist,
            "trav_threshold": args.trav_threshold,
            "risk_threshold": args.risk_threshold,
            "fear_weight": args.fear_weight,
            "alpha": args.alpha,
            "beta": args.beta,
            "epsilon": args.epsilon,
            "delta": args.delta,
            "zeta": args.zeta,
            "min_gamma": args.min_gamma
            }
    )
    config = wandb.config
    print("Training script started with the following configuration:")
    pprint(dict(config), sort_dicts=False)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    voxel_size = 0.15
    step = 0.1


    # Training and Hyperparameters
    num_epochs = config.num_epochs
    batch_size = config.batch_size
    num_workers = config.num_workers
    optimizer_type = config.optimizer
    learning_rate = config.learning_rate
    weight_decay = config.weight_decay
    encoder_channel = config.encoder_channel
    knodes = config.knodes
    ahead_dist = config.ahead_dist
    trav_threshold = config.trav_threshold
    risk_threshold = config.risk_threshold
    fear_weight = config.fear_weight
    alpha = config.alpha
    beta = config.beta
    epsilon = config.epsilon
    delta = config.delta
    zeta = config.zeta
    min_gamma = config.min_gamma

    # Initialize the dataset
    # TODO: Update with your Training Environments
    höngg_data = MapDataset(data_root='TrainingData/Hönggerberg', random_goals=True)
    seealpsee_data = MapDataset(data_root='TrainingData/seealpsee', random_goals=True)
    in2out1_data = MapDataset(data_root='TrainingData/in-to-out-1', random_goals=True)
    # important_data = MapDataset(data_root='TrainingData/Important', transform=transform)
    full_dataset = ConcatDataset([höngg_data, seealpsee_data, in2out1_data])

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

    if optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)


    # Watch the model with wandb
    wandb.watch(model, log="all")

    # Initialize variables for checkpointing and early stopping
    best_val_loss = float('inf')
    patience = config.patience
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
            total_loss, tloss, rloss, mloss, gloss, hloss, fear_labels = CostofTraj(
                waypoints=waypoints,
                waypoints_grid=transformed_waypoints,
                desired_wp = desired_wp,
                goals=goal_position,
                grid_maps=grid_map,
                grid_idxs=grid_idxs,
                length_x=length_x,
                length_y=length_y,
                device=device,
                ahead_dist=ahead_dist,
                trav_threshold=trav_threshold,
                risk_threshold=risk_threshold,
                alpha=alpha,
                beta=beta,
                epsilon=epsilon,
                delta=delta,
                zeta=zeta,
                is_map=True
            )

            fear_loss = F.binary_cross_entropy(fear, fear_labels)

            loss = total_loss + fear_weight * fear_loss
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()

            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5

            wandb.log({"Epoch": epoch+1, "Total Gradient Norm": total_norm})

            optimizer.step()

            running_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())

        # Update learning rate
        scheduler.step()

        # Calculate average training loss for the epoch
        avg_train_loss = running_loss / len(train_loader)
        wandb.log({"Epoch": epoch+1, "Training Loss": avg_train_loss})

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_tloss = 0.0
        val_rloss = 0.0
        val_mloss = 0.0
        val_gloss = 0.0
        val_floss = 0.0
        val_hloss = 0.0


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
                total_loss, tloss, rloss, mloss, gloss, hloss, fear_labels = CostofTraj(
                    waypoints=waypoints,
                    waypoints_grid=transformed_waypoints,
                    desired_wp = desired_wp,
                    goals=goal_position,
                    grid_maps=grid_map,
                    grid_idxs=grid_idxs,
                    length_x=length_x,
                    length_y=length_y,
                    device=device,
                    ahead_dist=ahead_dist,
                    trav_threshold=trav_threshold,
                    risk_threshold=risk_threshold,
                    alpha=alpha,
                    beta=beta,
                    epsilon=epsilon,
                    delta=delta,
                    zeta=zeta,
                    is_map=True
                )
                floss = F.binary_cross_entropy(fear, fear_labels)

                loss = total_loss + fear_weight * floss

                val_loss += loss.item()
                val_tloss += tloss.item()
                val_rloss += rloss.item()
                val_mloss += mloss.item()
                val_gloss += gloss.item()
                val_floss += floss.item()
                val_hloss += hloss.item()

                val_bar.set_postfix(loss=loss.item())


        # Calculate average validation loss for the epoch
        avg_val_tloss = val_tloss / len(val_loader)
        avg_val_rloss = val_rloss / len(val_loader)
        avg_val_mloss = val_mloss / len(val_loader)
        avg_val_gloss = val_gloss / len(val_loader)
        avg_val_floss = val_floss / len(val_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_val_hloss = val_hloss / len(val_loader)
        
        wandb.log({"Epoch": epoch+1,
                   "Traversability Loss": avg_val_tloss * alpha,
                   "Risk Loss": avg_val_rloss * beta,
                   "Motion Loss": avg_val_mloss * epsilon,
                   "Goal Loss": avg_val_gloss * delta,
                   "Fear Loss": avg_val_floss * fear_weight,
                   "Height Loss": avg_val_hloss * zeta,
                   "Validation Loss": avg_val_loss})

        print(f"Epoch [{epoch+1}/{num_epochs}] Training Loss: {avg_train_loss:.4f} Validation Loss: {avg_val_loss:.4f}")

        # Checkpointing: Save the model if validation loss has decreased
        if best_val_loss - avg_val_loss > min_gamma:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "checkpoints/best_model.pth")
            wandb.save("checkpoints/best_model.pth")
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