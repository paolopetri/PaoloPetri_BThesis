# LLM Nav

## Overview
Welcome to the LLM Nav code repository. LLMNav is a mid-range path planner that uses depth and risk images for planning. The depth images are generated using the Depth-Anything-V2 monocular depth estimator and the risk images using semantic image segmentation.

**Author: Paolo Petri<br />
Maintainer: Paolo Petri, ppetri@ethz.ch**

## Installation
To run LLM Nav you need to install [PyTorch](https://pytorch.org/). It is recommended to use [Anaconda](https://docs.anaconda.com/anaconda/install/index.html) for installation. Please refer to the official Anaconda and PyTorch websites for detailed installation instructions.

Please refer to the INSTALL.md file in the project's root directory for instructions on setting up your llmnav environment and installing all required packages.

## Data Collection
Go to the **data_collection** folder

    cd <your_llmnav_path>/data_collection

The data collection consists of two parts. First, the data needs to be extraced from the rosbag file and then the depth images need to be generated.

### Extraction from a Rosbag File

1. **Navigate to the `data_collection` Folder**  
   ```bash
   cd <your_llmnav_path>/data_collection
   ```
   Refer to the `INSTALL.md` file in this directory to set up your `data_ext` environment.

2. **Start the Roscore**  
   In your first terminal:
   ```bash
   roscore
   ```

3. **Run `data_extraction.py`**  
   In a second terminal, activate the `data_ext` environment and run the extraction script:
   ```bash
   conda activate data_ext
   python3 data_extraction.py
   ```

4. **Play the Bag File**  
   In a third terminal, replay your bag file at a slower rate:
   ```bash
   rosbag play your_bagfile.bag --clock -r 0.3
   ```

---

#### Training Data Directory Structure

Your training data should be organized as follows:

```
TrainingData/
└── Environment1/
    ├── camera/
    ├── depth_images/
    ├── maps/
    │   ├── elevation/
    │   ├── risk/
    │   ├── traversability/
    │   ├── center_positions.txt
    │   └── world_to_grid_transforms.txt
    ├── risk_images/
    └── base_to_world_transforms.txt

└── Environment2/
    ...
```

Each environment folder contains subdirectories for `camera`, `depth_images`, and `risk_images`, plus a `maps` directory that holds `elevation/`, `risk/`, `traversability/` subfolders and related `.txt` files. This structure ensures all necessary data is in the correct place for subsequent processing or training.

        

## Training
Go to the **llmnav** folder

    cd <your_llmnav_path>/lmmnav

## Visualization


## Comparison to iPlanner
