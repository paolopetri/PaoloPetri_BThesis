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

3. **Play the Bag File**  
   In a second terminal, replay your bag file at a slower rate:
   ```bash
   rosbag play your_bagfile.bag --clock -r 0.3
   ```

4. **Run `data_extraction.py`**  
   Shortly after step 3, in a third terminal, activate the `data_ext` environment and run the extraction script:
   ```bash
   conda activate data_ext
   python3 data_extraction.py
   ```

5. **Saving EnvironmentData**
    You can now save the `/EnvironmentData` folder into the `./llmnav/TrainingData` folder and rename it acoordingly. If you have another rosbag
    file repeat the process.

### Depth Image Generation

Go to the [**Depth-Anything-V2**](https://github.com/DepthAnything/Depth-Anything-V2) GitHub page, clone the repository and follow the installation requirements. Copy the `<your_llmnav_path>/data_collection/depth_generation.py` file into the Depth-Anything-V2 root folder. In that same root folder create a folder called checkpoints and copy the Depth-Anything-V2-Large model, that you can download form their GitHub page into the checkpoints folder.
Copy the `EnvironmentData/camera` folder into the depth-anything-v2 root folder. Activate your depth-anything-v2 conda environment and run python3 depth_generation.py. Finally, copy the depth_images folder that has all the depth images, back into the EnvionmentData/ folder.

### Depth Image Generation

1. **Clone the Depth-Anything-V2 Repository**  
   Visit the [Depth-Anything-V2 GitHub page](https://github.com/DepthAnything/Depth-Anything-V2) and clone the repository to your local machine.

2. **Follow the Installation Guide**  
   Make sure to follow the installation guide provided in the Depth-Anything-V2 repository to set up all required dependencies.

3. **Copy the Generation Script**  
   Copy the `depth_generation.py` file from `<your_llmnav_path>/data_collection` into the root directory of the cloned Depth-Anything-V2 repository.

4. **Set Up Model Checkpoints**  
   In the Depth-Anything-V2 repository, create a folder named `checkpoints`.  
   Download the Depth-Anything-V2-Large model from their GitHub page and place the model files into the `checkpoints` folder.

5. **Copy the Camera Data**  
   Copy the `camera` folder from `<your_llmnav_path>/data_collection/EnvironmentData/camera` into the root directory of the Depth-Anything-V2 repository.

6. **Run the Depth Generation Script**  
   Activate your Depth-Anything-V2 conda environment and execute the script:
   ```bash
   conda activate <depth_env>
   python3 depth_generation.py

7. **Retrieve Generated Depth Images**
   Once the script completes, copy the newly generated `depth_images` folder from the Depth-Anything-V2 repository back into the `EnvironmentData` folder within you LLM Nav directory.

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

## Training
Go to the **llmnav** folder

    cd <your_llmnav_path>/lmmnav
#### Last Modifications
Before the training scipt is able to run two little code adaptations have to be made.

1.  **Update the static transform**
    In line 250 of the `dataset.py` file, follow the TODO and update the transform based on the actual camera-to-base link calibration.

2.  **Define your Trainingdata**
    In line 177 of the `train.py` file, follow the TODO and update the following lines with your training environments.

You can now run the training script by using the best configuration

```
conda activate llmnav
python3 train.py --use_best_config
```

or you can use your own configuartion. Follow:

```
python3 train.py --help
```
The best performing model will automatically be saved as `checkpoints/best_model.pth

## Visualization
Go to the **llmnav** folder

    cd <your_llmnav_path>/lmmnav

To visualize the trajectories run the following commands:

```
conda activae llmnav
python3 test_run.py
```

You will find the visualizations in the `ouput` folder.

## Comparison to iPlanner
