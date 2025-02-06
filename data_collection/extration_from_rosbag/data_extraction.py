"""
data_extraction.py

This script automates the process of collecting and storing data from a rosbag
for navigation-related tasks. It subscribes to topics that provide:

• **RGB Images** (compressed): Saved as PNG files for visual reference or model training.
• **Risk Images**: Converted to NumPy arrays (.npy) to preserve floating-point detail.
• **Grid Map**: Specific layers (elevation, traversability, risk) are extracted, stored with high precision, and accompanied by metadata (e.g., center positions).
• **Odometry**: Position and orientation data for each frame, exported to a text file.
• **Camera Intrinsics**: Captured once per session from the `/hdr_camera_front/camera_info` topic.
• **Transforms** (via `extract_and_save_transforms`): Records transformations between
  `odom` and `map_o3d_localization_manager` to align sensor data with the grid map.

**Core Workflow**:
1. **Initialize ROS Node**: Sets up subscribers for RGB, risk images, odometry, and grid map data.
2. **Message Caching**: Stores the latest received messages, ensuring synchronous extraction.
3. **Data Extraction**: Converts messages to usable formats (e.g., PNG, NumPy arrays, text) and saves them to a structured folder hierarchy (`./EnvironmentData/` by default).
4. **Handling Transforms**: Calls `extract_and_save_transforms` to save the transform between `odom` and `map_o3d_localization_manager`, ensuring consistency between pose, map, and image data.
5. **Spinning**: Keeps the node alive to continually process new incoming data until manually stopped.

This script is fundamental for collecting synchronized, labeled data from various
ROS topics, making it easy to train and evaluate navigation models.

Usage: Refer to the README for detailed instructions on running the script.

Author: [Paolo Petri]
Date: [07.02.2025]
"""
import rospy
import os
import cv2
import numpy as np
import tf
import tf2_ros
import threading
from sensor_msgs.msg import CompressedImage, Image, CameraInfo
from nav_msgs.msg import Odometry
from grid_map_msgs.msg import GridMap
from cv_bridge import CvBridge, CvBridgeError

# Initialize CvBridge
bridge = CvBridge()

# Initialize ROS node
rospy.init_node('data_extraction_node', anonymous=False)

# Initialize TF listener and buffer
tf_listener_dep = tf.TransformListener()
tf_buffer = tf2_ros.Buffer(rospy.Duration(10.0))  # tf buffer length
tf_listener = tf.TransformListener(tf_buffer)
rospy.sleep(1.0)  # Wait for the buffer to fill


# Global variables to store the latest messages
latest_rgb_msg = None
latest_risk_msg = None
latest_odom_msg = None
camera_info_saved = False


# Lock to ensure thread safety
data_lock = threading.Lock()

# Topics
rgb_topic = '/hdr_camera_front/image_raw/compressed'
risk_image_topic = '/hdr_camera_front/lmm_nav_sem'
odom_topic = '/state_estimator/odometry'
camera_info_topic = '/hdr_camera_front/camera_info'
grid_map_topic = '/elevation_mapping_large/elevation_map_raw'

# Folder structure
root_folder = './EnvironmentData'  # Update with your desired path
camera_folder = os.path.join(root_folder, 'camera')
risk_image_folder = os.path.join(root_folder, 'risk_images')
maps_data_folder = os.path.join(root_folder, 'maps')
odom_file = os.path.join(root_folder, 'base_to_world_transforms.txt')
camera_intrinsic_file = os.path.join(root_folder, 'camera_intrinsic.txt')
center_position_file = os.path.join(maps_data_folder, 'center_positions.txt')  # New file for center positions
odom_to_map_transform_file = os.path.join(maps_data_folder, 'world_to_grid_transforms.txt')  # New file for transforms

# Create folders
for folder in [camera_folder, risk_image_folder, maps_data_folder]:
    os.makedirs(folder, exist_ok=True)

# Create or clear the necessary files at the start
# This ensures that old data does not persist between runs
for file_path in [odom_file, camera_intrinsic_file, center_position_file, odom_to_map_transform_file]:
    with open(file_path, 'w') as f:
        pass  # Just create/clear the file

# Subscribers to cache latest messages
def rgb_callback(msg):
    global latest_rgb_msg
    with data_lock:
        latest_rgb_msg = msg

def risk_callback(msg):
    global latest_risk_msg
    with data_lock:
        latest_risk_msg = msg

def odom_callback(msg):
    global latest_odom_msg
    with data_lock:
        latest_odom_msg = msg

# Subscribe to relevant topics
rospy.Subscriber(rgb_topic, CompressedImage, rgb_callback)
rospy.Subscriber(risk_image_topic, Image, risk_callback)
rospy.Subscriber(odom_topic, Odometry, odom_callback)

# Initialize a counter for filenames
idx = 0  # Initialize a counter for filenames

# Data extraction function
def extract_and_save_data(rgb_msg, risk_msg, map_msg, odom_msg):
    global idx
    extract_and_save_rgb_image(rgb_msg, idx)
    extract_and_save_risk_image(risk_msg, idx)
    frame_is_odom = extract_and_save_map_layers(map_msg, idx)
    extract_and_save_odometry(odom_msg, idx)
    extract_and_save_transforms(idx, frame_is_odom)  # Call the modified extrinsic saving function
    idx += 1

# Grid map callback
def grid_map_callback(map_msg):
    with data_lock:
        missing_msgs = []

        if latest_rgb_msg is None:
            missing_msgs.append('latest_rgb_msg')
        if latest_risk_msg is None:
            missing_msgs.append('latest_risk_msg')
        if latest_odom_msg is None:
            missing_msgs.append('latest_odom_msg')

        if missing_msgs:
            rospy.logwarn("Waiting for messages: %s", ", ".join(missing_msgs))
            return  # Skip if any message is missing

        # Extract and save data once all messages are available
        extract_and_save_data(latest_rgb_msg, latest_risk_msg, map_msg, latest_odom_msg)


rospy.Subscriber(grid_map_topic, GridMap, grid_map_callback)

# Data extraction functions
def extract_and_save_rgb_image(rgb_msg, idx):
    """
    Extracts and saves the RGB image.
    """
    try:
        np_arr = np.frombuffer(rgb_msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        cv2.imwrite(os.path.join(camera_folder, f'{idx}.png'), cv_image)
    except Exception as e:
        rospy.logerr(f"Failed to save RGB image: {e}")

def extract_and_save_risk_image(risk_msg, idx):
    """
    Extracts and saves the risk image as a NumPy array (.npy file) to preserve floating-point values.
    
    Parameters:
    - risk_msg: The ROS Image message containing the risk image.
    - idx: An index used to name the output files uniquely.
    """
    try:
        # Convert the ROS Image message to a NumPy array using CvBridge
        cv_image = bridge.imgmsg_to_cv2(risk_msg, desired_encoding='passthrough')
        
        # Define the file path for saving the NumPy array
        image_path = os.path.join(risk_image_folder, f'{idx}.npy')
        
        # Save the NumPy array directly to preserve floating-point values
        np.save(image_path, cv_image)
        
        rospy.loginfo(f"Risk image saved at {image_path}")
        
    except CvBridgeError as e:
        rospy.logerr(f"Failed to convert risk image: {e}")
    except Exception as e:
        rospy.logerr(f"Failed to save risk image: {e}")



def extract_and_save_map_layers(map_msg, idx) -> bool:
    """
    Extracts and saves specified layers from a GridMap message into text files and stores 
    the grid map's center position. The function processes the 'elevation', 'traversability', 
    and 'risk' layers by reshaping their data into a 2D grid, formatting values in high-precision 
    exponential notation, and replacing NaN values with 0 for the 'traversability' layer.

    Additionally, it logs the center position and the processing status of each layer.

    Parameters:
        map_msg: The GridMap message containing the map data.
        idx (int): An index used to uniquely name the output files.

    Side Effects:
        - Appends the center position (x, y) to a 'center_positions.txt' file.
        - Creates and writes text files for each desired map layer in a specified folder structure.
        - Logs informational and error messages via rospy.

    Returns:
        bool: True if the GridMap's frame_id is 'odom', otherwise False.
    """

    layers = map_msg.layers
    desired_layers = ['elevation', 'traversability', 'risk']

    # Extract map metadata
    info = map_msg.info
    resolution = info.resolution
    length_x = info.length_x
    length_y = info.length_y
    position_x = info.pose.position.x
    position_y = info.pose.position.y

    frame_id = info.header.frame_id
    rospy.loginfo(f"Frame ID of Grid Maps: {frame_id}")
    if frame_id == 'odom':
        frame_is_odom = True
    else:
        frame_is_odom = False
    

    # Save center positions to 'center_positions.txt'
    try:
        with open(center_position_file, 'a') as f:
            f.write(f'{position_x},{position_y}\n')
        rospy.loginfo(f"Center position saved at {center_position_file} for idx {idx}")
    except Exception as e:
        rospy.logerr(f"Failed to save center positions: {e}")

    # Map dimensions in number of cells
    size_x = int(length_x / resolution)
    size_y = int(length_y / resolution)

    for layer in desired_layers:
        if layer in layers:
            # Get the index of the layer
            layer_index = layers.index(layer)
            
            # Each layer's data is in map_msg.data[layer_index]
            data = np.array(map_msg.data[layer_index].data, dtype=np.float64)

            # Reshape data to its 2D grid shape
            data_grid = data.reshape((size_x, size_y))

            # If the layer is 'traversability', replace NaN values with 0
            if layer == 'traversability':
                data_grid = np.nan_to_num(data_grid, nan=0.0)
                rospy.loginfo(f"NaN values in '{layer}' layer replaced with 0 for idx {idx}")

            # Prepare the format for high precision exponential notation
            # '%.18e' ensures 18 decimal places and standard exponential format
            format_str = '%.18e'

            # Define the folder for the current layer
            layer_folder = os.path.join(maps_data_folder, layer)
            os.makedirs(layer_folder, exist_ok=True)

            # Define the path for the current layer's data file
            data_file = os.path.join(layer_folder, f'{idx}.txt')

            try:
                with open(data_file, 'w') as f:
                    # Write the data row by row
                    for row in data_grid:
                        # Format each value in the row with high precision
                        formatted_row = [format_str % val for val in row]
                        # Join the formatted values with commas
                        row_str = ','.join(formatted_row)
                        # Write the formatted row to the file
                        f.write(row_str + '\n')
                rospy.loginfo(f"Map layer '{layer}' saved at {data_file} for idx {idx}")
            except Exception as e:
                rospy.logerr(f"Failed to save map layer '{layer}': {e}")
        else:
            rospy.logwarn(f"Layer '{layer}' not found in map message.")

    return frame_is_odom


def extract_and_save_odometry(odom_msg, idx):
    """
    Extracts and saves the odometry data without any transformation.
    """
    try:
        pos = odom_msg.pose.pose.position
        ori = odom_msg.pose.pose.orientation
        odom_data = [pos.x, pos.y, pos.z, ori.x, ori.y, ori.z, ori.w]
        odom_data_str = ','.join(map(str, odom_data))
        with open(odom_file, 'a') as f:
            f.write(f'{odom_data_str}\n')  # Include idx for synchronization
    except Exception as e:
        rospy.logerr(f"Failed to save odometry data: {e}")

# Save camera intrinsics and extrinsics
def save_camera_intrinsics():
    """
    Saves the camera's P matrix to a file.
    """
    global camera_info_saved
    if camera_info_saved:
        return
    try:
        camera_info_msg = rospy.wait_for_message(camera_info_topic, CameraInfo, timeout=5.0)
        P = np.array(camera_info_msg.P).reshape(3, 4)
        np.savetxt(camera_intrinsic_file, P, delimiter=',')
        camera_info_saved = True
        rospy.loginfo("Camera intrinsics saved.")
    except rospy.ROSException:
        rospy.logerr("Failed to receive camera intrinsic parameters.")
    except Exception as e:
        rospy.logerr(f"Failed to save camera intrinsics: {e}")

def extract_and_save_transforms(idx: int, frame_is_odom: bool) -> None:
    """
    Extracts and saves the latest transform from 'odom' to 'map_o3d_localization_manager'.
    
    If the frame is not 'odom', the function attempts to look up the transform between 'odom' 
    and 'map_o3d_localization_manager' and saves the translation and rotation (as a quaternion) 
    to a file. If the frame is 'odom', it saves an identity transform instead.

    Parameters:
        idx (int): An index used to uniquely name or associate the saved transform with a data sample.
        frame_is_odom (bool): A flag indicating whether the grid map's frame ID is 'odom'.
                              - If False, the function will try to extract the transform.
                              - If True, the function will save an identity transform.
    """
    if not frame_is_odom:
        try:
            trans, rot = tf_listener_dep.lookupTransform('odom', 'map_o3d_localization_manager', rospy.Time(0))  # Fill the buffer
            print(trans)

            # Save the transform with the current idx
            with open(odom_to_map_transform_file, 'a') as f:
                f.write(f'{trans[0]},{trans[1]},{trans[2]},{rot[0]},{rot[1]},{rot[2]},{rot[3]}\n')

            rospy.loginfo(f"Transform for idx {idx} saved successfully at {odom_to_map_transform_file}")

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(f"Transform from 'odom' to 'map_o3d_localization_manager' not found for idx {idx}: {e}")
    elif frame_is_odom:
            with open (odom_to_map_transform_file, 'a') as f:
                f.write('0,0,0,0,0,0,1\n')

# Initialize camera intrinsics
camera_info_saved = False  # Initialize the flag
save_camera_intrinsics()

# Spin to keep the script running
rospy.spin()
