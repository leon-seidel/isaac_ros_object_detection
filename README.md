# YOLO11 instance segmentation for Isaac ROS

Adding YOLO11 (and YOLOv8) instance segmentation to Isaac ROS.

<div align="center"><img alt="segmentation image" src="./isaac_ros_yolo11_seg/example/segmentation_example.png" width="400px"/></div>

## Overview
While object detection with both YOLO11 and YOLOv8 is possible with the ISAAC ROS YOLOv8 node, instance segmentation is not supported. This repo adds a node and lauch files to run YOLO11 and YOLOv8 instance segmentation models with ISAAC ROS. The segmentation masks are visualized with an additional ROS2 `Image` message. Additionally the lowest point in every mask is computed as well and published within the `Detection2DArray` message.

## Installation
You can follow the [Developer Environment Setup](https://nvidia-isaac-ros.github.io/getting_started/dev_env_setup.html) to get started with ISAAC ROS.

Then follow these steps:

1. Prepare your YOLO11 instance segmentation model:
Install Ultralytics, create a folder and then export the YOLO11 model of your choice to ONNX there:
```
pip install ultralytics
mkdir -p ${ISAAC_ROS_WS}/isaac_ros_assets/models/yolo11
cd ${ISAAC_ROS_WS}/isaac_ros_assets/models/yolo11
yolo export model="yolo11n-seg.pt" format="onnx"
```
2. Clone `isaac_ros_common` under `${ISAAC_ROS_WS}/src`:
```
cd ${ISAAC_ROS_WS}/src && \
   git clone -b release-3.1 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common.git isaac_ros_common
```

3. Clone this repository under `${ISAAC_ROS_WS}/src`:
```
cd ${ISAAC_ROS_WS}/src && \
   git clone https://github.com/leon-seidel/isaac_ros_object_detection.git isaac_ros_object_detection
```
4. Launch the Docker container using the `run_dev.sh` script:
```
cd ${ISAAC_ROS_WS}/src/isaac_ros_common && \
./scripts/run_dev.sh
```
5. Use `rosdep` to install the package’s dependencies:
```
rosdep install --from-paths ${ISAAC_ROS_WS}/src/isaac_ros_object_detection --ignore-src -y
```
6. Build this package from source:
```
cd ${ISAAC_ROS_WS} && \
   colcon build --packages-up-to isaac_ros_yolo11_seg
```
7. Source the ROS workspace:
```
source install/setup.bash
```


## Usage

Launch the YOLO11 instance segmentation pipeline with visualisation using:
```
ros2 launch ./src/isaac_ros_object_detection/isaac_ros_yolo11_seg/launch/yolo11_seg_tensor_rt.launch.py model_file_path:=./isaac_ros_assets/models/yolo11/yolo11n-seg.onnx engine_file_path:=./isaac_ros_assets/models/yolo11/yolov11n-seg.plan input_image_width:=640 input_image_height:=640
```
For an example rosbag start a second terminal with the Docker container:
```
cd ${ISAAC_ROS_WS}/src/isaac_ros_common && \
./scripts/run_dev.sh
```
And then play the rosbag:
```
ros2 bag play -l ${ISAAC_ROS_WS}/isaac_ros_assets/isaac_ros_yolov8/quickstart.bag
```
To visualize the model output start a third terminal with the Docker container:
```
cd ${ISAAC_ROS_WS}/src/isaac_ros_common && \
./scripts/run_dev.sh
```
And then launch the visualizer node and the RQT Image View:
```
ros2 launch src/isaac_ros_object_detection/isaac_ros_yolo11_seg/launch/isaac_ros_yolo11_seg_visualize.launch.py
```
