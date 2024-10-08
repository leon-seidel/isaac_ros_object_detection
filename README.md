# YOLO11 instance segmentation for Isaac ROS Object Detection

Adding YOLO11 instance segmentation to Isaac ROS Object Detection.

<div align="center"><img alt="segmentation image" src="/isaac_ros_object_detection/isaac_ros_yolo11_seg/example/segmentation_example.png" width="400px"/>

## Overview
While object detection with both YOLO11 and YOLOv8 is possible with the ISAAC ROS YOLOv8 node, instance segmentation is not supported. This repo adds a node and lauch files to run YOLO11 and YOLOv8 instance segmentation models with ISAAC ROS. The segmentation masks are visualized with an additional ROS2 `image_msgs` message. Additionally the lowest point in every mask is computed as well and published within the `Detection2D` message.

## Installation
You can follow the [installation guide](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_object_detection/isaac_ros_yolov8/index.html#quickstart) for YOLOv8 object detection until `Build isaac_ros_yolov8`.

Then follow these steps:

1. Clone this repository under `${ISAAC_ROS_WS}/src`:
```
cd ${ISAAC_ROS_WS}/src && \
   git clone https://github.com/leon-seidel/isaac_ros_object_detection.git isaac_ros_object_detection
```
2. Launch the Docker container using the `run_dev.sh` script:
```
cd ${ISAAC_ROS_WS}/src/isaac_ros_common && \
./scripts/run_dev.sh
```
3. Use `rosdep` to install the package’s dependencies:
```
rosdep install --from-paths ${ISAAC_ROS_WS}/src/isaac_ros_object_detection --ignore-src -y
```
4. Build this package from source:
```
cd ${ISAAC_ROS_WS} && \
   colcon build --packages-up-to isaac_ros_yolo11_seg
```
5. Source the ROS workspace:
```
source install/setup.bash
```


## Usage

Launch the YOLO11 instance segmentation pipeline with visualisation using:
```
ros2 launch ./src/isaac_ros_object_detection/isaac_ros_yolo11_seg/launch/isaac_ros_yolo11_seg_visualize.launch.py model_file_path:=./isaac_ros_assets/models/yolo11/yolo11n-seg.onnx engine_file_path:=./isaac_ros_assets/models/yolo11/yolov11n-seg.plan input_binding_names:=['images'] output_binding_names:=['output0','output1'] output_tensor_names:=['output_tensor','output_tensor1']  network_image_width:=640 network_image_height:=640 force_engine_update:=False image_mean:=[0.0,0.0,0.0] image_stddev:=[1.0,1.0,1.0] input_image_width:=640 input_image_height:=640 confidence_threshold:=0.25 nms_threshold:=0.45 num_classes:=80
```
For an example rosbag install `sudo apt-get install -y ros-humble-isaac-ros-examples` in a second terminal in the Docker Container and run:
```
ros2 bag play -l ${ISAAC_ROS_WS}/isaac_ros_assets/isaac_ros_yolov8/quickstart.bag
```

