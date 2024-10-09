# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import os

from ament_index_python.packages import get_package_share_directory
import launch
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    """Generate launch description for TensorRT ROS 2 node."""
    # By default loads and runs mobilenetv2-1.0 included in isaac_ros_dnn_inference/models
    launch_args = [
        DeclareLaunchArgument(
            'model_file_path',
            default_value='',
            description='The absolute file path to the ONNX file'),
        DeclareLaunchArgument(
            'engine_file_path',
            default_value='',
            description='The absolute file path to the TensorRT engine file'),
        DeclareLaunchArgument(
            'input_tensor_names',
            default_value='["input_tensor"]',
            description='A list of tensor names to bound to the specified input binding names'),
        DeclareLaunchArgument(
            'input_binding_names',
            default_value='["images"]',
            description='A list of input tensor binding names (specified by model)'),
        DeclareLaunchArgument(
            'output_tensor_names',
            default_value='["output_tensor", "output_tensor1"]',
            description='A list of tensor names to bound to the specified output binding names'),
        DeclareLaunchArgument(
            'output_binding_names',
            default_value='["output0", "output1"]',
            description='A list of output tensor binding names (specified by model)'),
        DeclareLaunchArgument(
            'verbose',
            default_value='False',
            description='Whether TensorRT should verbosely log or not'),
        DeclareLaunchArgument(
            'force_engine_update',
            default_value='False',
            description='Whether TensorRT should update the TensorRT engine file or not'),
        DeclareLaunchArgument(
            'image_mean',
            default_value='[0.0, 0.0, 0.0]',
            description='The mean for image normalization'),
        DeclareLaunchArgument(
            'image_stddev',
            default_value='[1.0, 1.0, 1.0]',
            description='The standard deviation for image normalization'),
        DeclareLaunchArgument(
            'confidence_threshold',
            default_value='0.25',
            description='Confidence threshold to filter candidate detections during NMS'),
        DeclareLaunchArgument(
            'nms_threshold',
            default_value='0.45',
            description='NMS IOU threshold'),
        DeclareLaunchArgument(
            'num_classes',
            default_value='80',
            description='Number of classes in the model'),
        DeclareLaunchArgument(
            'network_image_width',
            default_value='640',
            description='The input image width that the network expects'),
        DeclareLaunchArgument(
            'network_image_height',
            default_value='640',
            description='The input image height that the network expects'),
        DeclareLaunchArgument(
            'input_image_width',
            default_value='640',
            description='The input image width from the camera'),
        DeclareLaunchArgument(
            'input_image_height',
            default_value='640',
            description='The input image height from the camera'),
    ]

    # DNN Image Encoder parameters
    input_image_width = LaunchConfiguration('input_image_width')
    input_image_height = LaunchConfiguration('input_image_height')
    network_image_width = LaunchConfiguration('network_image_width')
    network_image_height = LaunchConfiguration('network_image_height')
    image_mean = LaunchConfiguration('image_mean')
    image_stddev = LaunchConfiguration('image_stddev')

    # TensorRT parameters
    model_file_path = LaunchConfiguration('model_file_path')
    engine_file_path = LaunchConfiguration('engine_file_path')
    input_tensor_names = LaunchConfiguration('input_tensor_names')
    input_binding_names = LaunchConfiguration('input_binding_names')
    output_tensor_names = LaunchConfiguration('output_tensor_names')
    output_binding_names = LaunchConfiguration('output_binding_names')
    verbose = LaunchConfiguration('verbose')
    force_engine_update = LaunchConfiguration('force_engine_update')

    # YOLO11_seg Decoder parameters
    confidence_threshold = LaunchConfiguration('confidence_threshold')
    nms_threshold = LaunchConfiguration('nms_threshold')
    num_classes = LaunchConfiguration('num_classes')

    encoder_dir = get_package_share_directory('isaac_ros_dnn_image_encoder')
    yolo11_seg_encoder_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [os.path.join(encoder_dir, 'launch', 'dnn_image_encoder.launch.py')]
        ),
        launch_arguments={
            'input_image_width': input_image_width,
            'input_image_height': input_image_height,
            'network_image_width': network_image_width,
            'network_image_height': network_image_height,
            'image_mean': image_mean,
            'image_stddev': image_stddev,
            'attach_to_shared_component_container': 'True',
            'component_container_name': 'tensor_rt_container',
            'dnn_image_encoder_namespace': 'yolo11_seg_encoder',
            'image_input_topic': '/image_rect',
            'camera_info_input_topic': '/camera_info_rect',
            'tensor_output_topic': '/tensor_pub',
        }.items(),
    )

    tensor_rt_node = ComposableNode(
        name='tensor_rt',
        package='isaac_ros_tensor_rt',
        plugin='nvidia::isaac_ros::dnn_inference::TensorRTNode',
        parameters=[{
            'model_file_path': model_file_path,
            'engine_file_path': engine_file_path,
            'output_binding_names': output_binding_names,
            'output_tensor_names': output_tensor_names,
            'input_tensor_names': input_tensor_names,
            'input_binding_names': input_binding_names,
            'verbose': verbose,
            'force_engine_update': force_engine_update
        }]
    )

    yolo11_seg_decoder_node = ComposableNode(
        name='yolo11_seg_decoder_node',
        package='isaac_ros_yolo11_seg',
        plugin='nvidia::isaac_ros::yolo11_seg::Yolo11SegDecoderNode',
        parameters=[{
            'confidence_threshold': confidence_threshold,
            'nms_threshold': nms_threshold,
            'input_image_height': input_image_height,
            'input_image_width': input_image_width,
            'num_classes': num_classes,
        }]
    )

    tensor_rt_container = ComposableNodeContainer(
        name='tensor_rt_container',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[tensor_rt_node, yolo11_seg_decoder_node],
        output='screen',
        arguments=['--ros-args', '--log-level', 'INFO'],
        namespace=''
    )

    final_launch_description = launch_args + [tensor_rt_container, yolo11_seg_encoder_launch]
    return launch.LaunchDescription(final_launch_description)
