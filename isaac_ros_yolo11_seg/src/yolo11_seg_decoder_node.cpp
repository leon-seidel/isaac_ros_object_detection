// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

#include "isaac_ros_yolo11_seg/yolo11_seg_decoder_node.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list_view.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list.hpp"


#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/dnn.hpp>
#include <opencv4/opencv2/dnn/dnn.hpp>

#include "vision_msgs/msg/detection2_d_array.hpp"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"


namespace nvidia
{
namespace isaac_ros
{
namespace yolo11_seg
{
Yolo11SegDecoderNode::Yolo11SegDecoderNode(const rclcpp::NodeOptions options)
: rclcpp::Node("yolo11_seg_decoder_node", options),
  nitros_sub_{std::make_shared<nvidia::isaac_ros::nitros::ManagedNitrosSubscriber<
        nvidia::isaac_ros::nitros::NitrosTensorListView>>(
      this,
      "tensor_sub",
      nvidia::isaac_ros::nitros::nitros_tensor_list_nchw_rgb_f32_t::supported_type_name,
      std::bind(&Yolo11SegDecoderNode::InputCallback, this,
      std::placeholders::_1))},
  pub_{create_publisher<vision_msgs::msg::Detection2DArray>(
      "detections_output", 50)},
  pub_mask_{create_publisher<sensor_msgs::msg::Image>(
      "detections_mask", 50)},
  tensor_name_{declare_parameter<std::string>("tensor_name", "output_tensor")},
  tensor1_name_{declare_parameter<std::string>("tensor1_name", "output_tensor1")},
  confidence_threshold_{declare_parameter<double>("confidence_threshold", 0.25)},
  nms_threshold_{declare_parameter<double>("nms_threshold", 0.45)},
  input_image_width_{declare_parameter<long int>("input_image_width", 640)},
  input_image_height_{declare_parameter<long int>("input_image_height", 640)},
  num_classes_{declare_parameter<long int>("num_classes", 80)}
{}

Yolo11SegDecoderNode::~Yolo11SegDecoderNode() = default;

void Yolo11SegDecoderNode::InputCallback(const nvidia::isaac_ros::nitros::NitrosTensorListView & msg)
{
  auto tensor = msg.GetNamedTensor(tensor_name_);
  size_t buffer_size{tensor.GetTensorSize()};
  std::vector<float> results_vector{};
  results_vector.resize(buffer_size);
  cudaMemcpy(results_vector.data(), tensor.GetBuffer(), buffer_size, cudaMemcpyDefault);

  auto tensor1 = msg.GetNamedTensor(tensor1_name_);
  size_t buffer_size1{tensor1.GetTensorSize()};
  std::vector<float> results_vector1{};
  results_vector1.resize(buffer_size1);
  cudaMemcpy(results_vector1.data(), tensor1.GetBuffer(), buffer_size1, cudaMemcpyDefault);

  std::vector<cv::Rect> bboxes;
  std::vector<float> scores;
  std::vector<int> indices;
  std::vector<int> classes;

  //  Output dimensions = [1, num_classes+4+num_protomasks, out_dim] = [1, 116, 8400(@640x640)]
  float image_size_to_mask_size = 0.25;   // Mask size relative to image size, e.g mask size 160x160 for images with 640x640
  int num_proto_masks = 32;               // Number of prototype masks
  
  int out_dim = (input_image_width_/8) * (input_image_height_/8) + (input_image_width_/16) * (input_image_height_/16) + (input_image_width_/32) * (input_image_height_/32);
  int mask_points_x = int(image_size_to_mask_size * float(input_image_width_));
  int mask_points_y = int(image_size_to_mask_size * float(input_image_height_));
  int num_mask_points = mask_points_x * mask_points_y;
  float * results_data = reinterpret_cast<float *>(results_vector.data());
  float * results_data1 = reinterpret_cast<float *>(results_vector1.data());

  for (int i = 0; i < out_dim; i++) {
    float x = *(results_data + i);
    float y = *(results_data + (out_dim * 1) + i);
    float w = *(results_data + (out_dim * 2) + i);
    float h = *(results_data + (out_dim * 3) + i);

    float x1 = (x - (0.5 * w));
    float y1 = (y - (0.5 * h));
    float width = w;
    float height = h;

    std::vector<float> conf;
    for (int j = 0; j < num_classes_; j++) {
      conf.push_back(*(results_data + (out_dim * (4 + j)) + i));
    }

    std::vector<float>::iterator ind_max_conf;
    ind_max_conf = std::max_element(std::begin(conf), std::end(conf));
    int max_index = distance(std::begin(conf), ind_max_conf);
    float val_max_conf = *max_element(std::begin(conf), std::end(conf));

    bboxes.push_back(cv::Rect(x1, y1, width, height));
    indices.push_back(i);
    scores.push_back(val_max_conf);
    classes.push_back(max_index);
  }

  RCLCPP_DEBUG(this->get_logger(), "Count of bboxes: %lu", bboxes.size());
  cv::dnn::NMSBoxes(bboxes, scores, confidence_threshold_, nms_threshold_, indices, 5);
  int num_indices_after_nms = indices.size();
  RCLCPP_DEBUG(this->get_logger(), "# boxes after NMS: %lu", indices.size());

  vision_msgs::msg::Detection2DArray final_detections_arr;
  std::vector<u_int8_t> final_mask(num_mask_points);

  for (size_t i = 0; i < indices.size(); i++) {
    int ind = indices[i];
    vision_msgs::msg::Detection2D detection;

    geometry_msgs::msg::Pose center;
    geometry_msgs::msg::Point position;
    geometry_msgs::msg::Quaternion orientation;

    // 2D object Bbox
    vision_msgs::msg::BoundingBox2D bbox;
    float w = bboxes[ind].width;
    float h = bboxes[ind].height;
    float x_bbox = bboxes[ind].x;
    float y_bbox = bboxes[ind].y;
    float x_center = x_bbox + (0.5 * w);
    float y_center = y_bbox + (0.5 * h);
    detection.bbox.center.position.x = x_center;
    detection.bbox.center.position.y = y_center;
    detection.bbox.size_x = w;
    detection.bbox.size_y = h;

    // Segmentation mask
    int x_min = int(x_bbox * image_size_to_mask_size);
    int x_max = int((x_bbox + w) * image_size_to_mask_size);
    int y_min = int(y_bbox * image_size_to_mask_size);
    int y_max = int((y_bbox + h) * image_size_to_mask_size);
    
    // Calculate mask value
    int mask_value;
    if (i < 200) {
      if (num_indices_after_nms < 1) {
        mask_value = 0;
      } else if (num_indices_after_nms >= 200) {
        mask_value = int((float(i) / 200.0) * 200.0) + 54;
      } else {
        mask_value = int((float(i) / float(num_indices_after_nms)) * 200.0) + 54;
      }
    } else {
      mask_value = 255;
    }
        
    // Initiate values for lowest mask point
    float lowest_mask_point_x = -1.0f;
    std::vector<float> lowest_mask_point_x_vec;
    int lowest_mask_point_y = -1;
    // Iterate over bounding box
    for (int x = x_min; x < x_max; x++) {   
      for (int y = y_min; y < y_max; y++) {
        int vector_pos = (y * mask_points_x) + x;
        // Matrix multiplication
        float matrix_multi = 0.0f;
        for (int k = 0; k < num_proto_masks; k++) {
          matrix_multi += results_data[(num_classes_ + 4 + k) * out_dim + ind] * results_data1[k * num_mask_points + vector_pos];
        }
        // Sigmoid function
        float sigmoid = 1.0f / (1.0f + std::exp(-matrix_multi));
        // Use mask if Sigmoid larger than 0.5
        if (sigmoid > 0.5f) {
          final_mask[vector_pos] = mask_value;
          // Get lowest mask point
          if (y > lowest_mask_point_y) {
            lowest_mask_point_x_vec.clear();
            lowest_mask_point_x_vec.push_back(float(x));
            lowest_mask_point_y = y;
          } 
          else if (y == lowest_mask_point_y) {
            lowest_mask_point_x_vec.push_back(float(x));
          }
        }
      }
    }

    // Get lowest mask point
    if(!lowest_mask_point_x_vec.empty()){
      lowest_mask_point_x = std::reduce(lowest_mask_point_x_vec.begin(), lowest_mask_point_x_vec.end()) / static_cast<float>(lowest_mask_point_x_vec.size());
    }
    float lowest_max_point_rel_x = -1.0f;
    float lowest_max_point_rel_y = -1.0f;
    if (lowest_mask_point_x >= 0 && lowest_mask_point_y >= 0) {
      lowest_max_point_rel_x = lowest_mask_point_x / float(mask_points_x);
      lowest_max_point_rel_y = float(lowest_mask_point_y) / float(mask_points_y);
    }

    // Class probabilities
    vision_msgs::msg::ObjectHypothesisWithPose hyp;
    hyp.hypothesis.class_id = std::to_string(classes.at(ind));
    hyp.hypothesis.score = scores.at(ind);
    hyp.pose.pose.position.x = lowest_max_point_rel_x;
    hyp.pose.pose.position.y = lowest_max_point_rel_y;
    detection.results.push_back(hyp);

    detection.header.stamp.sec = msg.GetTimestampSeconds();
    detection.header.stamp.nanosec = msg.GetTimestampNanoseconds();

    final_detections_arr.detections.push_back(detection);
  }

  final_detections_arr.header.stamp.sec = msg.GetTimestampSeconds();
  final_detections_arr.header.stamp.nanosec = msg.GetTimestampNanoseconds();
  pub_->publish(final_detections_arr);

  // Mask image
  sensor_msgs::msg::Image mask_img;
  mask_img.header.stamp.sec = msg.GetTimestampSeconds();
  mask_img.header.stamp.nanosec = msg.GetTimestampNanoseconds();
  mask_img.width = mask_points_x;
  mask_img.height = mask_points_y;
  mask_img.encoding = "mono8"; // Binary encoding
  mask_img.step = mask_points_x;
  mask_img.data = final_mask;
  pub_mask_->publish(mask_img);
}

}  // namespace yolo11_seg
}  // namespace isaac_ros
}  // namespace nvidia

// Register as component
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::yolo11_seg::Yolo11SegDecoderNode)
