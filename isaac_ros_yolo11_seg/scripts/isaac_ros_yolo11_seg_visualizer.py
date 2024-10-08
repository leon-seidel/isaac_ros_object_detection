#!/usr/bin/env python3

# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# This script listens for images and object detections on the image,
# then renders the output boxes on top of the image and publishes
# the result as an image message

import cv2
import numpy as np
import cv_bridge
import message_filters
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray

names = {
        0: 'person',
        1: 'bicycle',
        2: 'car',
        3: 'motorcycle',
        4: 'airplane',
        5: 'bus',
        6: 'train',
        7: 'truck',
        8: 'boat',
        9: 'traffic light',
        10: 'fire hydrant',
        11: 'stop sign',
        12: 'parking meter',
        13: 'bench',
        14: 'bird',
        15: 'cat',
        16: 'dog',
        17: 'horse',
        18: 'sheep',
        19: 'cow',
        20: 'elephant',
        21: 'bear',
        22: 'zebra',
        23: 'giraffe',
        24: 'backpack',
        25: 'umbrella',
        26: 'handbag',
        27: 'tie',
        28: 'suitcase',
        29: 'frisbee',
        30: 'skis',
        31: 'snowboard',
        32: 'sports ball',
        33: 'kite',
        34: 'baseball bat',
        35: 'baseball glove',
        36: 'skateboard',
        37: 'surfboard',
        38: 'tennis racket',
        39: 'bottle',
        40: 'wine glass',
        41: 'cup',
        42: 'fork',
        43: 'knife',
        44: 'spoon',
        45: 'bowl',
        46: 'banana',
        47: 'apple',
        48: 'sandwich',
        49: 'orange',
        50: 'broccoli',
        51: 'carrot',
        52: 'hot dog',
        53: 'pizza',
        54: 'donut',
        55: 'cake',
        56: 'chair',
        57: 'couch',
        58: 'potted plant',
        59: 'bed',
        60: 'dining table',
        61: 'toilet',
        62: 'tv',
        63: 'laptop',
        64: 'mouse',
        65: 'remote',
        66: 'keyboard',
        67: 'cell phone',
        68: 'microwave',
        69: 'oven',
        70: 'toaster',
        71: 'sink',
        72: 'refrigerator',
        73: 'book',
        74: 'clock',
        75: 'vase',
        76: 'scissors',
        77: 'teddy bear',
        78: 'hair drier',
        79: 'toothbrush',
}


class Yolo11SegVisualizer(Node):
    QUEUE_SIZE = 10
    color = (0, 255, 0)
    bbox_thickness = 2

    def __init__(self):
        super().__init__('yolo11_seg_visualizer')
        self._bridge = cv_bridge.CvBridge()
        self._processed_image_pub = self.create_publisher(
            Image, 'yolo11_seg_processed_image',  self.QUEUE_SIZE)

        self._detections_subscription = message_filters.Subscriber(
            self,
            Detection2DArray,
            'detections_output')
        self._detections_mask_subscription = message_filters.Subscriber(
            self,
            Image,
            'detections_mask')
        self._image_subscription = message_filters.Subscriber(
            self,
            Image,
            'image')

        self.time_synchronizer = message_filters.TimeSynchronizer(
            [self._detections_subscription, self._image_subscription, self._detections_mask_subscription],
            self.QUEUE_SIZE)

        self.time_synchronizer.registerCallback(self.detections_callback)

    def detections_callback(self, detections_msg, img_msg, mask_msg):
        def model_to_image_pos(model_point, image_shape, model_shape):
            model_px, model_py = model_point
            image_x, image_y = image_shape
            model_x, model_y = model_shape
                        
            rel_image_y = image_y / image_x
            rel_model_y = model_y / model_x
            rel_model_point_y = model_py / model_y
            rel_model_point_x = model_px / model_x

            image_py = round(image_y * ((rel_model_point_y - ((rel_model_y - rel_image_y) / 2)) / rel_image_y))
            image_px = round(image_x * rel_model_point_x)
 
            return (image_px, image_py)

        
        txt_color = (255, 0, 255)
        cv2_img = self._bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
        cv2_mask = self._bridge.imgmsg_to_cv2(mask_msg)
        image_shape = [cv2_img.shape[1], cv2_img.shape[0]]
        model_shape = [cv2_mask.shape[1], cv2_mask.shape[0]]

        upscaled_mask = cv2.resize(cv2_mask, (cv2_img.shape[1], cv2_img.shape[1]), interpolation=cv2.INTER_LINEAR)
        height_start = round((cv2_img.shape[1] - cv2_img.shape[0]) / 2)
        height_end = cv2_img.shape[1] - height_start
        upscaled_mask = upscaled_mask[height_start:height_end, 0:cv2_img.shape[1]]
        red_color = (0, 0, 255)
        opacity = 0.6
        red_mask = np.zeros_like(cv2_img)
        red_mask[upscaled_mask > 0] = red_color
        cv2_img = cv2.addWeighted(cv2_img, 1.0, red_mask, opacity, 0.0)

        for detection in detections_msg.detections:
            center_x = detection.bbox.center.position.x
            center_y = detection.bbox.center.position.y
            width = detection.bbox.size_x
            height = detection.bbox.size_y

            label = names[int(detection.results[0].hypothesis.class_id)]
            conf_score = detection.results[0].hypothesis.score
            label = f'{label} {conf_score:.2f}'

            min_pt = (round(center_x - (width / 2.0)),
                      round(center_y - (height / 2.0)))
            max_pt = (round(center_x + (width / 2.0)),
                      round(center_y + (height / 2.0)))
            
            min_pt = model_to_image_pos(min_pt, image_shape, (cv2_img.shape[1], cv2_img.shape[1]))
            max_pt = model_to_image_pos(max_pt, image_shape, (cv2_img.shape[1], cv2_img.shape[1]))
            
            lw = max(round((img_msg.height + img_msg.width) / 2 * 0.003), 2)  # line width
            tf = max(lw - 1, 1)  # font thickness
            # text width, height
            w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]
            outside = min_pt[1] - h >= 3

            cv2.rectangle(cv2_img, min_pt, max_pt,
                          self.color, self.bbox_thickness)
            cv2.putText(cv2_img, label, (min_pt[0], min_pt[1]-2 if outside else min_pt[1]+h+2),
                        0, lw / 3, txt_color, thickness=tf, lineType=cv2.LINE_AA)
            
            lowest_point_x = round(detection.results[0].pose.pose.position.x * cv2_mask.shape[1])
            lowest_point_y = round(detection.results[0].pose.pose.position.y * cv2_mask.shape[0])
            lowest_point = model_to_image_pos((lowest_point_x, lowest_point_y), image_shape, model_shape)
            cv2.circle(cv2_img, lowest_point, radius=10, color=(255, 0, 0), thickness=-1)

        processed_img = self._bridge.cv2_to_imgmsg(
            cv2_img, encoding=img_msg.encoding)
        self._processed_image_pub.publish(processed_img)


def main():
    rclpy.init()
    rclpy.spin(Yolo11SegVisualizer())
    rclpy.shutdown()


if __name__ == '__main__':
    main()