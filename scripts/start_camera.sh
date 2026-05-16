#!/bin/bash
# 启动 RealSense D435i 相机

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=0
source /opt/ros/humble/setup.bash
source /home/nvidia/VLM_project/realsense-ros/install/realsense2_camera_msgs/share/realsense2_camera_msgs/local_setup.bash
source /home/nvidia/VLM_project/realsense-ros/install/realsense2_camera/share/realsense2_camera/local_setup.bash

echo "启动 RealSense D435i 相机..."
ros2 launch realsense2_camera rs_launch.py
