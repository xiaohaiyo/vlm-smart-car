#!/bin/bash
# VLM 智能小车一键启动脚本

echo "=========================================="
echo "    VLM 智能小车系统启动"
echo "=========================================="

# 设置环境变量
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=0
export EDGELLM_PLUGIN_PATH=/home/nvidia/VLM_project/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so
export LD_LIBRARY_PATH=/home/nvidia/VLM_project/TensorRT-Edge-LLM/build:$LD_LIBRARY_PATH

# Source ROS 和所有包
source /opt/ros/humble/setup.bash
source /home/nvidia/VLM_project/realsense-ros/install/realsense2_camera_msgs/share/realsense2_camera_msgs/local_setup.bash
source /home/nvidia/VLM_project/realsense-ros/install/realsense2_camera/share/realsense2_camera/local_setup.bash
source /home/nvidia/VLM_project/bot_project/install/bot_interfaces/share/bot_interfaces/local_setup.bash
source /home/nvidia/VLM_project/bot_project/install/bot_vlm/share/bot_vlm/local_setup.bash
source /home/nvidia/VLM_project/bot_project/install/bot_chassis/share/bot_chassis/local_setup.bash

echo "环境变量已设置"
echo "启动完整管线..."
echo ""

# 启动完整管线
ros2 launch /home/nvidia/VLM_project/bot_project/launch/vlm_pipeline.launch.py
