#!/bin/bash
# 启动 VLM 推理节点

export EDGELLM_PLUGIN_PATH=/home/nvidia/VLM_project/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so
export LD_LIBRARY_PATH=/home/nvidia/VLM_project/TensorRT-Edge-LLM/build:$LD_LIBRARY_PATH
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=0
source /opt/ros/humble/setup.bash
source /home/nvidia/VLM_project/bot_project/install/bot_interfaces/share/bot_interfaces/local_setup.bash
source /home/nvidia/VLM_project/bot_project/install/bot_vlm/share/bot_vlm/local_setup.bash

echo "启动 VLM 推理节点..."
ros2 run bot_vlm vlm_node
