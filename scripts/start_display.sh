#!/bin/bash
# 启动显示节点

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=0
source /opt/ros/humble/setup.bash
source /home/nvidia/VLM_project/bot_project/install/bot_interfaces/share/bot_interfaces/local_setup.bash
source /home/nvidia/VLM_project/bot_project/install/bot_display/share/bot_display/local_setup.bash

echo "启动显示节点..."
ros2 run bot_display display_node
