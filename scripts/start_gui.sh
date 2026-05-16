#!/bin/bash
# 启动 Qt GUI 显示界面

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=0

source /opt/ros/humble/setup.bash
source /home/nvidia/VLM_project/bot_project/install/bot_interfaces/share/bot_interfaces/local_setup.bash

python3 /home/nvidia/VLM_project/bot_project/gui/vlm_gui.py
