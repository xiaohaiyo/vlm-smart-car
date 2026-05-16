#!/bin/bash
# ============================================================================
# VLM 智能小车项目环境配置脚本
# ============================================================================
# 使用方法: source setup_env.sh
# ============================================================================

echo "=========================================="
echo "    VLM 智能小车环境配置"
echo "=========================================="

# ============================================================================
# 1. ROS 2 环境
# ============================================================================
echo "[1/6] 配置 ROS 2 环境..."
source /opt/ros/humble/setup.bash

# ============================================================================
# 2. RealSense 相机包
# ============================================================================
echo "[2/6] 配置 RealSense 相机..."
source /home/nvidia/VLM_project/realsense-ros/install/realsense2_camera_msgs/share/realsense2_camera_msgs/local_setup.bash
source /home/nvidia/VLM_project/realsense-ros/install/realsense2_camera/share/realsense2_camera/local_setup.bash

# ============================================================================
# 3. 项目包
# ============================================================================
echo "[3/6] 配置项目包..."
source /home/nvidia/VLM_project/bot_project/install/bot_interfaces/share/bot_interfaces/local_setup.bash
source /home/nvidia/VLM_project/bot_project/install/bot_vlm/share/bot_vlm/local_setup.bash
source /home/nvidia/VLM_project/bot_project/install/bot_chassis/share/bot_chassis/local_setup.bash
source /home/nvidia/VLM_project/bot_project/install/bot_speech/share/bot_speech/local_setup.bash

# ============================================================================
# 4. TensorRT-Edge-LLM 环境变量
# ============================================================================
echo "[4/6] 配置 TensorRT-Edge-LLM..."
export EDGELLM_PLUGIN_PATH=/home/nvidia/VLM_project/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so
export LD_LIBRARY_PATH=/home/nvidia/VLM_project/TensorRT-Edge-LLM/build:$LD_LIBRARY_PATH

# ============================================================================
# 5. ROS 2 DDS 配置
# ============================================================================
echo "[5/6] 配置 DDS..."
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=0

# ============================================================================
# 6. 模型路径
# ============================================================================
echo "[6/6] 配置模型路径..."
export VLM_ENGINE_DIR=/home/nvidia/VLM_project/Cosmos_ONNXINT4/engine
export ASR_ENGINE_DIR=/home/nvidia/VLM_project/Qwen3-ASR-0.6B_onnx/engine

echo ""
echo "=========================================="
echo "    环境配置完成！"
echo "=========================================="
echo ""
echo "可用命令:"
echo "  ros2 launch launch/vlm_pipeline.launch.py  # 一键启动"
echo "  ros2 run bot_speech speech_node            # 语音识别"
echo "  ros2 run bot_vlm vlm_node                  # VLM 推理"
echo "  ros2 run bot_chassis chassis_node           # 底盘控制"
echo "  python3 gui/vlm_gui.py                      # GUI 控制面板"
echo ""
