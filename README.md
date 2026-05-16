# VLM 智能小车实验

基于视觉语言模型（VLM）的智能小车项目，运行在 NVIDIA Jetson AGX Orin 平台上。

![[f57d1485b2dfd1c8def0b1f1d8ab855a.mp4]]
## 大概项目架构

![[Pasted image 20260515210344.png]]
## 环境配置

### 硬件要求

| 组件   | 型号                          | 说明       |
| ---- | --------------------------- | -------- |
| 计算平台 | NVIDIA Jetson AGX Orin 64GB | 主控       |
| 相机   | Intel RealSense D435i       | RGB-D 相机 |
| 底盘   | Codbot D100                 | 差速底盘     |
| 串口   | CH340 USB 转串口               | 底盘通信     |

### 软件依赖

| 组件                | 版本     | 说明       |
| ----------------- | ------ | -------- |
| Ubuntu            | 22.04  | 操作系统     |
| ROS 2             | Humble | 中间件      |
| CUDA              | 12.6   | GPU 计算   |
| TensorRT          | 10.3   | 推理加速     |
| RealSense SDK     | 2.57.7 | 相机驱动     |
| TensorRT-Edge-LLM | 0.7.0  | LLM 推理引擎 |

### TensorRT-Edge-LLM环境搭建教程：
https://nvidia.github.io/TensorRT-Edge-LLM/latest/overview.html

### 模型下载：
语音转文字ASR：
https://huggingface.co/Qwen/Qwen3-ASR-0.6B
VLM大模型：
https://huggingface.co/nvidia/Cosmos-Reason2-2B

导出并量化参考：
https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/features/quantization.html
c++API参考：
https://nvidia.github.io/TensorRT-Edge-LLM/latest/cpp_api.html

## 编译


### 1. 编译接口包

  

```bash

cd /path/bot_project

mkdir -p build/bot_interfaces install/bot_interfaces

cd build/bot_interfaces

source /opt/ros/humble/setup.bash

cmake ../../src/bot_interfaces \

  -DCMAKE_INSTALL_PREFIX=/path/bot_project/install/bot_interfaces \

  -DCMAKE_PREFIX_PATH=/opt/ros/humble

make -j$(nproc)

make install

```
  

### 2. 编译 realsense-ros

  

```bash

cd /path/VLM_project/realsense-ros

mkdir -p build install

cd build

source /opt/ros/humble/setup.bash

cmake .. -DCMAKE_INSTALL_PREFIX=/home/nvidia/VLM_project/realsense-ros/install -DCMAKE_PREFIX_PATH=/opt/ros/humble

make -j$(nproc)

make install

```
### 3. 编译 VLM 节点

```bash

cd /path/VLM_project/bot_project

mkdir -p build/bot_vlm install/bot_vlm

cd build/bot_vlm

source /opt/ros/humble/setup.bash

cmake ../../src/bot_vlm \

  -DCMAKE_INSTALL_PREFIX=/path/VLM_project/bot_project/install/bot_vlm \

  -DCMAKE_PREFIX_PATH="/opt/ros/humble;/path/bot_project/install/bot_interfaces"

make -j$(nproc)

make install

```
### 4. 编译语音识别节点（bot_speech）

语音识别节点需要额外依赖 FFTW3（快速傅里叶变换）用于音频预处理。

```bash

# 安装依赖

sudo apt-get install -y libfftw3-dev libasound2-dev

  

# 编译

cd /path/VLM_project/bot_project

colcon build --packages-select bot_interfaces bot_speech

```
### 5. 编译底盘控制节点

  

```bash

cd /home/nvidia/VLM_project/bot_project

mkdir -p build/bot_chassis install/bot_chassis

cd build/bot_chassis

source /opt/ros/humble/setup.bash

cmake ../../src/bot_chassis \

  -DCMAKE_INSTALL_PREFIX=/home/nvidia/VLM_project/bot_project/install/bot_chassis \

  -DCMAKE_PREFIX_PATH="/opt/ros/humble;/home/nvidia/VLM_project/bot_project/install/bot_interfaces"

make -j$(nproc)

make install

```
### 6. 编译 Codbot 底盘驱动

```bash

cd /path/VLM_project/codbot_base

mkdir -p build install

cd build

source /opt/ros/humble/setup.bash

cmake .. -DCMAKE_INSTALL_PREFIX=/自己小车的/codbot_base/install -DCMAKE_PREFIX_PATH=/opt/ros/humble

make -j$(nproc)

make install

```
## 目录结构

```
bot_project/
├── src/
│   ├── bot_interfaces/             # 自定义消息/服务
│   │   ├── msg/
│   │   │   ├── VLMResult.msg       # VLM 推理结果
│   │   │   └── VelocityCommand.msg # 速度控制指令
│   │   ├── srv/
│   │   │   └── VLMQuery.srv        # VLM 查询服务
│   │   ├── CMakeLists.txt
│   │   └── package.xml
│   │
│   ├── bot_vlm/                    # VLM 推理节点
│   │   ├── include/bot_vlm/
│   │   │   └── vlm_node.hpp
│   │   ├── src/
│   │   │   ├── vlm_node.cpp
│   │   │   └── cuda_helper.cu
│   │   ├── launch/
│   │   ├── config/
│   │   ├── CMakeLists.txt
│   │   └── package.xml
│   │
│   ├── bot_chassis/                # 底盘控制节点
│   │   ├── include/bot_chassis/
│   │   │   └── chassis_node.hpp
│   │   ├── src/
│   │   │   └── chassis_node.cpp
│   │   ├── launch/
│   │   ├── config/
│   │   ├── CMakeLists.txt
│   │   └── package.xml
│   │
│   └── bot_speech/                 # 语音识别节点
│       ├── include/bot_speech/
│       │   ├── speech_node.hpp
│       │   └── audio_preprocessor.hpp
│       ├── src/
│       │   ├── speech_node.cpp
│       │   ├── audio_preprocessor.cpp
│       │   └── cuda_helper.cu
│       ├── launch/
│       ├── config/
│       ├── CMakeLists.txt
│       └── package.xml
│
├── gui/
│   └── vlm_gui.py                  # PyQt5 GUI 控制面板
│
├── launch/
│   └── vlm_pipeline.launch.py      # 完整管线启动文件
│
├── scripts/
│   └── *.sh                        # 启动脚本
│
└── README.md
```
