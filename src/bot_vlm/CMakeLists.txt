cmake_minimum_required(VERSION 3.8)
project(bot_vlm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 87)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(bot_interfaces REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)

# TensorRT-Edge-LLM 路径（宿主机）
set(TRT_EDGELLM_DIR "/home/nvidia/VLM_project/TensorRT-Edge-LLM")
set(TRT_EDGELLM_BUILD "${TRT_EDGELLM_DIR}/build")
set(TRT_EDGELLM_CPP "${TRT_EDGELLM_DIR}/cpp")

find_library(NVINFER_LIB nvinfer HINTS /usr/lib/aarch64-linux-gnu)

# CUDA 链接辅助
add_library(cuda_helper STATIC src/cuda_helper.cu)
set_target_properties(cuda_helper PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# VLM 节点
add_executable(vlm_node src/vlm_node.cpp)
target_include_directories(vlm_node PRIVATE
  include
  ${TRT_EDGELLM_CPP}
  ${TRT_EDGELLM_DIR}/3rdParty
  ${CUDAToolkit_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(vlm_node
  -Wl,--start-group
  ${TRT_EDGELLM_BUILD}/cpp/libedgellmCore.a
  -Wl,--end-group
  ${TRT_EDGELLM_BUILD}/libNvInfer_edgellm_plugin.so
  cuda_helper
  ${NVINFER_LIB}
  CUDA::cudart
  CUDA::cuda_driver
  ${OpenCV_LIBS}
  dl
  -Wl,--unresolved-symbols=ignore-in-shared-libs
)
ament_target_dependencies(vlm_node rclcpp sensor_msgs std_msgs bot_interfaces)

install(TARGETS vlm_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})

ament_package()
