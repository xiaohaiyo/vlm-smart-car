cmake_minimum_required(VERSION 3.8)
project(bot_speech LANGUAGES CXX CUDA)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 87)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(bot_interfaces REQUIRED)
find_package(CUDAToolkit REQUIRED)

# TensorRT-Edge-LLM 路径（宿主机）
set(TENSORRT_EDGELLM_DIR "/home/nvidia/VLM_project/TensorRT-Edge-LLM")
set(TENSORRT_EDGELLM_BUILD "${TENSORRT_EDGELLM_DIR}/build")
set(TENSORRT_EDGELLM_CPP "${TENSORRT_EDGELLM_DIR}/cpp")

# TensorRT
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS /usr/include/aarch64-linux-gnu /usr/include
  PATH_SUFFIXES tensorrt)
find_library(NVINFER_LIB nvinfer
  HINTS /usr/lib/aarch64-linux-gnu /usr/lib)

# CUDA 链接辅助库
add_library(cuda_helper STATIC src/cuda_helper.cu)
target_include_directories(cuda_helper PRIVATE
  ${TENSORRT_EDGELLM_CPP}
  ${TENSORRT_EDGELLM_DIR}/3rdParty
  ${CUDAToolkit_INCLUDE_DIRS}
  ${TENSORRT_INCLUDE_DIR}
)
set_target_properties(cuda_helper PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
  CUDA_ARCHITECTURES 87
)

# FFTW3
find_library(FFTW3F_LIB fftw3f REQUIRED)

# 语音识别节点
add_executable(speech_node src/speech_node.cpp src/audio_preprocessor.cpp)
target_include_directories(speech_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${TENSORRT_EDGELLM_CPP}
  ${TENSORRT_EDGELLM_DIR}/3rdParty
  ${CUDAToolkit_INCLUDE_DIRS}
  ${TENSORRT_INCLUDE_DIR}
)

target_link_libraries(speech_node
  -Wl,--start-group
  ${TENSORRT_EDGELLM_BUILD}/cpp/libedgellmCore.a
  -Wl,--end-group
  ${TENSORRT_EDGELLM_BUILD}/libNvInfer_edgellm_plugin.so
  cuda_helper
  ${NVINFER_LIB}
  CUDA::cudart
  CUDA::cuda_driver
  ${FFTW3F_LIB}
  dl
  asound
  -Wl,--unresolved-symbols=ignore-in-shared-libs
)

set_target_properties(speech_node PROPERTIES CUDA_ARCHITECTURES 87)

ament_target_dependencies(speech_node
  rclcpp std_msgs bot_interfaces
)

install(TARGETS speech_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
