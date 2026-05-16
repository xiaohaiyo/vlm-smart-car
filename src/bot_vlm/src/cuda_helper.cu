// CUDA 链接辅助（用于正确链接 TensorRT-Edge-LLM 设备代码）
#include <cuda_runtime.h>
namespace bot_vlm { void cuda_link_helper() {} }
