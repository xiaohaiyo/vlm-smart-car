/**
 * @file vlm_node.hpp
 * @brief VLM 推理节点头文件
 *
 * 功能说明：
 * - 订阅 RealSense D435i 相机图像
 * - 使用 TensorRT-Edge-LLM 进行视觉语言模型推理
 * - 发布推理结果到 ROS 话题
 * - 提供同步查询服务
 *
 * 推理流程：
 *   Camera → ROS Image → TensorRT-Edge-LLM → VLM Result
 *
 * 优化特性：
 * - 跳帧策略：只处理最新帧
 * - GPU 缓冲区预分配
 * - 异步推理工作线程
 * - 直接 RGB 数据传递
 */

#ifndef BOT_VLM__VLM_NODE_HPP_
#define BOT_VLM__VLM_NODE_HPP_

#include <memory>
#include <string>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/string.hpp"
#include "bot_interfaces/msg/vlm_result.hpp"
#include "bot_interfaces/msg/velocity_command.hpp"
#include "bot_interfaces/srv/vlm_query.hpp"

// TensorRT-Edge-LLM 推理引擎头文件
#include "runtime/llmInferenceSpecDecodeRuntime.h"
#include "runtime/imageUtils.h"

// CUDA 头文件
#include <cuda_runtime.h>

namespace bot_vlm
{

/**
 * @struct InferenceRequest
 * @brief 推理请求结构体
 *
 * 封装单次 VLM 推理请求所需的数据。
 */
struct InferenceRequest
{
  sensor_msgs::msg::Image::SharedPtr image;  ///< ROS 图像消息（RGB8 格式）
  std::string prompt;                        ///< 用户查询提示词
};

/**
 * @class VlmNode
 * @brief VLM 推理节点
 *
 * 主要功能：
 * 1. 订阅相机图像
 * 2. 使用 TensorRT-Edge-LLM 进行 VLM 推理
 * 3. 发布推理结果
 * 4. 提供同步查询服务
 *
 * 使用方法：
 *   ros2 run bot_vlm vlm_node
 *   ros2 launch bot_project vlm_pipeline.launch.py
 */
class VlmNode : public rclcpp::Node
{
public:
  /**
   * @brief 构造函数
   * @param options 节点配置选项
   */
  explicit VlmNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

  /**
   * @brief 析构函数
   *
   * 清理 GPU 资源和工作线程。
   */
  virtual ~VlmNode();

private:
  /**
   * @brief 初始化 TensorRT-Edge-LLM 引擎
   *
   * 在独立线程中执行，避免阻塞节点启动。
   */
  void initialize_engine();

  /**
   * @brief 图像回调函数（跳帧策略）
   * @param msg 接收到的 ROS 图像消息
   *
   * 清空队列中的旧帧，只保留最新帧。
   */
  void image_callback(const sensor_msgs::msg::Image::SharedPtr msg);

  /**
   * @brief 语音识别结果回调
   * @param msg 语音识别文字消息
   *
   * 收到语音结果后触发 VLM 推理。
   */
  void speech_callback(const std_msgs::msg::String::SharedPtr msg);

  /**
   * @brief 处理同步查询请求
   * @param req 服务请求
   * @param res 服务响应
   */
  void handle_query(
    const std::shared_ptr<bot_interfaces::srv::VLMQuery::Request> req,
    std::shared_ptr<bot_interfaces::srv::VLMQuery::Response> res);

  /**
   * @brief 推理工作线程
   *
   * 持续从队列中获取推理请求并执行。
   */
  void inference_worker();

  /**
   * @brief 执行 VLM 推理
   * @param image ROS 图像消息（RGB8 格式）
   * @param prompt 用户提示词
   * @param output 输出的推理结果
   * @return 推理是否成功
   *
   * 推理流程：
   * 1. 构建 TensorRT-Edge-LLM 请求
   * 2. 将 ROS 图像转换为 ImageData（直接引用，无拷贝）
   * 3. 执行推理
   * 4. 返回推理结果
   */
  bool run_inference(const sensor_msgs::msg::Image::SharedPtr & image,
                     const std::string & prompt, std::string & output);

  /**
   * @brief 预热 System Prompt
   *
   * 在引擎初始化后调用，将系统提示词缓存到 KV Cache 中，
   * 后续推理时无需重复计算系统提示词，提升推理速度。
   */
  void build_system_prompt_cache();

  // ===== ROS 接口 =====
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;  ///< 图像订阅者
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr speech_sub_;   ///< 语音识别结果订阅者
  rclcpp::Publisher<bot_interfaces::msg::VLMResult>::SharedPtr result_pub_;  ///< 结果发布者
  rclcpp::Publisher<bot_interfaces::msg::VelocityCommand>::SharedPtr vel_pub_;  ///< 速度指令发布者
  rclcpp::Service<bot_interfaces::srv::VLMQuery>::SharedPtr query_srv_;  ///< 查询服务

  // ===== 语音驱动推理 =====
  std::string latest_speech_text_;        ///< 最新语音识别结果
  std::mutex speech_mutex_;               ///< 语音数据互斥锁
  sensor_msgs::msg::Image::SharedPtr latest_image_;  ///< 最新图像缓存

  // ===== 推理队列 =====
  std::queue<InferenceRequest> queue_;  ///< 推理请求队列
  std::mutex mutex_;                    ///< 队列互斥锁
  std::condition_variable cv_;          ///< 条件变量
  std::unique_ptr<std::thread> worker_; ///< 推理工作线程
  std::atomic<bool> running_{false};    ///< 线程运行标志

  // ===== 持续速度发布 =====
  float last_linear_vel_{0.0f};         ///< 上次线速度
  float last_angular_vel_{0.0f};        ///< 上次角速度
  rclcpp::TimerBase::SharedPtr vel_timer_;  ///< 速度发布定时器

  // ===== TensorRT-Edge-LLM 引擎 =====
  std::unique_ptr<trt_edgellm::rt::LLMInferenceSpecDecodeRuntime> runtime_;  ///< 推理运行时
  cudaStream_t stream_;  ///< CUDA 流

  // ===== GPU 缓冲区 =====
  uint8_t* gpu_buf_;    ///< GPU 内存指针
  uint8_t* pin_buf_;    ///< Pinned 内存指针
  size_t buf_size_;     ///< 缓冲区大小

  // ===== 参数 =====
  std::string engine_dir_;       ///< LLM 引擎目录
  std::string visual_dir_;       ///< 视觉引擎目录
  int max_tokens_;               ///< 最大生成 token 数
  float temperature_;            ///< 采样温度
  float top_p_;                  ///< Top-P 采样参数
  int64_t top_k_;                ///< Top-K 采样参数
  bool engine_ok_;               ///< 引擎是否就绪
  std::string default_prompt_;   ///< 默认提示词

  // ===== 记忆功能 =====
  std::string last_output_;      ///< 上次推理输出（用于记忆）
  std::mutex output_mutex_;      ///< 输出互斥锁
};

}  // namespace bot_vlm

#endif  // BOT_VLM__VLM_NODE_HPP_
