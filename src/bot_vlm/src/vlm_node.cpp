/**
 * @file vlm_node.cpp
 * @brief VLM 推理节点实现
 *
 * 功能说明：
 * - 订阅 RealSense D435i 相机图像
 * - 使用 TensorRT-Edge-LLM 进行视觉语言模型推理
 * - 发布推理结果到 ROS 话题
 *
 * 数据流：
 *   Camera → /camera/color/image_raw → VLM Node → /vlm/result
 *
 * 优化特性：
 * - 跳帧策略：只处理最新帧，避免队列积压
 * - GPU 缓冲区预分配：避免每帧分配内存
 * - 异步推理：独立工作线程，不阻塞 ROS 回调
 * - 直接 RGB 数据：跳过 JPEG 编解码，减少 CPU 开销
 */

#include "bot_vlm/vlm_node.hpp"
#include "common/trtUtils.h"

namespace bot_vlm
{

/**
 * @brief 构造函数
 *
 * 初始化 VLM 节点，包括：
 * 1. 声明并获取参数
 * 2. 创建 ROS 发布者、订阅者、服务
 * 3. 分配 GPU 缓冲区
 * 4. 异步初始化 TensorRT 引擎
 */
VlmNode::VlmNode(const rclcpp::NodeOptions & options)
: Node("vlm_node", options), engine_ok_(false),
  gpu_buf_(nullptr), pin_buf_(nullptr), buf_size_(0)
{
  // ===== 声明并获取参数 =====
  // LLM 引擎目录（包含 tokenizer、config 等）
  engine_dir_ = declare_parameter<std::string>("engine_dir",
    "/home/nvidia/VLM_project/Cosmos_ONNXINT4/engine/llm");

  // 视觉编码器引擎目录
  visual_dir_ = declare_parameter<std::string>("visual_dir",
    "/home/nvidia/VLM_project/Cosmos_ONNXINT4/engine/visual");

  // 推理参数
  max_tokens_ = declare_parameter<int>("max_tokens", 128);      // 最大生成 token 数
  temperature_ = declare_parameter<double>("temperature", 0.3); // 采样温度（越低越稳定）
  top_p_ = declare_parameter<double>("top_p", 0.8);            // Top-P 采样参数
  top_k_ = declare_parameter<int64_t>("top_k", 20);            // Top-K 采样参数

default_prompt_ = declare_parameter<std::string>(
"default_prompt",
R"(
你是机器人控制器。

当前图像来自小车前方摄像头。

根据图像中的环境和用户指令，
自主决定小车的linear_vel值,angular_vel值，控制小车移动,以及回答用户的指令。

控制规则：
- linear_vel ∈ [-0.01, 0.01]
  - >0 前进
  - <0 后退

- angular_vel ∈ [-0.5, 0.5]
  - >0 左转
  - <0 右转

要求：
- 目标在左侧时，应适当左转
- 目标在右侧时，应适当右转
- 目标在前方时，应向前移动
- 障碍物较近时，应减速或停止
- 根据目标距离和方向，自主调整速度大小
- 没有运动意图时，两个速度必须为 0.0

只能输出 JSON。

输出格式：
{
  "用户指令":"用户原话",
  "linear_vel":,
  "angular_vel":,
  "回复内容":",
}
)"
);
  // ===== 创建 ROS 接口 =====
  // 发布 VLM 推理结果
  result_pub_ = create_publisher<bot_interfaces::msg::VLMResult>("vlm/result", 10);

  // 发布速度控制指令
  vel_pub_ = create_publisher<bot_interfaces::msg::VelocityCommand>("vlm/velocity", 10);

  // 提供同步查询服务
  query_srv_ = create_service<bot_interfaces::srv::VLMQuery>("vlm/query",
    std::bind(&VlmNode::handle_query, this, std::placeholders::_1, std::placeholders::_2));

  // 订阅相机图像（QoS 队列=1，避免积压）
  rclcpp::QoS qos(1);
  image_sub_ = create_subscription<sensor_msgs::msg::Image>(
    "/camera/color/image_raw", qos,
    std::bind(&VlmNode::image_callback, this, std::placeholders::_1));

  // 订阅语音识别结果
  speech_sub_ = create_subscription<std_msgs::msg::String>(
    "/speech/text", 10,
    std::bind(&VlmNode::speech_callback, this, std::placeholders::_1));

  // ===== 持续速度发布定时器（10Hz）=====
  // 持续发布上次的速度指令，直到收到新指令
  vel_timer_ = create_wall_timer(
    std::chrono::milliseconds(100),  // 10Hz
    [this]() {
      if (!engine_ok_) return;
      bot_interfaces::msg::VelocityCommand vel_cmd;
      vel_cmd.stamp = now();
      vel_cmd.linear_velocity = last_linear_vel_;
      vel_cmd.angular_velocity = last_angular_vel_;
      vel_cmd.confidence = 0.0f;
      vel_cmd.description = "";
      vel_pub_->publish(vel_cmd);
    });

  // ===== 分配 GPU 缓冲区 =====
  // 预分配 640x480x3 的 GPU 内存，避免每帧分配
  cudaStreamCreate(&stream_);
  buf_size_ = 640 * 480 * 3;
  cudaMalloc(&gpu_buf_, buf_size_);      // GPU 内存
  cudaMallocHost(&pin_buf_, buf_size_);  // Pinned 内存（提高传输效率）

  RCLCPP_INFO(get_logger(), "GPU 缓冲区已分配 (640x480x3)");

  // 异步初始化引擎（避免阻塞节点启动）
  std::thread(&VlmNode::initialize_engine, this).detach();

  RCLCPP_INFO(get_logger(), "VLM 节点已初始化");
  RCLCPP_INFO(get_logger(), "  引擎: %s", engine_dir_.c_str());
  RCLCPP_INFO(get_logger(), "  订阅: /camera/color/image_raw");
  RCLCPP_INFO(get_logger(), "  发布: vlm/result, vlm/velocity");
  RCLCPP_INFO(get_logger(), "  模式: 小车大脑（速度控制）");
}

/**
 * @brief 析构函数
 *
 * 清理资源：
 * 1. 停止工作线程
 * 2. 释放 GPU 内存
 * 3. 销毁 CUDA 流
 */
VlmNode::~VlmNode()
{
  running_ = false;
  cv_.notify_all();
  if (worker_ && worker_->joinable()) worker_->join();
  if (gpu_buf_) cudaFree(gpu_buf_);
  if (pin_buf_) cudaFreeHost(pin_buf_);
  cudaStreamDestroy(stream_);
}

/**
 * @brief 初始化 TensorRT-Edge-LLM 引擎
 *
 * 在独立线程中执行，避免阻塞节点启动：
 * 1. 加载 EdgeLLM 插件库
 * 2. 创建推理运行时
 * 3. 捕获 CUDA 图以优化性能
 * 4. 启动推理工作线程
 */
void VlmNode::initialize_engine()
{
  RCLCPP_INFO(get_logger(), "正在加载 TensorRT-Edge-LLM 引擎...");
  try {
    // 加载 EdgeLLM 插件库
    trt_edgellm::loadEdgellmPluginLib();

    // 创建 LoRA 权重映射（不使用 LoRA）
    std::unordered_map<std::string, std::string> lora;

    // 创建推理运行时（标准模式，不使用 Eagle 投机解码）
    runtime_ = std::make_unique<trt_edgellm::rt::LLMInferenceSpecDecodeRuntime>(
      engine_dir_, visual_dir_, lora, stream_);

    // 捕获 CUDA 图以优化解码性能
    runtime_->captureDecodingCUDAGraph(stream_);

    engine_ok_ = true;
    running_ = true;

    // 预热 System Prompt（缓存到 KV Cache）
   build_system_prompt_cache();

    // 启动推理工作线程
    worker_ = std::make_unique<std::thread>(&VlmNode::inference_worker, this);

    RCLCPP_INFO(get_logger(), "引擎加载成功");
  } catch (const std::exception & e) {
    RCLCPP_ERROR(get_logger(), "引擎加载失败: %s", e.what());
  }
}

/**
 * @brief 处理同步查询请求
 *
 * 将请求中的图像数据转换为 ROS 消息格式，
 * 然后调用 run_inference 执行推理。
 */
void VlmNode::handle_query(
  const std::shared_ptr<bot_interfaces::srv::VLMQuery::Request> req,
  std::shared_ptr<bot_interfaces::srv::VLMQuery::Response> res)
{
  // 创建 ROS 图像消息
  auto msg = std::make_shared<sensor_msgs::msg::Image>();
  msg->header.stamp = now();
  msg->width = req->image_width;
  msg->height = req->image_height;
  msg->encoding = "rgb8";
  msg->step = req->image_width * 3;
  msg->data = req->image_data;

  // 执行推理
  std::string out;
  res->success = run_inference(msg, req->prompt, out);
  res->response = out;
  res->confidence = res->success ? 1.0f : 0.0f;
}

/**
 * @brief 图像回调函数（跳帧策略）
 *
 * 当新图像到达时：
 * 1. 清空队列中的旧帧
 * 2. 只保留最新帧
 * 3. 通知工作线程处理
 *
 * 这样可以避免队列积压，确保总是处理最新图像。
 */
void VlmNode::image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
{
  if (!engine_ok_) return;

  // 缓存最新图像（不触发推理，等待语音输入）
  std::lock_guard<std::mutex> lock(mutex_);
  latest_image_ = msg;
}

/**
 * @brief 语音识别结果回调
 *
 * 收到语音结果后，使用当前缓存的图像触发 VLM 推理。
 */
void VlmNode::speech_callback(const std_msgs::msg::String::SharedPtr msg)
{
  if (!engine_ok_ || msg->data.empty()) return;

  RCLCPP_INFO(get_logger(), "收到语音指令: %s", msg->data.c_str());

  // 保存语音文本
  {
    std::lock_guard<std::mutex> lock(speech_mutex_);
    latest_speech_text_ = msg->data;
  }

  // 有语音输入时，触发 VLM 推理
  std::lock_guard<std::mutex> lock(mutex_);
  if (latest_image_) {
    // 清空旧请求，只处理最新
    while (!queue_.empty()) queue_.pop();
    queue_.push({latest_image_, msg->data});
    cv_.notify_one();
  }
}

/**
 * @brief 推理工作线程
 *
 * 持续从队列中获取推理请求并执行：
 * 1. 等待新请求（条件变量）
 * 2. 调用 run_inference 执行推理
 * 3. 解析速度指令
 * 4. 发布推理结果和速度指令
 */
void VlmNode::inference_worker()
{
  RCLCPP_INFO(get_logger(), "推理工作线程已启动");
  while (running_) {
    InferenceRequest req;
    auto t_queue_start = std::chrono::steady_clock::now();
    {
      // 等待新请求
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [this] { return !queue_.empty() || !running_; });
      if (!running_) break;
      req = queue_.front();
      queue_.pop();
    }
    auto t_queue_end = std::chrono::steady_clock::now();
    float queue_wait_ms = std::chrono::duration<float, std::milli>(t_queue_end - t_queue_start).count();

    // 记录图像时间戳（计算图像从采集到现在的延迟）
    auto t_image_stamp = req.image->header.stamp;
    auto t_now = this->now();
    int64_t image_ns = static_cast<int64_t>(t_image_stamp.sec) * 1000000000LL + t_image_stamp.nanosec;
    int64_t now_ns = static_cast<int64_t>(t_now.seconds()) * 1000000000LL + t_now.nanoseconds();
    float image_age_ms = static_cast<float>(now_ns - image_ns) / 1000000.0f;

    // ===== 构建提示词 =====
    auto t_prompt_start = std::chrono::steady_clock::now();
    std::string full_prompt = default_prompt_;
    {
      std::lock_guard<std::mutex> lock(speech_mutex_);
      if (!latest_speech_text_.empty()) {
        full_prompt += "\n用户内容为: " + latest_speech_text_;
      }
    }
    auto t_prompt_end = std::chrono::steady_clock::now();
    float prompt_ms = std::chrono::duration<float, std::milli>(t_prompt_end - t_prompt_start).count();

    // ===== 执行推理 =====
    std::string output;
    auto t_infer_start = std::chrono::steady_clock::now();
    bool ok = run_inference(req.image, full_prompt, output);
    auto t_infer_end = std::chrono::steady_clock::now();
    float inference_ms = std::chrono::duration<float, std::milli>(t_infer_end - t_infer_start).count();

    // 保存本次输出（记忆）
    if (ok && !output.empty()) {
      std::lock_guard<std::mutex> lock(output_mutex_);
      last_output_ = output;
    }

    // ===== 发布结果 =====
    auto t_publish_start = std::chrono::steady_clock::now();
    bot_interfaces::msg::VLMResult result;
    result.header = req.image->header;
    result.query = req.prompt;
    result.response = output;
    result.confidence = ok ? 1.0f : 0.0f;
    result_pub_->publish(result);
    auto t_publish_end = std::chrono::steady_clock::now();
    float publish_ms = std::chrono::duration<float, std::milli>(t_publish_end - t_publish_start).count();

    // ===== 性能汇总 =====
    float total_ms = std::chrono::duration<float, std::milli>(t_publish_end - t_queue_start).count();
    RCLCPP_INFO(get_logger(),
      "===== VLM 性能报告 =====\n"
      "  队列等待: %.1f ms\n"
      "  图像延迟: %.1f ms (从采集到现在)\n"
      "  提示构建: %.1f ms\n"
      "  VLM推理:  %.1f ms (主要瓶颈)\n"
      "  结果发布: %.1f ms\n"
      "  总耗时:   %.1f ms",
      queue_wait_ms, image_age_ms, prompt_ms, inference_ms, publish_ms, total_ms);

    // 解析 JSON 格式的速度指令
    if (ok) {
      float linear_vel = 0.0f;
      float angular_vel = 0.0f;

      // 解析 JSON 格式: {"linear_vel": 0.0, "angular_vel": 0.0, ...}
      // 查找 "linear_vel" 字段
      size_t linear_pos = output.find("\"linear_vel\"");
      size_t angular_pos = output.find("\"angular_vel\"");

      if (linear_pos != std::string::npos && angular_pos != std::string::npos) {
        try {
          // 提取线速度: 查找 "linear_vel": 后面的数值
          size_t linear_start = output.find(":", linear_pos);
          if (linear_start != std::string::npos) {
            linear_start++;  // 跳过冒号
            // 跳过空格
            while (linear_start < output.length() && output[linear_start] == ' ') linear_start++;
            // 查找数值结束位置（逗号或右花括号）
            size_t linear_end = output.find_first_of(",}", linear_start);
            if (linear_end != std::string::npos) {
              std::string linear_str = output.substr(linear_start, linear_end - linear_start);
              // 去除首尾空格和引号
              linear_str.erase(0, linear_str.find_first_not_of(" \t\n\r\""));
              linear_str.erase(linear_str.find_last_not_of(" \t\n\r\"") + 1);
              if (!linear_str.empty() && linear_str != "null") {
                linear_vel = std::stof(linear_str);
              }
            }
          }

          // 提取角速度: 查找 "angular_vel": 后面的数值
          size_t angular_start = output.find(":", angular_pos);
          if (angular_start != std::string::npos) {
            angular_start++;  // 跳过冒号
            // 跳过空格
            while (angular_start < output.length() && output[angular_start] == ' ') angular_start++;
            // 查找数值结束位置（逗号或右花括号）
            size_t angular_end = output.find_first_of(",}", angular_start);
            if (angular_end != std::string::npos) {
              std::string angular_str = output.substr(angular_start, angular_end - angular_start);
              // 去除首尾空格和引号
              angular_str.erase(0, angular_str.find_first_not_of(" \t\n\r\""));
              angular_str.erase(angular_str.find_last_not_of(" \t\n\r\"") + 1);
              if (!angular_str.empty() && angular_str != "null") {
                angular_vel = std::stof(angular_str);
              }
            }
          }
        } catch (const std::exception & e) {
          RCLCPP_WARN(get_logger(), "JSON 速度解析失败: %s", e.what());
        }
      }

      // 限制范围（安全保护）
      linear_vel = std::max(-0.01f, std::min(0.01f, linear_vel));
      angular_vel = std::max(-0.5f, std::min(0.5f, angular_vel));

      // 保存速度值（定时器会持续发布）
      last_linear_vel_ = linear_vel;
      last_angular_vel_ = angular_vel;

      RCLCPP_INFO(get_logger(), "速度指令: linear=%.4f, angular=%.4f",
        linear_vel, angular_vel);
    }

    RCLCPP_INFO(get_logger(), "VLM 推理完成:\n%s", output.c_str());
  }
}

/**
 * @brief 执行 VLM 推理
 *
 * 推理流程：
 * 1. 构建 TensorRT-Edge-LLM 请求
 * 2. 将 ROS 图像转换为 ImageData（直接引用，无拷贝）
 * 3. 执行推理
 * 4. 返回推理结果
 *
 * @param image ROS 图像消息（RGB8 格式）
 * @param prompt 用户提示词
 * @param output 输出的推理结果
 * @return 推理是否成功
 */
bool VlmNode::run_inference(const sensor_msgs::msg::Image::SharedPtr & image,
                             const std::string & prompt, std::string & output)
{
  if (!engine_ok_ || !runtime_) return false;

  try {
    // ===== 构建请求 =====
    auto t1 = std::chrono::steady_clock::now();

    trt_edgellm::rt::LLMGenerationRequest req;
    req.temperature = temperature_;
    req.topP = top_p_;
    req.topK = top_k_;
    req.maxGenerateLength = max_tokens_;
    req.applyChatTemplate = true;

    trt_edgellm::rt::LLMGenerationRequest::Request r;

    // user message
    trt_edgellm::rt::Message user_msg;
    user_msg.role = "user";

    // 图像内容
    trt_edgellm::rt::Message::MessageContent img_c;
    img_c.type = "image";
    img_c.content = "";
    user_msg.contents.push_back(img_c);

    // 文本内容
    trt_edgellm::rt::Message::MessageContent txt_c;
    txt_c.type = "text";
    txt_c.content = prompt;
    user_msg.contents.push_back(txt_c);

    r.messages.push_back(user_msg);

    // 图像数据（零拷贝，直接引用 ROS 消息内存）
    int w = image->width, h = image->height;
    trt_edgellm::rt::Tensor imgTensor(
      const_cast<uint8_t*>(image->data.data()),
      {h, w, 3},
      trt_edgellm::rt::DeviceType::kCPU,
      nvinfer1::DataType::kUINT8,
      "ros_image");

    trt_edgellm::rt::imageUtils::ImageData img_data(std::move(imgTensor));
    r.imageBuffers.push_back(std::move(img_data));
    req.requests = {r};

    auto t2 = std::chrono::steady_clock::now();
    float build_ms = std::chrono::duration<float, std::milli>(t2 - t1).count();

    // ===== 执行推理（核心瓶颈）=====
    trt_edgellm::rt::LLMGenerationResponse res;
    bool ok = runtime_->handleRequest(req, res, stream_);

    auto t3 = std::chrono::steady_clock::now();
    float infer_ms = std::chrono::duration<float, std::milli>(t3 - t2).count();

    // ===== 解析结果 =====
    if (ok && !res.outputTexts.empty()) {
      output = res.outputTexts[0];

      auto t4 = std::chrono::steady_clock::now();
      float decode_ms = std::chrono::duration<float, std::milli>(t4 - t3).count();

      RCLCPP_DEBUG(get_logger(),
        "run_inference 细分: 请求构建=%.1fms, TRT推理=%.1fms, 结果解码=%.1fms",
        build_ms, infer_ms, decode_ms);

      return true;
    }
    return false;

  } catch (const std::exception & e) {
    RCLCPP_ERROR(get_logger(), "推理错误: %s", e.what());
    return false;
  }
}

/**
 * @brief 预热 System Prompt
 *
 * 将系统提示词发送给引擎，缓存到 KV Cache 中。
 * 后续推理时，系统提示词部分无需重复计算，显著提升推理速度。
 */
void VlmNode::build_system_prompt_cache()
{
  RCLCPP_INFO(get_logger(), "构建 System Prompt Cache...");

  trt_edgellm::rt::LLMGenerationRequest req;
  req.temperature = 0.0;
  req.maxGenerateLength = 1;
  req.applyChatTemplate = true;
  
  // 【关键修改 1】：开启保存系统提示词 KV 缓存的标志！
  // 注意：这里的变量名取决于 C++ API 头文件的具体拼写，
  // 通常是 saveSystemPromptKVCache 或 save_system_prompt_kv_cache
  req.saveSystemPromptKVCache = true; 

  trt_edgellm::rt::LLMGenerationRequest::Request r;

  trt_edgellm::rt::Message msg;
  msg.role = "system";
  trt_edgellm::rt::Message::MessageContent txt;
  txt.type = "text";
  txt.content = default_prompt_; // 缓存这段文本

  msg.contents.push_back(txt);
  r.messages = {msg};
  req.requests = {r};

  trt_edgellm::rt::LLMGenerationResponse res;
  runtime_->handleRequest(req, res, stream_);

  RCLCPP_INFO(get_logger(), "System Prompt Cache 构建完成");
}

}  // namespace bot_vlm

/**
 * @brief 主函数
 */
int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<bot_vlm::VlmNode>());
  rclcpp::shutdown();
  return 0;
}
