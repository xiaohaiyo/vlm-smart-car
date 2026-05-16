/**
 * @file speech_node.cpp
 * @brief 语音识别节点实现
 *
 * 数据流：
 *   蓝牙耳机 → ALSA → 音频缓冲区 → Qwen3-ASR → 文字 → VLM 节点
 *
 * 功能：
 * 1. 从蓝牙耳机/麦克风采集音频（16kHz, mono, float32）
 * 2. VAD 语音活动检测，只处理有语音的片段
 * 3. 使用 Qwen3-ASR TensorRT 引擎进行语音识别
 * 4. 发布识别结果到 ROS 话题
 */

#include "bot_speech/speech_node.hpp"
#include "bot_speech/audio_preprocessor.hpp"
#include "common/trtUtils.h"

#include <cmath>
#include <cstring>
#include <chrono>

namespace bot_speech
{

SpeechNode::SpeechNode(const rclcpp::NodeOptions & options)
: Node("speech_node", options)
{
  // 参数
  engine_dir_ = declare_parameter<std::string>("engine_dir",
    "/home/nvidia/VLM_project/Qwen3-ASR-0.6B_onnx/engine/llm");
  audio_dir_ = declare_parameter<std::string>("audio_dir",
    "/home/nvidia/VLM_project/Qwen3-ASR-0.6B_onnx/engine/audio");
  device_name_ = declare_parameter<std::string>("device_name", "default");
  sample_rate_ = declare_parameter<int>("sample_rate", 16000);
  chunk_duration_ms_ = declare_parameter<int>("chunk_duration_ms", 256);
  vad_threshold_ = declare_parameter<double>("vad_threshold", 0.03);

  // 发布者
  text_pub_ = create_publisher<std_msgs::msg::String>(
    "speech/text", 10);
  command_pub_ = create_publisher<bot_interfaces::msg::VLMResult>(
    "speech/command", 10);

  // 初始化 CUDA
  cudaStreamCreate(&stream_);

  // 初始化音频采集（通过 PulseAudio）
  std::string pa_device = "pulse";
  if (device_name_.find("bluez") != std::string::npos) {
    // 蓝牙设备通过 PulseAudio 访问
    pa_device = device_name_;
  }

  snd_pcm_hw_params_t* params;
  int err = snd_pcm_open(&pcm_handle_, pa_device.c_str(), SND_PCM_STREAM_CAPTURE, 0);
  if (err < 0) {
    // 尝试使用默认设备
    RCLCPP_WARN(get_logger(), "无法打开 %s，尝试默认设备", pa_device.c_str());
    err = snd_pcm_open(&pcm_handle_, "default", SND_PCM_STREAM_CAPTURE, 0);
  }

  if (err < 0) {
    RCLCPP_ERROR(get_logger(), "无法打开音频设备: %s", snd_strerror(err));
    return;
  }

  snd_pcm_hw_params_alloca(&params);
  snd_pcm_hw_params_any(pcm_handle_, params);
  snd_pcm_hw_params_set_access(pcm_handle_, params, SND_PCM_ACCESS_RW_INTERLEAVED);
  snd_pcm_hw_params_set_format(pcm_handle_, params, SND_PCM_FORMAT_FLOAT_LE);
  snd_pcm_hw_params_set_channels(pcm_handle_, params, 1);
  snd_pcm_hw_params_set_rate_near(pcm_handle_, params, &sample_rate_, 0);
  snd_pcm_hw_params(pcm_handle_, params);
  snd_pcm_prepare(pcm_handle_);

  RCLCPP_INFO(get_logger(), "音频设备已初始化: %s (%d Hz)", pa_device.c_str(), sample_rate_);

  // 异步初始化引擎
  std::thread(&SpeechNode::initialize_engine, this).detach();

  // 启动采集和识别线程
  running_ = true;
  capture_thread_ = std::make_unique<std::thread>(&SpeechNode::audio_capture_thread, this);
  asr_thread_ = std::make_unique<std::thread>(&SpeechNode::asr_worker_thread, this);

  RCLCPP_INFO(get_logger(), "语音节点已初始化");
  RCLCPP_INFO(get_logger(), "  ASR 引擎: %s", engine_dir_.c_str());
  RCLCPP_INFO(get_logger(), "  发布: speech/text, speech/command");
}

SpeechNode::~SpeechNode()
{
  running_ = false;
  queue_cv_.notify_all();
  if (capture_thread_ && capture_thread_->joinable()) capture_thread_->join();
  if (asr_thread_ && asr_thread_->joinable()) asr_thread_->join();
  if (pcm_handle_) snd_pcm_close(pcm_handle_);
  if (stream_) cudaStreamDestroy(stream_);
}

void SpeechNode::initialize_engine()
{
  RCLCPP_INFO(get_logger(), "正在加载 ASR 引擎...");
  try {
    trt_edgellm::loadEdgellmPluginLib();
    std::unordered_map<std::string, std::string> lora;

    runtime_ = std::make_unique<trt_edgellm::rt::LLMInferenceSpecDecodeRuntime>(
      engine_dir_, audio_dir_, lora, stream_);

    engine_ok_ = true;
    RCLCPP_INFO(get_logger(), "ASR 引擎加载成功");
  } catch (const std::exception & e) {
    RCLCPP_ERROR(get_logger(), "ASR 引擎加载失败: %s", e.what());
  }
}

void SpeechNode::audio_capture_thread()
{
  RCLCPP_INFO(get_logger(), "音频采集线程已启动（VAD 模式）");

  int chunk_samples = sample_rate_ * chunk_duration_ms_ / 1000;
  std::vector<float> buffer(chunk_samples);

  while (running_) {
    // 读取音频数据
    snd_pcm_readi(pcm_handle_, buffer.data(), chunk_samples);

    // 计算音量（RMS）
    float rms = 0.0f;
    for (auto s : buffer) {
      rms += s * s;
    }
    rms = std::sqrt(rms / buffer.size());

    // VAD 状态机
    switch (vad_state_) {
      case VadState::SILENCE:
        if (rms > vad_threshold_) {
          // 检测到语音，开始累积
          vad_state_ = VadState::SPEAKING;
          speech_buffer_ = buffer;
          silence_frames_ = 0;
          RCLCPP_DEBUG(get_logger(), "检测到语音开始");
        }
        break;

      case VadState::SPEAKING:
        // 追加到语音缓冲区
        speech_buffer_.insert(speech_buffer_.end(), buffer.begin(), buffer.end());

        if (rms < vad_threshold_) {
          silence_frames_++;
          if (silence_frames_ >= SILENCE_THRESHOLD) {
            // 静音足够长，一句话说完了
            AudioChunk chunk;
            chunk.data = std::move(speech_buffer_);
            chunk.sample_rate = sample_rate_;

            std::lock_guard<std::mutex> lock(queue_mutex_);
            audio_queue_.push(std::move(chunk));
            queue_cv_.notify_one();

            // 重置状态
            speech_buffer_.clear();
            vad_state_ = VadState::SILENCE;
            silence_frames_ = 0;
            RCLCPP_DEBUG(get_logger(), "检测到语音结束，送入队列");
          }
        } else {
          silence_frames_ = 0;
        }
        break;
    }
  }
}

void SpeechNode::asr_worker_thread()
{
  RCLCPP_INFO(get_logger(), "ASR 工作线程已启动");

  while (running_) {
    AudioChunk chunk;
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      queue_cv_.wait(lock, [this] { return !audio_queue_.empty() || !running_; });
      if (!running_) break;
      chunk = audio_queue_.front();
      audio_queue_.pop();
    }

    if (!engine_ok_) continue;

    // 计算音频时长
    float audio_duration_ms = static_cast<float>(chunk.data.size()) / chunk.sample_rate * 1000.0f;

    std::string text;
    auto t_asr_start = std::chrono::steady_clock::now();
    bool ok = run_asr(chunk.data, text);
    auto t_asr_end = std::chrono::steady_clock::now();

    if (ok && !text.empty()) {
      float asr_ms = std::chrono::duration<float, std::milli>(t_asr_end - t_asr_start).count();
      float rtf = asr_ms / audio_duration_ms;  // 实时率 (RTF)

      // 发布文字
      auto text_msg = std_msgs::msg::String();
      text_msg.data = text;
      text_pub_->publish(text_msg);

      // 发布语音指令（供 VLM 使用）
      auto cmd_msg = bot_interfaces::msg::VLMResult();
      cmd_msg.header.stamp = now();
      cmd_msg.header.frame_id = "speech";
      cmd_msg.query = "语音输入";
      cmd_msg.response = text;
      cmd_msg.confidence = 1.0f;
      command_pub_->publish(cmd_msg);

      RCLCPP_INFO(get_logger(),
        "===== ASR 性能报告 =====\n"
        "  音频时长: %.1f ms\n"
        "  推理耗时: %.1f ms\n"
        "  实时率RTF: %.2f (< 1.0 表示实时)\n"
        "  识别结果: %s",
        audio_duration_ms, asr_ms, rtf, text.c_str());
    }
  }
}

bool SpeechNode::run_asr(const std::vector<float> & audio, std::string & text)
{
  try {
    // ===== 1. 计算 Mel-spectrogram（内存中）=====
    static AudioPreprocessor preprocessor;
    int time_steps = 0;
    std::vector<float> mel_fp32 = preprocessor.extract_whisper_mel(audio, time_steps);

    if (mel_fp32.empty() || time_steps <= 0) {
      RCLCPP_ERROR(get_logger(), "Mel-spectrogram 计算失败");
      return false;
    }
    // ===== 在 mel_fp32 计算完之后，转换 FP16 之前，加这段 =====
    constexpr int MIN_TIME_STEPS = 100;  // TensorRT 引擎要求的最小时间步

    if (time_steps < MIN_TIME_STEPS) {
        size_t mel_bins = mel_fp32.size() / time_steps;  // 应该是 128
        mel_fp32.resize(mel_bins * MIN_TIME_STEPS, 0.0f); // 零填充
        time_steps = MIN_TIME_STEPS;
    }
    // ===== 2. 快速 FP16 转换（批量处理）=====
    size_t mel_size = mel_fp32.size();
    std::vector<uint16_t> mel_fp16(mel_size);

    // 使用批量转换，减少分支判断
    const float* src = mel_fp32.data();
    uint16_t* dst = mel_fp16.data();
    for (size_t i = 0; i < mel_size; i++) {
      float f = src[i];
      // 快速 FP16 转换（IEEE 754）
      uint32_t bits;
      std::memcpy(&bits, &f, sizeof(float));
      uint16_t sign = (bits >> 16) & 0x8000;
      int16_t exponent = ((bits >> 23) & 0xFF) - 127 + 15;
      uint16_t mantissa = (bits >> 13) & 0x3FF;
      // 使用无分支写入
      dst[i] = sign | ((exponent > 0 && exponent < 31) ? ((exponent << 10) | mantissa) : (exponent <= 0 ? 0 : 0x7C00));
    }

    // ===== 3. 内存中构建 safetensors（避免磁盘 I/O）=====
    // 预计算文件大小
    size_t data_size = mel_fp16.size() * sizeof(uint16_t);
    std::string json_header = "{\"mel_spectrogram\": {\"dtype\": \"F16\", \"shape\": [1, 128, " +
      std::to_string(time_steps) + "], \"data_offsets\": [0, " +
      std::to_string(data_size) + "]}}";
    size_t header_size = json_header.size();
    size_t padded_size = ((header_size + 7) / 8) * 8;
    size_t total_size = 8 + padded_size + data_size;

    // 使用静态缓冲区避免重复分配
    static std::vector<uint8_t> st_buffer;
    if (st_buffer.size() < total_size) {
      st_buffer.resize(total_size);
    }

    // 构建 safetensors 文件内容
    uint8_t* ptr = st_buffer.data();
    // 写入头部长度
    uint64_t hlen = padded_size;
    std::memcpy(ptr, &hlen, 8);
    ptr += 8;
    // 写入 JSON 头部（填充到 8 字节对齐）
    std::string padded_header = json_header;
    padded_header.resize(padded_size, ' ');
    std::memcpy(ptr, padded_header.c_str(), padded_size);
    ptr += padded_size;
    // 写入数据
    std::memcpy(ptr, mel_fp16.data(), data_size);

    // 使用内存中的 safetensors 路径（通过临时文件映射）
    // 注意：TensorRT-Edge-LLM 要求文件路径，所以我们仍然需要写入文件
    // 但使用更快的写入方式
    static const std::string st_path = "/tmp/asr_mel_cache.safetensors";
    {
      // 使用 O_DIRECT 和 O_SYNC 优化写入（如果支持）
      FILE* fp = fopen(st_path.c_str(), "wb");
      if (fp) {
        // 一次性写入整个文件
        fwrite(st_buffer.data(), 1, total_size, fp);
        fclose(fp);
      }
    }

    // ===== 4. 构建 ASR 请求（优化配置）=====
    trt_edgellm::rt::LLMGenerationRequest req;
    req.maxGenerateLength = 64;  // 减少最大生成长度（语音识别通常较短）
    req.temperature = 0.0f;      // 确定性输出
    req.topP = 1.0f;
    req.topK = 1;
    req.applyChatTemplate = true;
    req.addGenerationPrompt = true;

    trt_edgellm::rt::LLMGenerationRequest::Request r;

    // 用户消息（只包含音频）
    trt_edgellm::rt::Message msg;
    msg.role = "user";
    trt_edgellm::rt::Message::MessageContent audio_c;
    audio_c.type = "audio";
    audio_c.content = st_path;
    msg.contents.push_back(audio_c);
    r.messages = {msg};

    // 音频数据
    trt_edgellm::rt::audioUtils::AudioData audio_data;
    audio_data.melSpectrogramPath = st_path;
    audio_data.melSpectrogramFormat = "safetensors";
    r.audioBuffers.push_back(std::move(audio_data));
    req.requests = {r};

    // ===== 5. 执行推理（带性能计时）=====
    auto t_start = std::chrono::steady_clock::now();
    trt_edgellm::rt::LLMGenerationResponse res;
    bool ok = runtime_->handleRequest(req, res, stream_);
    auto t_end = std::chrono::steady_clock::now();

    float inference_ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();
    RCLCPP_INFO(get_logger(), "ASR 推理耗时: %.1f ms", inference_ms);

    if (ok && !res.outputTexts.empty()) {
      text = std::move(res.outputTexts[0]);

      // ===== 6. 快速清理输出 =====
      // 移除语言前缀
      size_t pos = text.find("Chinese");
      if (pos != std::string::npos) {
        text = text.substr(pos + 7);
      }

      // 快速去除首尾空白
      size_t start = text.find_first_not_of(" \t\n\r");
      size_t end = text.find_last_not_of(" \t\n\r");
      if (start != std::string::npos && end != std::string::npos) {
        text = text.substr(start, end - start + 1);
      }

      return true;
    }
    return false;
  } catch (const std::exception & e) {
    RCLCPP_ERROR(get_logger(), "ASR 错误: %s", e.what());
    return false;
  }
}

}  // namespace bot_speech

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<bot_speech::SpeechNode>());
  rclcpp::shutdown();
  return 0;
}
