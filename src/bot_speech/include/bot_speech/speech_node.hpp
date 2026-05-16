/**
 * @file speech_node.hpp
 * @brief 语音识别节点头文件
 *
 * 功能：
 * - 从蓝牙耳机/麦克风采集音频
 * - 使用 Qwen3-ASR TensorRT 引擎进行语音识别
 * - 发布识别结果到 ROS 话题
 * - 提供语音指令给 VLM 节点
 */

#ifndef BOT_SPEECH__SPEECH_NODE_HPP_
#define BOT_SPEECH__SPEECH_NODE_HPP_

#include <memory>
#include <string>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "bot_interfaces/msg/vlm_result.hpp"

#include "runtime/llmInferenceSpecDecodeRuntime.h"
#include "runtime/imageUtils.h"

#include <cuda_runtime.h>
#include <alsa/asoundlib.h>

namespace bot_speech
{

// 前向声明 AudioPreprocessor
class AudioPreprocessor;

/**
 * @struct AudioChunk
 * @brief 音频数据块
 */
struct AudioChunk
{
  std::vector<float> data;  ///< 音频数据（float32）
  int sample_rate;          ///< 采样率
};

/**
 * @class SpeechNode
 * @brief 语音识别节点
 *
 * 从蓝牙耳机采集音频，使用 ASR 引擎识别语音，
 * 将识别结果发布到 ROS 话题供 VLM 节点使用。
 */
class SpeechNode : public rclcpp::Node
{
public:
  explicit SpeechNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  virtual ~SpeechNode();

private:
  void initialize_engine();
  void audio_capture_thread();
  void asr_worker_thread();
  bool run_asr(const std::vector<float> & audio, std::string & text);

  // ROS 接口
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr text_pub_;
  rclcpp::Publisher<bot_interfaces::msg::VLMResult>::SharedPtr command_pub_;

  // 音频采集
  snd_pcm_t* pcm_handle_{nullptr};
  std::unique_ptr<std::thread> capture_thread_;
  std::unique_ptr<std::thread> asr_thread_;
  std::atomic<bool> running_{false};

  // VAD 状态机
  enum class VadState { SILENCE, SPEAKING };
  VadState vad_state_{VadState::SILENCE};
  std::vector<float> speech_buffer_;      // 累积的语音数据
  int silence_frames_{0};                 // 连续静音帧数
  static constexpr int SILENCE_THRESHOLD = 3;  // 

  // 音频队列（完整语音段）
  std::queue<AudioChunk> audio_queue_;
  std::mutex queue_mutex_;
  std::condition_variable queue_cv_;

  // ASR 引擎
  std::unique_ptr<trt_edgellm::rt::LLMInferenceSpecDecodeRuntime> runtime_;
  cudaStream_t stream_;
  bool engine_ok_{false};

  // 参数
  std::string engine_dir_;
  std::string audio_dir_;
  std::string device_name_;
  unsigned int sample_rate_;
  int chunk_duration_ms_;
  float vad_threshold_;
};

}  // namespace bot_speech

#endif  // BOT_SPEECH__SPEECH_NODE_HPP_
