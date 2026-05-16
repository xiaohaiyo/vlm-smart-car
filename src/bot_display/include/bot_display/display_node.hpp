/**
 * @file display_node.hpp
 * @brief 显示节点头文件
 *
 * 功能：
 * - 显示相机画面
 * - 显示 VLM 推理结果
 * - 显示速度指令
 * - 显示系统状态
 * - 显示蓝牙连接状态
 * - 显示 ASR/TTS 状态
 */

#ifndef BOT_DISPLAY__DISPLAY_NODE_HPP_
#define BOT_DISPLAY__DISPLAY_NODE_HPP_

#include <memory>
#include <string>
#include <thread>
#include <atomic>
#include <mutex>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/string.hpp"
#include "bot_interfaces/msg/vlm_result.hpp"
#include "bot_interfaces/msg/velocity_command.hpp"

#include <opencv2/opencv.hpp>

namespace bot_display
{

class DisplayNode : public rclcpp::Node
{
public:
  explicit DisplayNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  virtual ~DisplayNode();

private:
  void image_callback(const sensor_msgs::msg::Image::SharedPtr msg);
  void vlm_callback(const bot_interfaces::msg::VLMResult::SharedPtr msg);
  void velocity_callback(const bot_interfaces::msg::VelocityCommand::SharedPtr msg);
  void speech_callback(const std_msgs::msg::String::SharedPtr msg);
  void display_thread();
  std::string get_system_info();
  std::string get_bluetooth_status();

  // ROS 订阅者
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Subscription<bot_interfaces::msg::VLMResult>::SharedPtr vlm_sub_;
  rclcpp::Subscription<bot_interfaces::msg::VelocityCommand>::SharedPtr vel_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr speech_sub_;

  // 数据
  cv::Mat latest_frame_;
  std::string latest_vlm_result_;
  std::string latest_speech_text_;
  float latest_linear_vel_{0.0f};
  float latest_angular_vel_{0.0f};
  int frame_count_{0};
  int vlm_count_{0};
  int speech_count_{0};
  std::mutex mutex_;

  // 显示线程
  std::unique_ptr<std::thread> display_worker_;
  std::atomic<bool> running_{false};

  // 参数
  std::string window_name_;
  int display_width_;
  int display_height_;
};

}  // namespace bot_display

#endif  // BOT_DISPLAY__DISPLAY_NODE_HPP_
