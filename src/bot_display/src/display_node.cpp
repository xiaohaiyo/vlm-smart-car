/**
 * @file display_node.cpp
 * @brief 显示节点实现
 *
 * 功能：
 * - 显示相机画面
 * - 显示 VLM 推理结果和速度指令
 * - 显示语音识别结果
 * - 显示蓝牙连接状态
 * - 显示系统状态
 */

#include "bot_display/display_node.hpp"
#include <cv_bridge/cv_bridge.h>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <cstdlib>

namespace bot_display
{

DisplayNode::DisplayNode(const rclcpp::NodeOptions & options)
: Node("display_node", options)
{
  window_name_ = declare_parameter<std::string>("window_name", "VLM Smart Car");
  display_width_ = declare_parameter<int>("display_width", 1280);
  display_height_ = declare_parameter<int>("display_height", 720);

  // 订阅相机图像
  image_sub_ = create_subscription<sensor_msgs::msg::Image>(
    "/camera/color/image_raw", rclcpp::SensorDataQoS(),
    std::bind(&DisplayNode::image_callback, this, std::placeholders::_1));

  // 订阅 VLM 推理结果
  vlm_sub_ = create_subscription<bot_interfaces::msg::VLMResult>(
    "/vlm/result", 10,
    std::bind(&DisplayNode::vlm_callback, this, std::placeholders::_1));

  // 订阅速度指令
  vel_sub_ = create_subscription<bot_interfaces::msg::VelocityCommand>(
    "/vlm/velocity", 10,
    std::bind(&DisplayNode::velocity_callback, this, std::placeholders::_1));

  // 订阅语音识别结果
  speech_sub_ = create_subscription<std_msgs::msg::String>(
    "/speech/text", 10,
    std::bind(&DisplayNode::speech_callback, this, std::placeholders::_1));

  // 启动显示线程
  running_ = true;
  display_worker_ = std::make_unique<std::thread>(&DisplayNode::display_thread, this);

  RCLCPP_INFO(get_logger(), "显示节点已初始化");
}

DisplayNode::~DisplayNode()
{
  running_ = false;
  if (display_worker_ && display_worker_->joinable()) {
    display_worker_->join();
  }
  cv::destroyAllWindows();
}

void DisplayNode::image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
{
  try {
    auto cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
    std::lock_guard<std::mutex> lock(mutex_);
    latest_frame_ = cv_ptr->image;
    frame_count_++;
  } catch (const cv_bridge::Exception & e) {
    RCLCPP_ERROR(get_logger(), "图像转换失败: %s", e.what());
  }
}

void DisplayNode::vlm_callback(const bot_interfaces::msg::VLMResult::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(mutex_);
  latest_vlm_result_ = msg->response;
  vlm_count_++;
}

void DisplayNode::velocity_callback(const bot_interfaces::msg::VelocityCommand::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(mutex_);
  latest_linear_vel_ = msg->linear_velocity;
  latest_angular_vel_ = msg->angular_velocity;
}

void DisplayNode::speech_callback(const std_msgs::msg::String::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(mutex_);
  latest_speech_text_ = msg->data;
  speech_count_++;
}

std::string DisplayNode::get_system_info()
{
  std::string info;

  // CPU 使用率
  std::ifstream stat("/proc/stat");
  if (stat.is_open()) {
    std::string line;
    std::getline(stat, line);
    info += "CPU: " + line.substr(0, 30) + "...";
    stat.close();
  }

  return info;
}

std::string DisplayNode::get_bluetooth_status()
{
  // 检查蓝牙状态
  FILE* pipe = popen("bluetoothctl show 2>/dev/null | grep Powered", "r");
  if (pipe) {
    char buffer[128];
    std::string result;
    while (fgets(buffer, sizeof(buffer), pipe)) {
      result += buffer;
    }
    pclose(pipe);
    if (result.find("yes") != std::string::npos) {
      return "ON";
    }
  }
  return "OFF";
}

void DisplayNode::display_thread()
{
  RCLCPP_INFO(get_logger(), "显示线程已启动");

  // 创建窗口
  cv::namedWindow(window_name_, cv::WINDOW_AUTOSIZE);

  auto start_time = std::chrono::steady_clock::now();

  while (running_ && rclcpp::ok()) {
    cv::Mat frame;
    std::string vlm_result;
    std::string speech_text;
    float linear_vel = 0.0f;
    float angular_vel = 0.0f;
    int fc = 0, vc = 0, sc = 0;

    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!latest_frame_.empty()) {
        frame = latest_frame_.clone();
      }
      vlm_result = latest_vlm_result_;
      speech_text = latest_speech_text_;
      linear_vel = latest_linear_vel_;
      angular_vel = latest_angular_vel_;
      fc = frame_count_;
      vc = vlm_count_;
      sc = speech_count_;
    }

    if (frame.empty()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(30));
      continue;
    }

    // 缩放到显示尺寸
    cv::Mat display;
    cv::resize(frame, display, cv::Size(display_width_, display_height_));

    // 计算 FPS
    auto now = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(now - start_time).count();
    double fps = fc / elapsed;

    // ===== 绘制信息面板 =====
    int panel_height = 200;
    cv::Mat panel(display_height_ + panel_height, display_width_, CV_8UC3, cv::Scalar(30, 30, 30));

    // 复制图像到面板
    display.copyTo(panel(cv::Rect(0, 0, display_width_, display_height_)));

    // 绘制信息区域背景
    cv::rectangle(panel,
      cv::Point(0, display_height_),
      cv::Point(display_width_, display_height_ + panel_height),
      cv::Scalar(40, 40, 40), -1);

    int y_offset = display_height_ + 20;

    // ===== 左侧信息 =====
    // 1. VLM 推理结果
    cv::putText(panel, "VLM: " + vlm_result.substr(0, 50),
      cv::Point(10, y_offset),
      cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);

    // 2. 语音识别结果
    cv::putText(panel, "ASR: " + speech_text.substr(0, 50),
      cv::Point(10, y_offset + 20),
      cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 1);

    // 3. 速度指令
    std::stringstream vel_ss;
    vel_ss << std::fixed << std::setprecision(2)
           << "Vel: L=" << linear_vel << " A=" << angular_vel;
    cv::putText(panel, vel_ss.str(),
      cv::Point(10, y_offset + 40),
      cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1);

    // 4. 性能信息
    std::stringstream perf_ss;
    perf_ss << "FPS: " << std::fixed << std::setprecision(1) << fps
            << " | F:" << fc << " V:" << vc << " S:" << sc;
    cv::putText(panel, perf_ss.str(),
      cv::Point(10, y_offset + 60),
      cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1);

    // ===== 右侧信息 =====
    int right_x = display_width_ - 250;

    // 5. 蓝牙状态
    std::string bt_status = get_bluetooth_status();
    cv::Scalar bt_color = (bt_status == "ON") ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
    cv::putText(panel, "BT: " + bt_status,
      cv::Point(right_x, y_offset),
      cv::FONT_HERSHEY_SIMPLEX, 0.5, bt_color, 1);

    // 6. 系统状态
    cv::putText(panel, "System: RUNNING",
      cv::Point(right_x, y_offset + 20),
      cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);

    // 7. 时间戳
    auto time_now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(time_now);
    std::stringstream time_ss;
    time_ss << std::put_time(std::localtime(&time_t), "%H:%M:%S");
    cv::putText(panel, time_ss.str(),
      cv::Point(right_x, y_offset + 40),
      cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1);

    // ===== 速度条形图 =====
    int bar_x = display_width_ - 200;
    int bar_y = y_offset + 60;
    int bar_width = 150;
    int bar_height = 12;

    // 线速度条
    cv::rectangle(panel, cv::Point(bar_x, bar_y), cv::Point(bar_x + bar_width, bar_y + bar_height),
      cv::Scalar(60, 60, 60), -1);
    int linear_fill = static_cast<int>((linear_vel / 0.1f) * bar_width);
    linear_fill = std::max(0, std::min(bar_width, linear_fill));
    cv::rectangle(panel, cv::Point(bar_x, bar_y), cv::Point(bar_x + linear_fill, bar_y + bar_height),
      cv::Scalar(0, 200, 0), -1);
    cv::putText(panel, "L", cv::Point(bar_x - 15, bar_y + 10),
      cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(200, 200, 200), 1);

    // 角速度条
    bar_y += 20;
    cv::rectangle(panel, cv::Point(bar_x, bar_y), cv::Point(bar_x + bar_width, bar_y + bar_height),
      cv::Scalar(60, 60, 60), -1);
    int angular_fill = static_cast<int>(((angular_vel + 0.5f) / 1.0f) * bar_width);
    angular_fill = std::max(0, std::min(bar_width, angular_fill));
    cv::rectangle(panel, cv::Point(bar_x, bar_y), cv::Point(bar_x + angular_fill, bar_y + bar_height),
      cv::Scalar(0, 200, 200), -1);
    cv::putText(panel, "A", cv::Point(bar_x - 15, bar_y + 10),
      cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(200, 200, 200), 1);

    // 显示
    cv::imshow(window_name_, panel);

    int key = cv::waitKey(30);
    if (key == 27 || key == 'q') {
      running_ = false;
      rclcpp::shutdown();
    }
  }
}

}  // namespace bot_display

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<bot_display::DisplayNode>());
  rclcpp::shutdown();
  return 0;
}
