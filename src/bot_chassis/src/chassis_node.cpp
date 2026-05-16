/**
 * @file chassis_node.cpp
 * @brief 底盘控制节点实现
 *
 * 功能：
 * - 订阅 /vlm/velocity (VelocityCommand)
 * - 发布 /cmd_vel (Twist) 到 codbot_base
 * - D100 差速底盘控制
 */

#include "bot_chassis/chassis_node.hpp"
#include <algorithm>

namespace bot_chassis
{

ChassisNode::ChassisNode(const rclcpp::NodeOptions & options)
: Node("chassis_node", options)
{
  // 声明参数
  max_linear_vel_ = declare_parameter<double>("max_linear_vel", 0.3);
  max_angular_vel_ = declare_parameter<double>("max_angular_vel", 2.0);
  cmd_timeout_ = declare_parameter<double>("cmd_timeout", 20);
  publish_rate_ = declare_parameter<double>("publish_rate", 20.0);
  cmd_vel_topic_ = declare_parameter<std::string>("cmd_vel_topic", "/cmd_vel");

  // 创建发布者
  cmd_vel_pub_ = create_publisher<geometry_msgs::msg::Twist>(cmd_vel_topic_, 10);

  // 创建订阅者
  vel_sub_ = create_subscription<bot_interfaces::msg::VelocityCommand>(
    "vlm/velocity", 10,
    std::bind(&ChassisNode::velocity_callback, this, std::placeholders::_1));

  // 创建定时器
  timer_ = create_wall_timer(
    std::chrono::milliseconds(static_cast<int>(1000.0 / publish_rate_)),
    std::bind(&ChassisNode::timer_callback, this));

  last_cmd_time_ = now();

  RCLCPP_INFO(get_logger(), "底盘控制节点已初始化");
  RCLCPP_INFO(get_logger(), "  订阅: vlm/velocity");
  RCLCPP_INFO(get_logger(), "  发布: %s", cmd_vel_topic_.c_str());
  RCLCPP_INFO(get_logger(), "  最大线速度: %.2f m/s", max_linear_vel_);
  RCLCPP_INFO(get_logger(), "  最大角速度: %.2f rad/s", max_angular_vel_);
  RCLCPP_INFO(get_logger(), "  底盘类型: D100 (差速)");
}

void ChassisNode::velocity_callback(
  const bot_interfaces::msg::VelocityCommand::SharedPtr msg)
{
  // 限制速度范围
  double linear_vel = std::clamp(
    static_cast<double>(msg->linear_velocity),
    -max_linear_vel_, max_linear_vel_);

  double angular_vel = std::clamp(
    static_cast<double>(msg->angular_velocity),
    -max_angular_vel_, max_angular_vel_);

  // 更新速度指令
  current_cmd_.linear.x = linear_vel;
  current_cmd_.linear.y = 0.0;
  current_cmd_.linear.z = 0.0;
  current_cmd_.angular.x = 0.0;
  current_cmd_.angular.y = 0.0;
  current_cmd_.angular.z = angular_vel;

  last_cmd_time_ = now();

  RCLCPP_DEBUG(get_logger(), "速度指令: linear=%.2f, angular=%.2f",
    linear_vel, angular_vel);
}

void ChassisNode::timer_callback()
{
  // 检查是否超时
  double elapsed = (now() - last_cmd_time_).seconds();

  if (elapsed > cmd_timeout_) {
    // 超时，停止小车
    current_cmd_ = geometry_msgs::msg::Twist();
    RCLCPP_WARN(get_logger(), "速度指令超时 (%.1f秒)，停车", elapsed);
  }

  // 发布速度指令
  cmd_vel_pub_->publish(current_cmd_);
}

}  // namespace bot_chassis

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<bot_chassis::ChassisNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
