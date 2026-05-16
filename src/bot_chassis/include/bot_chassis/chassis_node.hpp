/**
 * @file chassis_node.hpp
 * @brief 底盘控制节点头文件
 *
 * 功能：
 * - 订阅 VLM 速度指令
 * - 转换为 cmd_vel 发送给 codbot_base
 * - D100 差速底盘
 */

#ifndef BOT_CHASSIS__CHASSIS_NODE_HPP_
#define BOT_CHASSIS__CHASSIS_NODE_HPP_

#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "bot_interfaces/msg/velocity_command.hpp"

namespace bot_chassis
{

/**
 * @class ChassisNode
 * @brief 底盘控制节点
 *
 * 将 VLM 速度指令转换为 cmd_vel，控制 D100 差速底盘
 */
class ChassisNode : public rclcpp::Node
{
public:
  explicit ChassisNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  virtual ~ChassisNode() = default;

private:
  /**
   * @brief VLM 速度指令回调
   * @param msg VLM 速度指令消息
   */
  void velocity_callback(const bot_interfaces::msg::VelocityCommand::SharedPtr msg);

  /**
   * @brief 定时发送速度指令
   *
   * 以固定频率发送 cmd_vel，如果超时则停止
   */
  void timer_callback();

  // ROS 接口
  rclcpp::Subscription<bot_interfaces::msg::VelocityCommand>::SharedPtr vel_sub_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;
  rclcpp::TimerBase::SharedPtr timer_;

  // 速度指令
  geometry_msgs::msg::Twist current_cmd_;
  rclcpp::Time last_cmd_time_;

  // 参数
  double max_linear_vel_;    ///< 最大线速度 (m/s)
  double max_angular_vel_;   ///< 最大角速度 (rad/s)
  double cmd_timeout_;       ///< 指令超时时间 (秒)
  double publish_rate_;      ///< 发布频率 (Hz)
  std::string cmd_vel_topic_; ///< cmd_vel 话题名
};

}  // namespace bot_chassis

#endif  // BOT_CHASSIS__CHASSIS_NODE_HPP_
