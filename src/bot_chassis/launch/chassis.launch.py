"""底盘控制节点启动文件"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('max_linear_vel', default_value='0.3'),
        DeclareLaunchArgument('max_angular_vel', default_value='2.0'),
        DeclareLaunchArgument('cmd_timeout', default_value='0.5'),

        Node(
            package='bot_chassis',
            executable='chassis_node',
            name='chassis_node',
            output='screen',
            parameters=[{
                'max_linear_vel': LaunchConfiguration('max_linear_vel'),
                'max_angular_vel': LaunchConfiguration('max_angular_vel'),
                'cmd_timeout': LaunchConfiguration('cmd_timeout'),
                'publish_rate': 20.0,
                'cmd_vel_topic': '/cmd_vel',
            }],
        ),
    ])
