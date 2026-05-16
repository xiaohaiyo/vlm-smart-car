"""显示节点启动文件（宿主机运行）"""
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='bot_display',
            executable='display_node',
            name='display_node',
            output='screen',
            parameters=[{
                'window_name': 'VLM 智能小车',
                'display_width': 1280,
                'display_height': 720,
            }],
        ),
    ])
