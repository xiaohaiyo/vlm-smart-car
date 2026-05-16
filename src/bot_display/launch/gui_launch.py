"""GUI 显示节点启动文件"""
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='bot_display',
            executable='gui_node.py',
            name='gui_display',
            output='screen',
        ),
    ])
