"""VLM 推理节点启动文件（宿主机运行）"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('engine_dir',
          default_value='/home/nvidia/VLM_project/Cosmos_ONNXINT4/engine/llm'),
        DeclareLaunchArgument('visual_dir',
          default_value='/home/nvidia/VLM_project/Cosmos_ONNXINT4/engine/visual'),
        DeclareLaunchArgument('default_prompt',
          default_value=''),
        Node(
            package='bot_vlm',
            executable='vlm_node',
            name='vlm_node',
            output='screen',
            parameters=[{
                'engine_dir': LaunchConfiguration('engine_dir'),
                'visual_dir': LaunchConfiguration('visual_dir'),
                'max_tokens': 128,
                'temperature': 0.3,
                'default_prompt': LaunchConfiguration('default_prompt'),
            }],
        ),
    ])
