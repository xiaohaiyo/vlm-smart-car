"""语音识别节点启动文件"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'engine_dir',
            default_value='/home/nvidia/VLM_project/Qwen3-ASR-0.6B_onnx/engine/llm',
            description='ASR LLM 引擎目录'
        ),
        DeclareLaunchArgument(
            'audio_dir',
            default_value='/home/nvidia/VLM_project/Qwen3-ASR-0.6B_onnx/engine/audio',
            description='ASR 音频编码器目录'
        ),
        DeclareLaunchArgument(
            'device_name',
            default_value='default',
            description='音频设备名称'
        ),
        DeclareLaunchArgument(
            'vad_threshold',
            default_value='0.01',
            description='VAD 阈值'
        ),

        Node(
            package='bot_speech',
            executable='speech_node',
            name='speech_node',
            output='screen',
            parameters=[{
                'engine_dir': LaunchConfiguration('engine_dir'),
                'audio_dir': LaunchConfiguration('audio_dir'),
                'device_name': LaunchConfiguration('device_name'),
                'sample_rate': 16000,
                'chunk_duration_ms': 1000,
                'vad_threshold': LaunchConfiguration('vad_threshold'),
            }],
        ),
    ])
