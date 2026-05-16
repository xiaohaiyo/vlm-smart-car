"""VLM 智能小车完整管线启动文件

启动顺序：
1. RealSense D435i 相机
2. VLM 推理节点
3. 底盘控制节点 (VLM → cmd_vel)
4. Codbot 底盘驱动 (D100)

数据流：
Camera → VLM → /vlm/velocity → Chassis → /cmd_vel → Codbot D100
"""
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # ===== 环境变量 =====
        SetEnvironmentVariable('EDGELLM_PLUGIN_PATH',
          '/home/nvidia/VLM_project/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so'),
        SetEnvironmentVariable('LD_LIBRARY_PATH',
          '/home/nvidia/VLM_project/TensorRT-Edge-LLM/build:' + os.environ.get('LD_LIBRARY_PATH', '')),

        # ===== 参数声明 =====
        DeclareLaunchArgument('engine_dir',
          default_value='/home/nvidia/VLM_project/Cosmos_ONNXINT4/engine/llm'),
        DeclareLaunchArgument('visual_dir',
          default_value='/home/nvidia/VLM_project/Cosmos_ONNXINT4/engine/visual'),
        DeclareLaunchArgument('default_prompt',
          default_value='你是自动驾驶小车的视觉大脑。根据摄像头图像，判断前方路况并输出速度指令。'),

        # ===== 1. RealSense D435i 相机 =====
        Node(
            package='realsense2_camera',
            executable='realsense2_camera_node',
            name='camera',
            namespace='camera',
            output='screen',
            parameters=[{
                'camera_name': 'camera',
                'enable_color': True,
                'enable_depth': False,
                'enable_infra1': False,
                'enable_infra2': False,
                'rgb_camera.color_profile': '640,480,30',
                'pointcloud.enable': False,
                'enable_sync': False,
            }],
        ),

        # ===== 2. VLM 推理节点 =====
        Node(
            package='bot_vlm',
            executable='vlm_node',
            name='vlm_node',
            output='screen',
            parameters=[{
                'engine_dir': LaunchConfiguration('engine_dir'),
                'visual_dir': LaunchConfiguration('visual_dir'),
                'max_tokens': 64,
                'temperature': 0.3,
                'default_prompt': LaunchConfiguration('default_prompt'),
            }],
        ),

        # ===== 3. 底盘控制节点 (VLM → cmd_vel) =====
        Node(
            package='bot_chassis',
            executable='chassis_node',
            name='chassis_node',
            output='screen',
            parameters=[{
                'max_linear_vel': 0.05,     # 最大线速度 0.05 m/s（很慢，匹配推理速度）
                'max_angular_vel': 0.3,     # 最大角速度 0.3 rad/s
                'cmd_timeout': 2.0,         # 超时 2 秒
                'publish_rate': 20.0,
                'cmd_vel_topic': '/cmd_vel',
            }],
        ),

        # ===== 4. Codbot 底盘驱动 (D100 差速) =====
        # 使用绝对路径运行 codbot_base（不在 ROS 2 包路径中）
        ExecuteProcess(
            cmd=[
                '/home/nvidia/VLM_project/codbot_base/install/lib/codbot_base/codbot_base',
                '--ros-args',
                '-p', 'cmd_port:=/dev/base',
                '-p', 'baud_rate:=B115200',
                '-p', 'current_type:=D100',
                '-p', 'cmd_vel_topic:=/cmd_vel',
                '-p', 'odom_topic:=/odom',
                '-p', 'imu_topic:=/imu',
                '-p', 'cmd_vel_linear_max:=0.1',
                '-p', 'cmd_vel_linear_min:=0.05',
                '-p', 'cmd_vel_anglar_max:=0.5',
                '-p', 'cmd_vel_anglar_min:=0.0',
                '-p', 'odom_frame:=odom',
                '-p', 'odom_child_frame:=base_footprint',
            ],
            output='screen',
        ),
    ])
