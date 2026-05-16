#!/usr/bin/env python3
"""
VLM 智能小车 GUI 显示节点

功能：
- 蓝牙连接选择
- 实时相机画面
- 语音输入显示
- VLM 推理结果
- 速度控制显示
- 系统状态监控
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from bot_interfaces.msg import VLMResult, VelocityCommand

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import subprocess
import cv2
import numpy as np
from PIL import Image as PILImage, ImageTk
from datetime import datetime
import time


class GUIDisplayNode(Node):
    def __init__(self, root):
        super().__init__('gui_display_node')
        self.root = root
        self.root.title("VLM 智能小车控制系统")
        self.root.geometry("1400x900")

        # 数据
        self.latest_frame = None
        self.latest_vlm_result = ""
        self.latest_speech = ""
        self.linear_vel = 0.0
        self.angular_vel = 0.0
        self.frame_count = 0
        self.vlm_count = 0
        self.speech_count = 0
        self.start_time = time.time()

        # 蓝牙状态
        self.bt_connected = False
        self.bt_device = ""

        # 创建 GUI
        self.create_gui()

        # 订阅 ROS 话题
        self.image_sub = self.create_subscription(
            Image, '/camera/color/image_raw', self.image_callback, 1)
        self.vlm_sub = self.create_subscription(
            VLMResult, '/vlm/result', self.vlm_callback, 10)
        self.vel_sub = self.create_subscription(
            VelocityCommand, '/vlm/velocity', self.vel_callback, 10)
        self.speech_sub = self.create_subscription(
            String, '/speech/text', self.speech_callback, 10)

        # 定时更新 GUI
        self.update_gui()

        self.get_logger().info("GUI 显示节点已初始化")

    def create_gui(self):
        """创建 GUI 界面"""
        # 主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 顶部状态栏
        self.create_status_bar(main_frame)

        # 中间内容区域
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # 左侧：相机画面
        self.create_camera_panel(content_frame)

        # 右侧：信息面板
        self.create_info_panel(content_frame)

        # 底部：控制面板
        self.create_control_panel(main_frame)

    def create_status_bar(self, parent):
        """创建顶部状态栏"""
        status_frame = ttk.LabelFrame(parent, text="系统状态", padding=10)
        status_frame.pack(fill=tk.X, pady=(0, 10))

        # 第一行
        row1 = ttk.Frame(status_frame)
        row1.pack(fill=tk.X)

        # 蓝牙状态
        self.bt_status_label = ttk.Label(row1, text="蓝牙: 未连接", foreground="red")
        self.bt_status_label.pack(side=tk.LEFT, padx=10)

        # 相机状态
        self.camera_status_label = ttk.Label(row1, text="相机: 等待中", foreground="gray")
        self.camera_status_label.pack(side=tk.LEFT, padx=10)

        # VLM 状态
        self.vlm_status_label = ttk.Label(row1, text="VLM: 等待中", foreground="gray")
        self.vlm_status_label.pack(side=tk.LEFT, padx=10)

        # ASR 状态
        self.asr_status_label = ttk.Label(row1, text="ASR: 等待中", foreground="gray")
        self.asr_status_label.pack(side=tk.LEFT, padx=10)

        # FPS
        self.fps_label = ttk.Label(row1, text="FPS: 0.0")
        self.fps_label.pack(side=tk.RIGHT, padx=10)

        # 时间
        self.time_label = ttk.Label(row1, text="00:00:00")
        self.time_label.pack(side=tk.RIGHT, padx=10)

    def create_camera_panel(self, parent):
        """创建相机画面面板"""
        camera_frame = ttk.LabelFrame(parent, text="相机画面", padding=10)
        camera_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 相机画布
        self.camera_canvas = tk.Canvas(camera_frame, bg="black", width=640, height=480)
        self.camera_canvas.pack(fill=tk.BOTH, expand=True)

        # 默认显示
        self.camera_canvas.create_text(320, 240, text="等待相机...",
                                      fill="white", font=("Arial", 20))

    def create_info_panel(self, parent):
        """创建信息面板"""
        info_frame = ttk.Frame(parent)
        info_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))

        # VLM 推理结果
        vlm_frame = ttk.LabelFrame(info_frame, text="VLM 推理结果", padding=10)
        vlm_frame.pack(fill=tk.X, pady=(0, 10))

        self.vlm_text = tk.Text(vlm_frame, height=6, wrap=tk.WORD, font=("Arial", 11))
        self.vlm_text.pack(fill=tk.X)

        # 语音输入
        speech_frame = ttk.LabelFrame(info_frame, text="语音输入", padding=10)
        speech_frame.pack(fill=tk.X, pady=(0, 10))

        self.speech_text = tk.Text(speech_frame, height=3, wrap=tk.WORD, font=("Arial", 11))
        self.speech_text.pack(fill=tk.X)

        # 速度控制
        vel_frame = ttk.LabelFrame(info_frame, text="速度控制", padding=10)
        vel_frame.pack(fill=tk.X, pady=(0, 10))

        # 线速度
        ttk.Label(vel_frame, text="线速度:").grid(row=0, column=0, sticky=tk.W)
        self.linear_vel_bar = ttk.Progressbar(vel_frame, length=200, mode='determinate')
        self.linear_vel_bar.grid(row=0, column=1, padx=10)
        self.linear_vel_label = ttk.Label(vel_frame, text="0.00 m/s")
        self.linear_vel_label.grid(row=0, column=2)

        # 角速度
        ttk.Label(vel_frame, text="角速度:").grid(row=1, column=0, sticky=tk.W)
        self.angular_vel_bar = ttk.Progressbar(vel_frame, length=200, mode='determinate')
        self.angular_vel_bar.grid(row=1, column=1, padx=10)
        self.angular_vel_label = ttk.Label(vel_frame, text="0.00 rad/s")
        self.angular_vel_label.grid(row=1, column=2)

        # 统计信息
        stats_frame = ttk.LabelFrame(info_frame, text="统计信息", padding=10)
        stats_frame.pack(fill=tk.X)

        self.stats_label = ttk.Label(stats_frame, text="帧数: 0 | VLM: 0 | 语音: 0")
        self.stats_label.pack(fill=tk.X)

    def create_control_panel(self, parent):
        """创建底部控制面板"""
        control_frame = ttk.LabelFrame(parent, text="控制", padding=10)
        control_frame.pack(fill=tk.X)

        # 蓝牙连接
        bt_frame = ttk.Frame(control_frame)
        bt_frame.pack(fill=tk.X, pady=5)

        ttk.Label(bt_frame, text="蓝牙设备:").pack(side=tk.LEFT)

        self.bt_device_combo = ttk.Combobox(bt_frame, width=30)
        self.bt_device_combo.pack(side=tk.LEFT, padx=10)

        self.bt_scan_btn = ttk.Button(bt_frame, text="扫描", command=self.scan_bluetooth)
        self.bt_scan_btn.pack(side=tk.LEFT, padx=5)

        self.bt_connect_btn = ttk.Button(bt_frame, text="连接", command=self.connect_bluetooth)
        self.bt_connect_btn.pack(side=tk.LEFT, padx=5)

        self.bt_disconnect_btn = ttk.Button(bt_frame, text="断开", command=self.disconnect_bluetooth)
        self.bt_disconnect_btn.pack(side=tk.LEFT, padx=5)

        # 状态消息
        self.status_msg = ttk.Label(control_frame, text="就绪")
        self.status_msg.pack(fill=tk.X, pady=5)

    def scan_bluetooth(self):
        """扫描蓝牙设备"""
        self.status_msg.config(text="正在扫描蓝牙设备...")
        self.root.update()

        try:
            # 扫描蓝牙设备
            result = subprocess.run(
                ['bluetoothctl', 'scan', 'on'],
                capture_output=True, text=True, timeout=5
            )
            time.sleep(3)

            # 获取设备列表
            result = subprocess.run(
                ['bluetoothctl', 'devices'],
                capture_output=True, text=True, timeout=5
            )

            devices = []
            for line in result.stdout.split('\n'):
                if 'Device' in line:
                    parts = line.split()
                    if len(parts) >= 3:
                        mac = parts[1]
                        name = ' '.join(parts[2:])
                        devices.append(f"{name} ({mac})")

            self.bt_device_combo['values'] = devices
            if devices:
                self.bt_device_combo.set(devices[0])
                self.status_msg.config(text=f"找到 {len(devices)} 个设备")
            else:
                self.status_msg.config(text="未找到蓝牙设备")

        except Exception as e:
            self.status_msg.config(text=f"扫描失败: {str(e)}")

    def connect_bluetooth(self):
        """连接蓝牙设备"""
        device = self.bt_device_combo.get()
        if not device:
            messagebox.showwarning("警告", "请先选择蓝牙设备")
            return

        # 提取 MAC 地址
        mac = device.split('(')[-1].rstrip(')')
        self.status_msg.config(text=f"正在连接 {mac}...")

        try:
            # 连接蓝牙
            subprocess.run(['bluetoothctl', 'pair', mac], capture_output=True, timeout=10)
            subprocess.run(['bluetoothctl', 'connect', mac], capture_output=True, timeout=10)
            subprocess.run(['bluetoothctl', 'trust', mac], capture_output=True, timeout=10)

            self.bt_connected = True
            self.bt_device = mac
            self.bt_status_label.config(text=f"蓝牙: 已连接", foreground="green")
            self.status_msg.config(text=f"已连接到 {device}")

        except Exception as e:
            self.status_msg.config(text=f"连接失败: {str(e)}")

    def disconnect_bluetooth(self):
        """断开蓝牙连接"""
        if self.bt_device:
            try:
                subprocess.run(['bluetoothctl', 'disconnect', self.bt_device],
                             capture_output=True, timeout=10)
            except:
                pass

        self.bt_connected = False
        self.bt_device = ""
        self.bt_status_label.config(text="蓝牙: 未连接", foreground="red")
        self.status_msg.config(text="已断开蓝牙连接")

    def image_callback(self, msg):
        """图像回调"""
        try:
            # 转换 ROS 图像为 OpenCV 格式
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            self.latest_frame = img
            self.frame_count += 1

            # 更新相机状态
            self.camera_status_label.config(text="相机: 运行中", foreground="green")

        except Exception as e:
            self.get_logger().error(f"图像转换失败: {e}")

    def vlm_callback(self, msg):
        """VLM 结果回调"""
        self.latest_vlm_result = msg.response
        self.vlm_count += 1

        # 更新 VLM 状态
        self.vlm_status_label.config(text="VLM: 运行中", foreground="green")

    def vel_callback(self, msg):
        """速度指令回调"""
        self.linear_vel = msg.linear_velocity
        self.angular_vel = msg.angular_velocity

    def speech_callback(self, msg):
        """语音识别回调"""
        self.latest_speech = msg.data
        self.speech_count += 1

        # 更新 ASR 状态
        self.asr_status_label.config(text="ASR: 运行中", foreground="green")

    def update_gui(self):
        """定时更新 GUI"""
        try:
            # 更新相机画面
            if self.latest_frame is not None:
                self.update_camera_display()

            # 更新 VLM 结果
            self.vlm_text.delete(1.0, tk.END)
            self.vlm_text.insert(tk.END, self.latest_vlm_result)

            # 更新语音输入
            self.speech_text.delete(1.0, tk.END)
            self.speech_text.insert(tk.END, self.latest_speech)

            # 更新速度显示
            self.update_velocity_display()

            # 更新统计信息
            self.update_stats()

            # 更新时间
            now = datetime.now().strftime("%H:%M:%S")
            self.time_label.config(text=now)

            # 更新 FPS
            elapsed = time.time() - self.start_time
            fps = self.frame_count / elapsed if elapsed > 0 else 0
            self.fps_label.config(text=f"FPS: {fps:.1f}")

        except Exception as e:
            self.get_logger().error(f"GUI 更新失败: {e}")

        # 100ms 后再次更新
        self.root.after(100, self.update_gui)

    def update_camera_display(self):
        """更新相机画面显示"""
        if self.latest_frame is None:
            return

        # 缩放图像
        frame = cv2.resize(self.latest_frame, (640, 480))

        # 转换为 PIL 格式
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(frame_rgb)

        # 转换为 Tkinter 格式
        tk_image = ImageTk.PhotoImage(pil_image)

        # 更新画布
        self.camera_canvas.delete("all")
        self.camera_canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)
        self.camera_canvas.image = tk_image  # 保持引用

    def update_velocity_display(self):
        """更新速度显示"""
        # 线速度 (-0.1 到 0.1 映射到 0-100)
        linear_percent = int((self.linear_vel + 0.1) / 0.2 * 100)
        linear_percent = max(0, min(100, linear_percent))
        self.linear_vel_bar['value'] = linear_percent
        self.linear_vel_label.config(text=f"{self.linear_vel:.2f} m/s")

        # 角速度 (-0.5 到 0.5 映射到 0-100)
        angular_percent = int((self.angular_vel + 0.5) / 1.0 * 100)
        angular_percent = max(0, min(100, angular_percent))
        self.angular_vel_bar['value'] = angular_percent
        self.angular_vel_label.config(text=f"{self.angular_vel:.2f} rad/s")

    def update_stats(self):
        """更新统计信息"""
        self.stats_label.config(
            text=f"帧数: {self.frame_count} | VLM: {self.vlm_count} | 语音: {self.speech_count}"
        )


def main():
    rclpy.init()

    root = tk.Tk()
    node = GUIDisplayNode(root)

    # 在单独线程中运行 ROS
    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    # 运行 GUI
    root.mainloop()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
