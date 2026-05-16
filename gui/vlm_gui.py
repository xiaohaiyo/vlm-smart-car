#!/usr/bin/env python3
"""
VLM 智能小车控制面板

布局：
┌──────────────────────────────────────────────┐
│              机器人VLM控制面板               │
├───────────────┬───────────────────┬──────────┤
│   左侧面板     │     中间主视图     │ 右侧面板 │
│ (输入/日志)    │   (图像+bbox)      │(任务JSON)│
├───────────────┴───────────────────┴──────────┤
│               底部状态栏（控制/速度/模式）    │
└──────────────────────────────────────────────┘

功能：
- 实时相机画面显示（中间主视图）
- 语音输入和日志显示（左侧面板）
- VLM 推理结果 JSON 显示（右侧面板）
- 速度控制和系统状态（底部状态栏）
- 蓝牙设备管理
"""

import sys
import subprocess
import time
import json
import numpy as np
import cv2

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QGroupBox, QProgressBar,
    QTextEdit, QFrame, QSplitter, QStatusBar, QLineEdit, QGridLayout
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPainter

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from bot_interfaces.msg import VLMResult, VelocityCommand

from cv_bridge import CvBridge


class RosThread(QThread):
    """ROS 话题订阅线程"""
    # 信号定义
    image_signal = pyqtSignal(object)      # 相机图像信号
    vlm_signal = pyqtSignal(str)           # VLM 推理结果信号
    vel_signal = pyqtSignal(float, float)  # 速度控制信号
    speech_signal = pyqtSignal(str)        # 语音输入信号

    def __init__(self):
        super().__init__()
        self.bridge = CvBridge()
        self.node = None

    def run(self):
        """运行 ROS 节点，订阅话题"""
        rclpy.init()
        self.node = Node('qt_gui_node')

        # 订阅相机图像
        self.node.create_subscription(
            Image, '/camera/color/image_raw', self.image_callback, 1)
        # 订阅 VLM 推理结果
        self.node.create_subscription(
            VLMResult, '/vlm/result', self.vlm_callback, 10)
        # 订阅速度控制指令
        self.node.create_subscription(
            VelocityCommand, '/vlm/velocity', self.vel_callback, 10)
        # 订阅语音识别结果
        self.node.create_subscription(
            String, '/speech/text', self.speech_callback, 10)

        rclpy.spin(self.node)

    def image_callback(self, msg):
        """相机图像回调"""
        try:
            img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.image_signal.emit(img)
        except Exception:
            pass

    def vlm_callback(self, msg):
        """VLM 推理结果回调"""
        self.vlm_signal.emit(msg.response)

    def vel_callback(self, msg):
        """速度控制指令回调"""
        self.vel_signal.emit(msg.linear_velocity, msg.angular_velocity)

    def speech_callback(self, msg):
        """语音识别结果回调"""
        self.speech_signal.emit(msg.data)


class VLMGui(QMainWindow):
    """VLM 智能小车控制面板主窗口"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("机器人VLM控制面板")
        self.setGeometry(100, 100, 1400, 900)

        # 统计数据
        self.frame_count = 0
        self.vlm_count = 0
        self.speech_count = 0
        self.start_time = time.time()
        self.linear_vel = 0.0
        self.angular_vel = 0.0
        self.current_mode = "待机"

        # 创建 GUI
        self.create_gui()

        # 启动 ROS 线程
        self.ros_thread = RosThread()
        self.ros_thread.image_signal.connect(self.update_image)
        self.ros_thread.vlm_signal.connect(self.update_vlm)
        self.ros_thread.vel_signal.connect(self.update_velocity)
        self.ros_thread.speech_signal.connect(self.update_speech)
        self.ros_thread.start()

        # 定时更新统计信息
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_stats)
        self.timer.start(100)

    def create_gui(self):
        """创建 GUI 布局"""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # ===== 顶部标题栏 =====
        title_frame = QGroupBox("机器人VLM控制面板")
        title_layout = QHBoxLayout()
        title_label = QLabel("VLM 智能小车控制系统")
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_layout.addWidget(title_label)
        title_frame.setLayout(title_layout)
        main_layout.addWidget(title_frame)

        # ===== 中间三栏布局 =====
        content_layout = QHBoxLayout()
        main_layout.addLayout(content_layout)

        # 左侧面板：输入/日志
        self.create_left_panel(content_layout)

        # 中间主视图：图像+bbox
        self.create_center_panel(content_layout)

        # 右侧面板：任务JSON
        self.create_right_panel(content_layout)

        # ===== 底部状态栏 =====
        self.create_bottom_status_bar(main_layout)

    def create_left_panel(self, layout):
        """创建左侧面板：输入/日志"""
        left_frame = QGroupBox("输入/日志")
        left_layout = QVBoxLayout()

        # 语音输入显示
        speech_label = QLabel("语音输入：")
        speech_label.setFont(QFont("Arial", 10, QFont.Bold))
        left_layout.addWidget(speech_label)

        self.speech_text = QTextEdit()
        self.speech_text.setMaximumHeight(100)
        self.speech_text.setPlaceholderText("等待语音输入...")
        self.speech_text.setFont(QFont("Arial", 11))
        left_layout.addWidget(self.speech_text)

        # 系统日志
        log_label = QLabel("系统日志：")
        log_label.setFont(QFont("Arial", 10, QFont.Bold))
        left_layout.addWidget(log_label)

        self.log_text = QTextEdit()
        self.log_text.setPlaceholderText("系统日志将在此显示...")
        self.log_text.setFont(QFont("Consolas", 9))
        left_layout.addWidget(self.log_text)

        # 蓝牙控制
        bt_frame = QGroupBox("蓝牙控制")
        bt_layout = QGridLayout()

        bt_layout.addWidget(QLabel("设备:"), 0, 0)
        self.bt_combo = QComboBox()
        self.bt_combo.setMinimumWidth(150)
        bt_layout.addWidget(self.bt_combo, 0, 1, 1, 2)

        self.scan_btn = QPushButton("扫描")
        self.scan_btn.clicked.connect(self.scan_bluetooth)
        bt_layout.addWidget(self.scan_btn, 1, 0)

        self.connect_btn = QPushButton("连接")
        self.connect_btn.clicked.connect(self.connect_bluetooth)
        bt_layout.addWidget(self.connect_btn, 1, 1)

        self.disconnect_btn = QPushButton("断开")
        self.disconnect_btn.clicked.connect(self.disconnect_bluetooth)
        bt_layout.addWidget(self.disconnect_btn, 1, 2)

        bt_frame.setLayout(bt_layout)
        left_layout.addWidget(bt_frame)

        left_frame.setLayout(left_layout)
        layout.addWidget(left_frame, stretch=1)

    def create_center_panel(self, layout):
        """创建中间主视图：图像+bbox"""
        center_frame = QGroupBox("相机画面")
        center_layout = QVBoxLayout()

        # 图像显示区域
        self.camera_label = QLabel()
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setStyleSheet("background-color: black;")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setText("等待相机...")
        self.camera_label.setFont(QFont("Arial", 20))
        center_layout.addWidget(self.camera_label)

        # 图像信息
        info_layout = QHBoxLayout()
        self.fps_label = QLabel("FPS: 0.0")
        self.resolution_label = QLabel("分辨率: --")
        info_layout.addWidget(self.fps_label)
        info_layout.addWidget(self.resolution_label)
        center_layout.addLayout(info_layout)

        center_frame.setLayout(center_layout)
        layout.addWidget(center_frame, stretch=2)

    def create_right_panel(self, layout):
        """创建右侧面板：任务JSON"""
        right_frame = QGroupBox("任务JSON")
        right_layout = QVBoxLayout()

        # VLM 推理结果 JSON 显示
        json_label = QLabel("VLM 推理结果：")
        json_label.setFont(QFont("Arial", 10, QFont.Bold))
        right_layout.addWidget(json_label)

        self.json_text = QTextEdit()
        self.json_text.setPlaceholderText("等待 VLM 推理结果...")
        self.json_text.setFont(QFont("Consolas", 10))
        right_layout.addWidget(self.json_text)

        # 解析后的关键信息
        info_frame = QGroupBox("关键信息")
        info_layout = QGridLayout()

        info_layout.addWidget(QLabel("用户指令:"), 0, 0)
        self.cmd_label = QLabel("--")
        self.cmd_label.setWordWrap(True)
        info_layout.addWidget(self.cmd_label, 0, 1)

        info_layout.addWidget(QLabel("线速度:"), 1, 0)
        self.linear_label = QLabel("0.00 m/s")
        info_layout.addWidget(self.linear_label, 1, 1)

        info_layout.addWidget(QLabel("角速度:"), 2, 0)
        self.angular_label = QLabel("0.00 rad/s")
        info_layout.addWidget(self.angular_label, 2, 1)

        info_layout.addWidget(QLabel("回复内容:"), 3, 0)
        self.reply_label = QLabel("--")
        self.reply_label.setWordWrap(True)
        info_layout.addWidget(self.reply_label, 3, 1)

        info_frame.setLayout(info_layout)
        right_layout.addWidget(info_frame)

        right_frame.setLayout(right_layout)
        layout.addWidget(right_frame, stretch=1)

    def create_bottom_status_bar(self, layout):
        """创建底部状态栏：控制/速度/模式"""
        status_frame = QGroupBox("状态栏")
        status_layout = QHBoxLayout()

        # 系统状态
        status_layout.addWidget(QLabel("系统:"))
        self.system_status = QLabel("运行中")
        self.system_status.setStyleSheet("color: green; font-weight: bold;")
        status_layout.addWidget(self.system_status)

        status_layout.addWidget(QLabel("|"))

        # 速度显示
        status_layout.addWidget(QLabel("线速度:"))
        self.linear_bar = QProgressBar()
        self.linear_bar.setRange(0, 100)
        self.linear_bar.setMaximumWidth(150)
        status_layout.addWidget(self.linear_bar)
        self.linear_vel_label = QLabel("0.00 m/s")
        status_layout.addWidget(self.linear_vel_label)

        status_layout.addWidget(QLabel("角速度:"))
        self.angular_bar = QProgressBar()
        self.angular_bar.setRange(0, 100)
        self.angular_bar.setMaximumWidth(150)
        status_layout.addWidget(self.angular_bar)
        self.angular_vel_label = QLabel("0.00 rad/s")
        status_layout.addWidget(self.angular_vel_label)

        status_layout.addWidget(QLabel("|"))

        # 运行模式
        status_layout.addWidget(QLabel("模式:"))
        self.mode_label = QLabel("待机")
        self.mode_label.setStyleSheet("color: blue; font-weight: bold;")
        status_layout.addWidget(self.mode_label)

        status_layout.addWidget(QLabel("|"))

        # 统计信息
        status_layout.addWidget(QLabel("统计:"))
        self.stats_label = QLabel("帧:0 | VLM:0 | 语音:0")
        status_layout.addWidget(self.stats_label)

        status_layout.addWidget(QLabel("|"))

        # 时间
        self.time_label = QLabel("00:00:00")
        status_layout.addWidget(self.time_label)

        status_frame.setLayout(status_layout)
        layout.addWidget(status_frame)

    def scan_bluetooth(self):
        """扫描蓝牙设备"""
        self.log_text.append("正在扫描蓝牙设备...")
        QApplication.processEvents()

        try:
            # 确保蓝牙开启
            subprocess.run(['bluetoothctl', 'power', 'on'],
                         capture_output=True, timeout=5)
            time.sleep(0.5)

            # 开启可发现
            subprocess.run(['bluetoothctl', 'discoverable', 'on'],
                         capture_output=True, timeout=5)
            time.sleep(0.5)

            # 开始扫描
            subprocess.Popen(['bluetoothctl', 'scan', 'on'],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
            time.sleep(5)

            # 停止扫描
            subprocess.run(['bluetoothctl', 'scan', 'off'],
                         capture_output=True, timeout=5)

            # 获取设备列表
            result = subprocess.run(['bluetoothctl', 'devices'],
                                  capture_output=True, text=True, timeout=10)

            devices = []
            for line in result.stdout.split('\n'):
                if 'Device' in line:
                    parts = line.split()
                    if len(parts) >= 3:
                        mac = parts[1]
                        name = ' '.join(parts[2:])
                        devices.append(f"{name} ({mac})")

            self.bt_combo.clear()
            self.bt_combo.addItems(devices)

            if devices:
                self.log_text.append(f"找到 {len(devices)} 个蓝牙设备")
            else:
                self.log_text.append("未找到蓝牙设备")

        except FileNotFoundError:
            self.log_text.append("错误: bluetoothctl 未安装")
        except Exception as e:
            self.log_text.append(f"扫描失败: {str(e)}")

    def connect_bluetooth(self):
        """连接蓝牙设备"""
        device = self.bt_combo.currentText()
        if not device:
            self.log_text.append("请先扫描并选择蓝牙设备")
            return

        mac = device.split('(')[-1].rstrip(')')
        self.log_text.append(f"正在连接 {mac}...")
        QApplication.processEvents()

        try:
            # 配对
            result = subprocess.run(['bluetoothctl', 'pair', mac],
                                  capture_output=True, text=True, timeout=15)
            if 'Failed' in result.stdout or 'Failed' in result.stderr:
                self.log_text.append(f"配对失败: {result.stdout}")
                return

            # 连接
            result = subprocess.run(['bluetoothctl', 'connect', mac],
                                  capture_output=True, text=True, timeout=15)
            if 'Failed' in result.stdout or 'Failed' in result.stderr:
                self.log_text.append(f"连接失败: {result.stdout}")
                return

            # 信任
            subprocess.run(['bluetoothctl', 'trust', mac],
                         capture_output=True, timeout=10)

            self.log_text.append(f"已连接到 {device}")

        except subprocess.TimeoutExpired:
            self.log_text.append("连接超时，请重试")
        except Exception as e:
            self.log_text.append(f"连接失败: {str(e)}")

    def disconnect_bluetooth(self):
        """断开蓝牙连接"""
        self.log_text.append("已断开蓝牙连接")

    def update_image(self, img):
        """更新相机画面"""
        self.frame_count += 1

        # 缩放图像
        h, w = img.shape[:2]
        scale = min(640 / w, 480 / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img, (new_w, new_h))

        # 转换为 QPixmap
        rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, new_w, new_h, 3 * new_w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        self.camera_label.setPixmap(pixmap)
        self.resolution_label.setText(f"分辨率: {w}x{h}")

    def update_vlm(self, text):
        """更新 VLM 推理结果"""
        self.vlm_count += 1

        # 显示原始 JSON
        self.json_text.setText(text)

        # 尝试解析 JSON
        try:
            data = json.loads(text)
            self.cmd_label.setText(data.get("用户指令", "--"))
            self.linear_label.setText(f"{data.get('linear_vel', 0.0):.3f} m/s")
            self.angular_label.setText(f"{data.get('angular_vel', 0.0):.3f} rad/s")
            self.reply_label.setText(data.get("回复内容", "--") or "--")

            # 更新模式
            if data.get("linear_vel", 0.0) != 0.0 or data.get("angular_vel", 0.0) != 0.0:
                self.current_mode = "运动"
                self.mode_label.setStyleSheet("color: green; font-weight: bold;")
            else:
                self.current_mode = "对话"
                self.mode_label.setStyleSheet("color: blue; font-weight: bold;")
            self.mode_label.setText(self.current_mode)

        except json.JSONDecodeError:
            # 非 JSON 格式，直接显示
            self.cmd_label.setText("--")
            self.reply_label.setText(text[:100] + "..." if len(text) > 100 else text)

        # 添加到日志
        self.log_text.append(f"[VLM] {text[:80]}...")

    def update_velocity(self, linear, angular):
        """更新速度显示"""
        self.linear_vel = linear
        self.angular_vel = angular

        # 更新进度条
        linear_pct = int((linear + 0.1) / 0.2 * 100)
        linear_pct = max(0, min(100, linear_pct))
        self.linear_bar.setValue(linear_pct)
        self.linear_vel_label.setText(f"{linear:.3f} m/s")

        angular_pct = int((angular + 0.5) / 1.0 * 100)
        angular_pct = max(0, min(100, angular_pct))
        self.angular_bar.setValue(angular_pct)
        self.angular_vel_label.setText(f"{angular:.3f} rad/s")

    def update_speech(self, text):
        """更新语音输入"""
        self.speech_count += 1
        self.speech_text.setText(text)
        self.log_text.append(f"[语音] {text}")

    def update_stats(self):
        """更新统计信息"""
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0

        self.fps_label.setText(f"FPS: {fps:.1f}")
        self.stats_label.setText(
            f"帧:{self.frame_count} | VLM:{self.vlm_count} | 语音:{self.speech_count}")

        from datetime import datetime
        self.time_label.setText(datetime.now().strftime("%H:%M:%S"))


def main():
    """主函数"""
    app = QApplication(sys.argv)
    window = VLMGui()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
