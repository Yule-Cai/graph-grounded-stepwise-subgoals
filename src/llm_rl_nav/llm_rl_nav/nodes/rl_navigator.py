#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker 
from stable_baselines3 import PPO
import numpy as np
import math
import os

from llm_rl_nav.utils import latest_successful_ppo_path, project_root

print(">>> 正在初始化 RL 导航节点，请稍候...")

class RLNavigator(Node):
    def __init__(self):
        super().__init__('rl_navigator')

        self.declare_parameter("model_path", "")
        
        # --- 1. 加载 RL 模型 ---
        configured_model = str(self.get_parameter("model_path").value).strip()
        default_model = latest_successful_ppo_path(project_root())
        model_path = configured_model or os.environ.get("RL_MODEL_PATH", str(default_model))
        if not os.path.exists(model_path):
            self.get_logger().error(f"严重错误：找不到模型文件 '{model_path}'！")
            self.get_logger().error("请设置参数 model_path / 环境变量 RL_MODEL_PATH，或先运行 train_ppo。")
            # 标记为加载失败，防止后面报错
            self.model = None
        else:
            try:
                # cpu 模式加载
                self.model = PPO.load(model_path, device="cpu")
                self.get_logger().info(">>> RL 模型加载成功！大脑已就绪。")
            except Exception as e:
                self.get_logger().error(f"模型加载出错: {e}")
                self.model = None

        # --- 2. 初始化虚拟机器人状态 ---
        self.robot_pos = [0.0, 0.0] # x, y
        self.robot_angle = 0.0      # 朝向
        self.target_pos = None      # 目标点

        # --- 3. ROS 通信接口 ---
        # 订阅来自 LLM 的目标点
        self.subscription = self.create_subscription(
            Float32MultiArray, 'rl_target', self.target_callback, 10)
        
        # 发布可视化 Marker 给 Rviz
        self.marker_pub = self.create_publisher(Marker, 'visualization_marker', 10)
        
        # 启动控制循环 (10Hz = 0.1秒一次)
        self.timer = self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info("RL Navigator 节点已启动，等待指令中...")

    def target_callback(self, msg):
        self.target_pos = msg.data
        self.get_logger().info(f"收到新任务: 前往目标 {self.target_pos}")

    def publish_markers(self):
        """发布 Rviz 可视化标记"""
        timestamp = self.get_clock().now().to_msg()

        # A. 机器人 (蓝色方块)
        robot_marker = Marker()
        robot_marker.header.frame_id = "map"
        robot_marker.header.stamp = timestamp
        robot_marker.ns = "robot"
        robot_marker.id = 0
        robot_marker.type = Marker.CUBE
        robot_marker.action = Marker.ADD
        robot_marker.pose.position.x = self.robot_pos[0]
        robot_marker.pose.position.y = self.robot_pos[1]
        robot_marker.scale.x = 0.5; robot_marker.scale.y = 0.3; robot_marker.scale.z = 0.2
        robot_marker.color.a = 1.0; robot_marker.color.b = 1.0; robot_marker.color.g = 0.0; robot_marker.color.r = 0.0
        self.marker_pub.publish(robot_marker)

        # B. 目标点 (红色球体)
        if self.target_pos:
            target_marker = Marker()
            target_marker.header.frame_id = "map"
            target_marker.header.stamp = timestamp
            target_marker.ns = "target"
            target_marker.id = 1
            target_marker.type = Marker.SPHERE
            target_marker.action = Marker.ADD
            target_marker.pose.position.x = float(self.target_pos[0])
            target_marker.pose.position.y = float(self.target_pos[1])
            target_marker.scale.x = 0.5; target_marker.scale.y = 0.5; target_marker.scale.z = 0.5
            target_marker.color.a = 1.0; target_marker.color.r = 1.0; target_marker.color.g = 0.0; target_marker.color.b = 0.0
            self.marker_pub.publish(target_marker)

    def control_loop(self):
        # 始终刷新可视化，确保 Rviz 不会闪烁
        self.publish_markers()

        # 如果没有模型或者没有目标，就待机
        if self.model is None or self.target_pos is None:
            return

        # --- 物理计算 ---
        dx = self.target_pos[0] - self.robot_pos[0]
        dy = self.target_pos[1] - self.robot_pos[1]
        dist = math.sqrt(dx**2 + dy**2)
        
        # 简单判定到达
        if dist < 0.2:
            if dist > 0.05: # 防止重复刷屏
                self.get_logger().info(f"到达目的地！当前坐标: [{self.robot_pos[0]:.2f}, {self.robot_pos[1]:.2f}]")
            # 到达后不清除目标，保持停留在原地，或者你可以选择 self.target_pos = None
            return

        # 计算相对角度
        target_angle = math.atan2(dy, dx)
        angle_diff = target_angle - self.robot_angle
        # 归一化到 [-pi, pi]
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi

        # --- RL 推理 (核心) ---
        # 构造 observation: [距离, 角度差]
        obs = np.array([dist, angle_diff], dtype=np.float32)
        
        # 获取动作
        action, _ = self.model.predict(obs, deterministic=True)
        v = float(action[0])      # 线速度
        omega = float(action[1])  # 角速度

        # --- 运动学更新 ---
        dt = 0.1 # 时间步长
        self.robot_angle += omega * dt
        self.robot_pos[0] += v * math.cos(self.robot_angle) * dt
        self.robot_pos[1] += v * math.sin(self.robot_angle) * dt
        
        # 偶尔打印一下状态 (每20次循环打印一次，防止刷屏)
        # 你可以把 20 改成 1 来查看实时数据
        current_time = self.get_clock().now().nanoseconds / 1e9
        if int(current_time * 10) % 20 == 0:
            self.get_logger().info(
                f"导航中... Dist: {dist:.2f}m | Action: v={v:.2f}, w={omega:.2f}"
            )

# --- 之前缺失的关键部分 ---
def main(args=None):
    rclpy.init(args=args)
    node = RLNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
