import gym
import rclpy
from rclpy.node import Node
from gym import spaces
import numpy as np
import math
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty

class TurtleBotEnv(gym.Env, Node):
    def __init__(self):
        Node.__init__(self, 'turtlebot_env_node')
        
        # --- 1. 动作空间 ---
        self.action_space = spaces.Box(
            low=np.array([0.0, -2.0]), 
            high=np.array([0.22, 2.0]), 
            dtype=np.float32
        )
        
        # --- 2. 状态空间 (62维) ---
        self.observation_space = spaces.Box(
            low=-float('inf'), 
            high=float('inf'), 
            shape=(62,), 
            dtype=np.float32
        )

        # --- 3. ROS2 接口 ---
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)
        self.sub_scan = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.sub_odom = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.reset_client = self.create_client(Empty, '/reset_simulation')

        self.latest_scan = np.full(360, 3.5, dtype=np.float32)
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.goal_x = 0.0
        self.goal_y = 0.0
        self.current_dist = 0.0

        print("🚀 TurtleBot环境 (Anti-Rubbing Repulsive Mode)")

    def scan_callback(self, msg):
        scan = np.array(msg.ranges)
        scan[scan == 0.0] = 3.5
        scan = np.nan_to_num(scan, posinf=3.5, nan=3.5)
        self.latest_scan = scan

    def odom_callback(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.robot_yaw = math.atan2(siny_cosp, cosy_cosp)

    def step(self, action):
        # --- 1. 安全屏蔽层 (稍微放宽一点，让奖励函数去教它) ---
        min_dist = np.min(self.latest_scan)
        override_action = False
        safe_action = [0.0, 0.0]
        
        # 0.16m 才会强制接管 (给它更多自己操作的空间去感受"电网")
        if min_dist < 0.16:
            override_action = True
            safe_action = [-0.05, 0.0] 
            
        # --- 2. 动作执行 ---
        vel_cmd = Twist()
        if override_action:
            vel_cmd.linear.x = float(safe_action[0])
            vel_cmd.angular.z = float(safe_action[1])
        else:
            v = np.clip(float(action[0]), 0.0, 0.22)
            w = float(action[1])
            vel_cmd.linear.x = v
            vel_cmd.angular.z = w
        
        self.pub_cmd_vel.publish(vel_cmd)
        rclpy.spin_once(self, timeout_sec=0.05)

        # 3. 计算状态
        dist = math.sqrt((self.goal_x - self.robot_x)**2 + (self.goal_y - self.robot_y)**2)
        target_angle = math.atan2(self.goal_y - self.robot_y, self.goal_x - self.robot_x)
        heading_error = target_angle - self.robot_yaw
        while heading_error > np.pi: heading_error -= 2 * np.pi
        while heading_error < -np.pi: heading_error += 2 * np.pi
        
        state_laser = []
        num_readings = 60
        len_scan = len(self.latest_scan)
        for i in range(num_readings):
            idx_start = int(i * (len_scan / num_readings))
            idx_end = int((i + 1) * (len_scan / num_readings))
            val = np.min(self.latest_scan[idx_start:idx_end]) if len_scan >= 360 else 3.5
            state_laser.append(val)
        state = state_laser + [dist, heading_error]
        state = np.array(state, dtype=np.float32)

        # --- 4. 奖励函数 (包含高压电场) ---
        reward = 0.0
        done = False
        
        # (A) 核心距离奖励
        reward += (self.current_dist - dist) * 60.0 
        self.current_dist = dist 
        
        # (B) 角度修正奖励
        reward -= abs(heading_error) * 1.0

        # (C) 动作逻辑: 先转正，再跑
        if not override_action:
            if abs(heading_error) < 0.35:
                reward += v * 2.0 
                reward -= abs(w) * 0.1 
            else:
                reward -= v * 2.0
        
        # (D) 【核心修改】斥力场惩罚 (Repulsive Force) 🔥
        # 只要距离小于 0.45米，就开始扣分
        # 比如：距离 0.2m 时，penalty = (0.45 - 0.2) * 20 = 5.0 (非常痛！)
        # 这比往前跑赚的那点分 (0.4分) 高多了，它绝对不敢靠近
        if min_dist < 0.45:
            p_factor = 20.0
            reward -= (0.45 - min_dist) * p_factor

        # (E) 撞墙/屏蔽惩罚
        if override_action:
            reward -= 0.5
        if min_dist < 0.12:
            reward = -100.0
            done = True
            print(f"💥 撞墙重置")
            
        # (F) 成功奖励
        if dist < 0.25:
            reward = 100.0
            done = True
            print("🚩 成功到达")
            
        reward -= 0.01 

        return state, reward, done, {}

    def reset(self):
        # 保持你原来的 reset 代码不变
        # ------------------------
        print("🔄 重置仿真...")
        while not self.reset_client.wait_for_service(timeout_sec=1.0):
            print('等待 Gazebo 服务...')
        req = Empty.Request()
        future = self.reset_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        self.pub_cmd_vel.publish(Twist())
        for _ in range(10):
            rclpy.spin_once(self, timeout_sec=0.1)
        rand_dist = np.random.uniform(0.5, 2.0)
        rand_angle = np.random.uniform(-np.pi, np.pi)
        self.goal_x = self.robot_x + rand_dist * math.cos(rand_angle)
        self.goal_y = self.robot_y + rand_dist * math.sin(rand_angle)
        self.current_dist = math.sqrt((self.goal_x - self.robot_x)**2 + (self.goal_y - self.robot_y)**2)
        print(f"🎯 新目标: Dist={rand_dist:.2f}m")
        return np.zeros(62, dtype=np.float32)
