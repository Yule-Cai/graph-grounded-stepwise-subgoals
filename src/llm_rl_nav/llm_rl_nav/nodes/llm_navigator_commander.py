import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import numpy as np
import math
import time
import threading
import os
from llama_cpp import Llama

class LLMNavigatorCommanderV2(Node):
    def __init__(self):
        super().__init__('llm_navigator_commander_v2')

        self.declare_parameter("final_goal_x", 15.0)
        self.declare_parameter("final_goal_y", 15.0)
        self.declare_parameter("model_path", "")
        self.declare_parameter("n_ctx", 2048)
        self.declare_parameter("n_threads", 6)
        self.declare_parameter("n_gpu_layers", -1)
        
        # 初始目标
        self.final_goal_x = float(self.get_parameter("final_goal_x").value)
        self.final_goal_y = float(self.get_parameter("final_goal_y").value)
        self.virtual_target_x = self.final_goal_x
        self.virtual_target_y = self.final_goal_y
        self.strategy = "GOAL"
        
        # 机器人状态
        self.robot_x = 0.0; self.robot_y = 0.0; self.robot_yaw = 0.0; self.current_v = 0.0
        
        # === 📡 双通道监听 (接收 Rviz2 点击指令) ===
        self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)
        self.create_subscription(PoseStamped, '/move_base_simple/goal', self.goal_callback, 10)
        
        # 敢死队参数
        self.REFLEX_THRESHOLD = 0.25
        self.PHYSICS_RANGE = 0.70
        self.MAX_SPEED = 0.22
        
        # QoS
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10, durability=DurabilityPolicy.VOLATILE)
        self.sub_scan = self.create_subscription(LaserScan, '/scan', self.scan_callback, qos)
        self.sub_odom = self.create_subscription(Odometry, '/odom', self.odom_callback, qos)
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)
        
        self.get_logger().info("[Commander V2] Listening for RViz map clicks.")
        self.get_logger().info("Use RViz2 '2D Goal Pose' to set a target.")

        # 模型加载
        configured_model = str(self.get_parameter("model_path").value).strip()
        self.model_path = configured_model or os.environ.get("LLM_MODEL_PATH", "").strip()
        if not self.model_path or not os.path.exists(self.model_path):
            raise FileNotFoundError(
                "LLM model not found. Set the ROS parameter 'model_path' or export "
                "LLM_MODEL_PATH=/absolute/path/to/model.gguf"
            )
        
        self.llm = Llama(
            model_path=self.model_path,
            n_gpu_layers=int(self.get_parameter("n_gpu_layers").value),
            n_ctx=int(self.get_parameter("n_ctx").value),
            n_threads=int(self.get_parameter("n_threads").value),
            verbose=False,
        )
        self.get_logger().info("LLM brain is ready.")
        
        self.latest_scan = None
        self.timer_fast = self.create_timer(0.05, self.control_loop) 
        self.brain_thread = threading.Thread(target=self.brain_loop)
        self.brain_thread.daemon = True; self.brain_thread.start()

    def goal_callback(self, msg):
        new_x = msg.pose.position.x
        new_y = msg.pose.position.y
        self.get_logger().info(f"New RViz target: ({new_x:.2f}, {new_y:.2f})")
        
        self.final_goal_x = new_x
        self.final_goal_y = new_y
        self.virtual_target_x = new_x
        self.virtual_target_y = new_y
        self.strategy = "GOAL"

    def scan_callback(self, msg):
        self.latest_scan = np.array(msg.ranges)
        self.latest_scan = np.nan_to_num(self.latest_scan, posinf=10.0, nan=10.0)
        self.scan_angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))

    def odom_callback(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y); cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.robot_yaw = math.atan2(siny_cosp, cosy_cosp)
        self.current_v = msg.twist.twist.linear.x

    def brain_loop(self):
        while rclpy.ok():
            if self.latest_scan is None: time.sleep(0.1); continue
            
            # 感知
            scan = self.latest_scan
            f = np.min(np.concatenate((scan[0:30], scan[330:360])))
            l = np.min(scan[45:135]); r = np.min(scan[225:315])
            
            # 罗盘
            dx = self.final_goal_x - self.robot_x; dy = self.final_goal_y - self.robot_y
            goal_angle = math.atan2(dy, dx)
            heading_err = goal_angle - self.robot_yaw
            while heading_err > math.pi: heading_err -= 2*math.pi
            while heading_err < -math.pi: heading_err += 2*math.pi
            
            goal_desc = "FRONT"
            if heading_err > 0.6: goal_desc = "LEFT"
            elif heading_err < -0.6: goal_desc = "RIGHT"
            
            is_stuck = (abs(self.current_v) < 0.02) and (self.strategy == "GOAL")
            condition = "BLOCKED" if (f < 0.7 or is_stuck) else "NORMAL"
            
            prompt = f"Pos:[{self.robot_x:.1f},{self.robot_y:.1f}]. Goal:{goal_desc}({heading_err:.1f}). Obs:[F:{f:.1f} L:{l:.1f} R:{r:.1f}]. Stat:{condition}. Task:Go to ({self.final_goal_x:.1f},{self.final_goal_y:.1f}). Choice(GOAL/LEFT/RIGHT/BACK):"
            
            try:
                output = self.llm.create_completion(prompt=f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n", max_tokens=3, temperature=0.1)
                text = output['choices'][0]['text'].upper()
                new_strat = "GOAL"
                if "LEFT" in text: new_strat = "LEFT"
                elif "RIGHT" in text: new_strat = "RIGHT"
                elif "BACK" in text: new_strat = "BACK"
                
                # 强制修正
                if "LEFT" in goal_desc and l > 0.6 and new_strat == "RIGHT": new_strat = "LEFT"
                if "RIGHT" in goal_desc and r > 0.6 and new_strat == "LEFT": new_strat = "RIGHT"
                self.strategy = new_strat
                
                # 移动胡萝卜
                offset = 4.0; target_angle = self.robot_yaw
                if new_strat == "GOAL": self.virtual_target_x = self.final_goal_x; self.virtual_target_y = self.final_goal_y
                else:
                    if new_strat == "LEFT": target_angle += 1.0
                    elif new_strat == "RIGHT": target_angle -= 1.0
                    elif new_strat == "BACK": target_angle += 3.14
                    self.virtual_target_x = self.robot_x + offset * math.cos(target_angle); self.virtual_target_y = self.robot_y + offset * math.sin(target_angle)
            except: pass
            time.sleep(0.4)

    def control_loop(self):
        if self.latest_scan is None: return
        scan = self.latest_scan; min_dist = np.min(scan)
        final_v = 0.0; final_w = 0.0

        if min_dist < self.REFLEX_THRESHOLD:
            fl = np.mean(scan[30:90]); fr = np.mean(scan[270:330])
            final_w = (fl - fr) * 6.0; final_v = -0.10
        else:
            dx = self.virtual_target_x - self.robot_x; dy = self.virtual_target_y - self.robot_y
            dist = math.sqrt(dx**2 + dy**2)
            
            if dist < 0.3: final_v = 0.0; final_w = 0.0
            else:
                u_x, u_y = (dx/dist, dy/dist) if dist > 0 else (0,0)
                f_att_x = u_x * 4.5; f_att_y = u_y * 4.5
                
                f_rep_x = 0.0; f_rep_y = 0.0
                mask = scan < self.PHYSICS_RANGE; valid_scan = scan[mask]; valid_angles = self.scan_angles[mask]
                
                if len(valid_scan) > 0:
                    mags = 0.4 / (valid_scan**2 + 0.01)
                    rep_x = -np.cos(valid_angles); rep_y = -np.sin(valid_angles)
                    flow_dir = 1.0 if self.strategy != "RIGHT" else -1.0
                    vor_x = -rep_y * flow_dir; vor_y = rep_x * flow_dir
                    mix_x = rep_x * 0.4 + vor_x * 0.6; mix_y = rep_y * 0.4 + vor_y * 0.6
                    f_rep_x = np.sum(mix_x * mags); f_rep_y = np.sum(mix_y * mags)
                    cy = math.cos(self.robot_yaw); sy = math.sin(self.robot_yaw)
                    g_rep_x = f_rep_x * cy - f_rep_y * sy; g_rep_y = f_rep_x * sy + f_rep_y * cy
                    f_total_x = f_att_x + g_rep_x; f_total_y = f_att_y + g_rep_y
                else: f_total_x, f_total_y = f_att_x, f_att_y
                
                local_fx = f_total_x * math.cos(self.robot_yaw) + f_total_y * math.sin(self.robot_yaw)
                local_fy = -f_total_x * math.sin(self.robot_yaw) + f_total_y * math.cos(self.robot_yaw)
                final_v = 0.15 * local_fx; final_w = 2.5 * local_fy

        final_v = np.clip(final_v, -0.15, self.MAX_SPEED); final_w = np.clip(final_w, -1.8, 1.8)
        cmd = Twist(); cmd.linear.x = float(final_v); cmd.angular.z = float(final_w)
        self.pub_cmd_vel.publish(cmd)

def main():
    rclpy.init(); node = LLMNavigatorCommanderV2()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__': main()
