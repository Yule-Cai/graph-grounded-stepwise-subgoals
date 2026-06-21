import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SetEntityState
from geometry_msgs.msg import Pose, Point, Quaternion
import math
import time
import sys

class CrowdControllerV2(Node):
    def __init__(self):
        super().__init__('crowd_controller_v2')
        
        # === 关键修改：使用 Client 连接 Gazebo 服务 ===
        self.client = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        
        # 等待服务上线 (如果 Gazebo 没启动，这里会卡住等待)
        print("⏳ 正在连接 Gazebo 服务...")
        if not self.client.wait_for_service(timeout_sec=5.0):
            print("❌ 错误：找不到 /gazebo/set_entity_state 服务！")
            print("请检查：1. Gazebo 是否已启动？ 2. crowd.world 是否加载了 libgazebo_ros_state.so？")
            # 正常情况下不应退出，但在调试脚本中为了提示用户，我们可以退出
            return 

        print("✅ 服务已连接！人群开始运动！")
        
        # 使用 Timer 定时发送移动请求 (20Hz)
        self.timer = self.create_timer(0.05, self.update_positions)
        self.start_time = time.time()

    def move_model(self, name, x, y):
        # 构建服务请求
        req = SetEntityState.Request()
        req.state.name = name
        req.state.pose.position = Point(x=float(x), y=float(y), z=0.25)
        # 保持直立，不旋转
        req.state.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        # 必须指定参考系 (world)
        req.state.reference_frame = 'world'
        
        # 异步发送请求 (不等待结果，防止卡顿)
        self.client.call_async(req)

    def update_positions(self):
        t = time.time() - self.start_time
        
        # 1. 守门员 (红色): 快速横跳
        self.move_model('human_0', 1.5, 1.5 * math.sin(t * 1.5))

        # 2. 巡逻者 (蓝色): 反向横跳
        self.move_model('human_1', 2.5, -1.5 * math.sin(t * 1.2))

        # 3. 绕圈者 (绿色): 中心画圆
        self.move_model('human_2', 3.5 + 0.8 * math.cos(t * 0.8), 0.8 * math.sin(t * 0.8))

        # 4. 纵向干扰 (黄色)
        self.move_model('human_3', 3.0 + 1.0 * math.sin(t * 0.5), 1.5)

        # 5. 纵向干扰 (紫色)
        self.move_model('human_4', 3.0 + 1.0 * math.cos(t * 0.5), -1.5)

def main():
    rclpy.init()
    node = CrowdControllerV2()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
