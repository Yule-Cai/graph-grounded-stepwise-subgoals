import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32MultiArray
from openai import OpenAI
import json
import os
import traceback

class LLMPlanner(Node):
    def __init__(self):
        super().__init__('llm_planner')

        self.declare_parameter("lm_studio_url", "http://localhost:1234/v1")
        self.declare_parameter("model_name", os.environ.get("LM_STUDIO_MODEL", "local-model"))
        
        # --- 配置部分 ---
        # 确保你的 LM Studio 服务器端口是 1234 (默认)
        self.lm_studio_url = str(self.get_parameter("lm_studio_url").value)
        self.model_name = str(self.get_parameter("model_name").value)
        
        # 1. 连接到 LM Studio 本地服务器
        try:
            self.client = OpenAI(
                base_url=self.lm_studio_url, 
                api_key="lm-studio" # 本地运行不需要真实 key
            )
            self.get_logger().info(f"已连接到 LM Studio: {self.lm_studio_url}")
        except Exception as e:
            self.get_logger().error(f"连接 LM Studio 失败，请检查 Server 是否开启: {e}")

        # 2. 定义 ROS 通信
        # 订阅用户指令
        self.subscription = self.create_subscription(
            String, 'user_command', self.command_callback, 10)
        
        # 发布目标坐标给 RL 节点 (格式: [x, y])
        self.publisher_ = self.create_publisher(Float32MultiArray, 'rl_target', 10)
        
        self.get_logger().info("LLM Brain is ready! Waiting for commands on topic '/user_command'...")

    def command_callback(self, msg):
        user_input = msg.data
        self.get_logger().info(f"--------------------------------")
        self.get_logger().info(f"收到指令: {user_input}")
        
        # 3. 构造 Prompt (提示词工程)
        system_prompt = """
        你是一个机器人导航助手。已知地图坐标如下：
        - 厨房 (Kitchen): [5.0, 5.0]
        - 卧室 (Bedroom): [2.0, -2.0]
        - 门口 (Door): [0.0, 0.0]
        - 实验室 (Lab): [8.0, 3.0]
        
        请根据用户指令，输出目标坐标的 JSON 格式。
        规则：
        1. 必须严格输出 JSON 格式: {"target": [x, y]}
        2. 不要输出任何解释、前缀或后缀文字。
        3. 如果找不到对应地点，请输出: {"target": null}
        """
        
        # 4. 调用 LM Studio
        try:
            response = self.client.chat.completions.create(
                model=self.model_name, 
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ],
                temperature=0.1, # 低温度，让输出更稳定
                max_tokens=50    # 限制输出长度，防止模型废话，加快速度
            )
            
            content = response.choices[0].message.content
            
            # --- 调试打印: 看看模型到底说了什么 ---
            self.get_logger().info(f"======> LM Studio 原始返回: [{content}]")
            
            if not content:
                self.get_logger().warn("警告：模型返回了空内容！")
                return

            # 5. 数据清洗与解析
            # 有时候模型会输出 ```json ... ```，需要去掉
            clean_content = content.replace("```json", "").replace("```", "").strip()
            
            try:
                data = json.loads(clean_content)
            except json.JSONDecodeError:
                self.get_logger().error(f"解析失败: 模型返回的不是有效 JSON。内容: {clean_content}")
                return

            target = data.get("target")
            
            # 6. 安全检查与发布
            if target is None:
                self.get_logger().warn(f"未识别到有效地点或 target 为 null。数据: {data}")
            elif isinstance(target, list) and len(target) == 2:
                # 只有当 target 是列表且有两个元素时才发布
                out_msg = Float32MultiArray()
                out_msg.data = [float(target[0]), float(target[1])]
                self.publisher_.publish(out_msg)
                self.get_logger().info(f"成功发布目标点: {target}")
            else:
                self.get_logger().error(f"坐标格式错误，应为 [x, y]。实际收到: {target}")
                
        except Exception as e:
            self.get_logger().error(f"处理过程中发生未知错误: {e}")
            self.get_logger().error(traceback.format_exc())

def main(args=None):
    rclpy.init(args=args)
    node = LLMPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
